import botocore
import json
import logging
import newrelic.agent
import sys
import time
import uuid

from anthropic import _tokenizers
from datetime import datetime
from nr_openai_observability.monitor import _patched_call, monitor, EventName, MessageEventName, SummaryEventName, TransactionBeginEventName
from nr_openai_observability.error_handling_decorator import handle_errors


logger = logging.getLogger("nr_openai_observability")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

def perform_patch_bedrock():
    try:
        botocore.client.ClientCreator.create_client = _patched_call(
            botocore.client.ClientCreator.create_client, patcher_aws_create_api
        )
    except AttributeError as error:
        logger.error(f'failed to instrument botocore.client.ClientCreator.create_client: {error}')


def patcher_aws_create_api(original_fn, *args, **kwargs):
    # We are in the AWS API to create a new object, we need to invoke that first!
    try:
        from botocore.model import ServiceModel
        response = original_fn(*args, **kwargs)

        cls = type(response)
        name = cls.__qualname__
        if cls.__module__ is not None and cls.__module__ != '__builtin__':
            name = f'{cls.__module__}.{cls.__qualname__}'

        if name == 'botocore.client.BedrockRuntime':
            bedrock_method = 'invoke_model'
            original_invoke_model = getattr(response, bedrock_method)

            if original_invoke_model is not None:
                setattr(response, bedrock_method, _patched_call(
                    original_invoke_model, 
                    patcher_bedrock_create_completion
                ))
            else:
                logger.error(f"failed to find method '{bedrock_method}' on {response}")

        return response

    except BaseException as error:
        logger.error(f'caught exception trying to invoke original function: {error}')
        raise error


def patcher_bedrock_create_completion(original_fn, *args, **kwargs):
    timestamp = time.time()
    result, time_delta = None, None

    try:
        with newrelic.agent.FunctionTrace(
            name="AI/Bedrock/Chat/Completions/Create", group="", terminal=True
        ):
            result = original_fn(*args, **kwargs)
            time_delta = time.time() - timestamp
    except Exception as error:
        build_completion_summary_for_error(error, **kwargs)
        raise result

    # print the HTTP body
    from botocore.response import StreamingBody
    from io import BytesIO

    contents = result['body'].read()
    bio = BytesIO(contents)
    result['body'] = StreamingBody(bio, len(contents))

    handle_bedrock_create_completion(result, time_delta, **kwargs)

    # we have to reset the body after we read it. The handle function is going to read the contents,
    # and we'll have to apply the body again before returning back.
    bio = BytesIO(contents)
    result['body'] = StreamingBody(bio, len(contents))
    return result


def build_completion_summary_for_error(error, **kwargs):
    logger.error(f"error invoking bedrock function: {error}")

    completion = {
        "id": str(uuid.uuid4()),
        "vendor": "bedrock",
        "ingest_source": "PythonSDK",
        "error_status": error.http_status,
        "error_message": error.error.message,
        "error_type": error.error.type,
        "error_code": error.error.code,
        "error_param": error.error.param,
    }

    monitor.record_event(completion, EventName)

@handle_errors
def handle_bedrock_create_completion(response, time_delta, **kwargs):
    from nr_openai_observability.monitor import flatten_dict
    try:
        contents = response['body'].read()
        response_body = json.loads(contents)

        event_dict = {
            **kwargs,
            "response_time": time_delta,
            **flatten_dict(response_body, separator="."),
            'vendor': 'bedrock',
        }

        if 'credentials' in event_dict:
            event_dict.pop('credentials')

        # convert body from a str to a json dictionary
        if 'body' in event_dict and type(event_dict['body']) is str:
            body = json.loads(event_dict['body'])
            event_dict['body'] = body

        (summary, messages, transaction_begin_event) = build_bedrock_events(response, event_dict, time_delta)

        for event in messages:
            monitor.record_event(event, MessageEventName)
        monitor.record_event(summary, SummaryEventName)
        monitor.record_event(transaction_begin_event, TransactionBeginEventName)

    except Exception as error:
        build_completion_summary_for_error(error, **kwargs)
        raise error

def build_bedrock_events(response, event_dict, time_delta):
    """
    returns (summary_event, list(message_events), transaction_event)
    """
    (input_message, input_tokens, response_tokens, stop_reason, temperature, max_tokens) = get_bedrock_info(event_dict)
    summary = {}
    messages = []
    model = "bedrock-unknown"
    trace_id = newrelic.agent.current_trace_id()
    transaction_id = (
        newrelic.agent.current_transaction().guid
        if newrelic.agent.current_transaction() != None
        else None
    )

    if 'modelId' in event_dict:
        completion_id = newrelic.agent.current_span_id() or str(uuid.uuid4())
        model = event_dict['modelId']
        vendor = 'bedrock'
        message_id = str(uuid.uuid4())
        tokens = input_tokens or 0

        if tokens and response_tokens:
            tokens += response_tokens

        if 'vendor' in event_dict:
            vendor = event_dict['vendor']

        if 'temperature' in event_dict['body']:
            temperature = event_dict['body']['temperature']

        if 'body' in event_dict and 'max_tokens_to_sample' in event_dict['body']:
            max_tokens = event_dict['body']['max_tokens_to_sample']

        if 'ResponseMetadata' in response and 'RequestId' in response['ResponseMetadata']:
            # TODO Is this general to all Bedrock LLMs? Can we move it up?
            message_id = response['ResponseMetadata']['RequestId']

        # input message
        messages.append(
            build_bedrock_result_message(
                completion_id=completion_id,
                message_id=message_id,
                content=input_message[:4095],
                tokens=input_tokens,
                role='user',
                sequence=len(messages),
                model=model,
                vendor=vendor,
                trace_id=trace_id
            )
        )

        if 'titan' in model:
            # handle 1 or more response messages
            if isinstance(event_dict['results'], list):
                for result in event_dict['results']:
                    messages.append(
                        build_bedrock_result_message(
                            completion_id=completion_id,
                            message_id=message_id,
                            content=result['outputText'],
                            tokens=result['tokenCount'],
                            role='assistant',
                            sequence=len(messages),
                            stop_reason=result['completionReason'],
                            model=model,
                            vendor=vendor,
                            trace_id=trace_id
                        )
                    )
            else: # TODO Is this actually a case? Or is it always a list?
                result = event_dict['results']
                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=message_id,
                        content=result['outputText'],
                        tokens=result['tokenCount'],
                        role='assistant',
                        sequence=len(messages),
                        stop_reason=result['completionReason'],
                        model=model,
                        vendor=vendor,
                        trace_id=trace_id
                    )
                )
        elif 'claude' in model:
            tokenizer = _tokenizers.sync_get_tokenizer()
            encoded = tokenizer.encode(event_dict['completion'])
            tokens += len(encoded)
            messages.append(
                build_bedrock_result_message(
                    completion_id=completion_id,
                    message_id=message_id,
                    content=event_dict['completion'],
                    tokens=len(encoded),
                    role='assistant',
                    sequence=len(messages),
                    stop_reason=event_dict['stop_reason'],
                    model=model,
                    vendor=vendor,
                    trace_id=trace_id
                )
            )
        elif 'ai21.j2' in model:
            for result in event_dict['completions']:
                message_tokens = 0
                if 'data' in result and 'tokens' in result['data']:
                    message_tokens = len(result['data']['tokens'])

                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=message_id,
                        content=result['data']['text'],
                        tokens=message_tokens,
                        role='assistant',
                        sequence=len(messages),
                        stop_reason=result['finishReason']['reason'],
                        model=model,
                        vendor=vendor,
                        trace_id=trace_id
                    )
                )
        elif 'cohere.command' in model:
            for result in event_dict['generations']:
                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=message_id,
                        content=result['text'],
                        role='assistant',
                        sequence=len(messages),
                        model=model,
                        vendor=vendor,
                        trace_id=trace_id
                    )
                )

        if len(messages) > 0:
            messages[-1]["is_final_response"] = True

        summary = {
            "id": completion_id,
            "timestamp": datetime.now(),
            "response_time": int(time_delta * 1000),
            "model": model,
            "request.model": model,
            "response.model": model,
            "temperature": temperature,
            "api_type": None,
            "vendor": vendor,
            "ingest_source": "PythonSDK",
            "number_of_messages": len(messages), 
            "trace.id": trace_id,
            "transactionId": transaction_id,
            "response": messages[-1]['content'][:4095],
        }

        if stop_reason:
            summary["finish_reason"] = stop_reason
        if response_tokens:
            summary["usage.completion_tokens"] = response_tokens
        if tokens:
            summary["usage.total_tokens"] = tokens
        if input_tokens:
            summary["usage.prompt_tokens"] = input_tokens
        if max_tokens:
            summary["max_tokens"] = max_tokens

        transaction_begin_event = {
            "human_prompt": messages[0]['content'],
            "vendor": vendor,
            "trace.id": trace_id,
            "ingest_source": "PythonAgentHybrid"
        }

    return (summary, messages, transaction_begin_event)



def build_bedrock_result_message(completion_id, message_id, content, tokens=None, role=None, sequence=None, stop_reason=None, model=None, vendor=None, trace_id=None):
    message = {
        "id": message_id,
        "content": content[:4095],
        "role": role,
        "completion_id": completion_id,
        "sequence": sequence,
        "model": model,
        "vendor": vendor,
        "ingest_source": "PythonSDK",
    }

    if tokens:
        message["tokens"] = tokens
    if stop_reason:
        message["stop_reason"] = stop_reason
    if trace_id:
        message["trace.id"] = trace_id

    return message

def get_bedrock_info(event_dict):
    """
    (input_message, input_tokens, response_tokens, completion_reason, default_temp, max_tokens) =

    default temperature and max tokens per model was found at https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
    """
    (input_message, input_tokens, response_tokens, stop_reason, default_temp, default_max_tokens) = (None, None, None, None, 0, 0)

    if 'modelId' in event_dict:
        model = event_dict['modelId']

        if 'titan' in model:
            response_tokens = 0
            default_temp = 0
            default_max_tokens = 512

            if isinstance(event_dict['results'], list):
                for result in event_dict['results']:
                    response_tokens += result['tokenCount']
                    stop_reason = result['completionReason'] # keep the last one

            input_message = event_dict['body']['inputText']
            input_tokens = event_dict['inputTextTokenCount']
        if 'claude' in model:
            input_message = event_dict['body']['prompt']
            stop_reason = event_dict['stop_reason']
            default_temp = 0.5
            default_max_tokens = 200

            tokenizer = _tokenizers.sync_get_tokenizer()
            encoded = tokenizer.encode(input_message)
            input_tokens = len(encoded)
        if 'ai21.j2' in model:
            input_message = event_dict['prompt.text']
            input_tokens = len(event_dict['prompt.tokens'])
            response_tokens = 0
            default_temp = 0.5
            default_max_tokens = 200

            for result in event_dict['completions']:
                stop_reason = result['finishReason']['reason'] # keep the last one

                if 'data' in result and 'tokens' in result['data']:
                    response_tokens += len(result['data']['tokens'])
        if 'cohere.command' in model:
            input_message = event_dict['prompt']
            default_temp = 0.9
            default_max_tokens = 20

    return (input_message, input_tokens, response_tokens, stop_reason, default_temp, default_max_tokens)


def bind__create_api_method(py_operation_name, operation_name, service_model,
        *args, **kwargs):
    return (py_operation_name, service_model)
