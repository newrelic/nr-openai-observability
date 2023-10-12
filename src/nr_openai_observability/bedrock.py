import botocore
import json
import logging
import newrelic.agent
import sys
import time
import traceback
import uuid

from datetime import datetime
from nr_openai_observability.monitor import _patched_call, monitor, MessageEventName, SummaryEventName, TransactionBeginEventName
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.call_vars import set_ai_message_ids, create_ai_message_id


logger = logging.getLogger("nr_openai_observability")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

def perform_patch_bedrock():
    try:
        logger.info('Josh --- Attempting to patch Bedrock invoke_model')
        botocore.client.ClientCreator.create_client = _patched_call(
            botocore.client.ClientCreator.create_client, patcher_aws_create_api
        )
    except AttributeError as error:
        logger.error(f'failed to instrument botocore.client.ClientCreator.create_client: {error}')


def patcher_aws_create_api(original_fn, *args, **kwargs):
    # We are in the AWS API to create a new object, we need to invoke that first!
    try:
        from botocore.model import ServiceModel
        logger.info(f'Josh --- invoking method {original_fn} with args {args}, kwargs {kwargs}')
        response = original_fn(*args, **kwargs)
        logger.info(f'Josh --- original_fn {original_fn} returned {response}')

        cls = type(response)
        name = cls.__qualname__
        if cls.__module__ is not None and cls.__module__ != '__builtin__':
            name = f'{cls.__module__}.{cls.__qualname__}'

        logger.info(f'Josh --- returned object is of type {name}')

        if name == 'botocore.client.BedrockRuntime':
            bedrock_method = 'invoke_model'
            logger.info(f'Josh --- instrument bedrock class {bedrock_method}!')
            original_invoke_model = getattr(response, bedrock_method)

            if original_invoke_model is not None:
                logger.info(f"Josh --- found method '{bedrock_method}' on {response}")
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
    logger.info(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    result, time_delta = None, None

    try:
        with newrelic.agent.FunctionTrace(
            name="AI/Bedrock/Chat/Completions/Create", group="", terminal=True
        ):
            result = original_fn(*args, **kwargs)
            time_delta = time.time() - timestamp

            logger.info(f"Finished running function: '{original_fn.__qualname__}' in {time_delta}.")
            logger.info(f"Josh ---  returned result '${result}'")
    except Exception as error:
        logger.error(f"error invoking bedrock function: {error}")
        return result

    # print the HTTP body
    from botocore.response import StreamingBody
    from io import BytesIO

    contents = result['body'].read()
    bio = BytesIO(contents)
    result['body'] = StreamingBody(bio, len(contents))
    logger.info(f'Josh --- result body {contents}')

    handle_bedrock_create_completion(result, time_delta, **kwargs)

    # we have to reset the body after we read it. The handle function is going to read the contents,
    # and we'll have to apply the body again before returning back.
    bio = BytesIO(contents)
    result['body'] = StreamingBody(bio, len(contents))
    return result


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

        logger.info(f'handle_bedrock_create_completion: response {response},\n\ttime_delta {time_delta},\n\tkwargs {kwargs},\n\tcontents {contents}')

        # convert body from a str to a json dictionary
        if 'body' in event_dict and type(event_dict['body']) is str:
            body = json.loads(event_dict['body'])
            event_dict['body'] = body

        (summary, messages, transaction_begin_event) = build_bedrock_events(response, event_dict, time_delta)

        logger.info(f"Bedrock Reported event dictionary:\n{event_dict}\n")
        logger.info(f'Bedrock summary event: {summary}')
        for event in messages:
            monitor.record_event(event, MessageEventName)
        monitor.record_event(summary, SummaryEventName)
        monitor.record_event(transaction_begin_event, TransactionBeginEventName)

    except Exception as error:
        stacks = traceback.format_exception(error)
        logger.error(f'error writing bedrock event summary: {error}')
        logger.error(stacks)


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

    logger.info(f'\n\nevent_dict = {event_dict}')

    logger.info("parsed Bedrock info:")
    logger.info(f"\tinput_message = {input_message[:30]}")
    logger.info(f"\tinput_tokens = {input_tokens}")
    logger.info(f"\tresponse_tokens = {response_tokens}")


    if 'modelId' in event_dict:
        completion_id = newrelic.agent.current_span_id() or str(uuid.uuid4())
        model = event_dict['modelId']
        vendor = 'bedrock'
        message_id = str(uuid.uuid4())
        tokens = input_tokens

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
                    logger.info(f"\tresponse_message = {messages[-1]['content'][:30]}")
                    logger.info(f"\tcompletion_reason = {messages[-1]['stop_reason']}")
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
                logger.info(f"\tresponse_message = {messages[-1]['content'][:30]}")
                logger.info(f"\tcompletion_reason = {messages[-1]['stop_reason']}")
        elif 'claude' in model:
            messages.append(
                build_bedrock_result_message(
                    completion_id=completion_id,
                    message_id=message_id,
                    content=event_dict['completion'],
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
                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=message_id,
                        content=result['data']['text'],
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
            ai_message_id = create_ai_message_id(messages[-1].get("message_id"))
            set_ai_message_ids([ai_message_id])
            

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
        if 'ai21.j2' in model:
            input_message = event_dict['prompt.text']
            default_temp = 0.5
            default_max_tokens = 200

            for result in event_dict['completions']:
                stop_reason = result['finishReason']['reason'] # keep the last one
        if 'cohere.command' in model:
            input_message = event_dict['prompt']
            default_temp = 0.9
            default_max_tokens = 20

    return (input_message, input_tokens, response_tokens, stop_reason, default_temp, default_max_tokens)


def bind__create_api_method(py_operation_name, operation_name, service_model,
        *args, **kwargs):
    return (py_operation_name, service_model)
