import botocore
import json
import logging
import newrelic.agent
import sys
import time
import traceback
import uuid

from datetime import datetime
from nr_openai_observability.monitor import _patched_call, monitor, MessageEventName, SummaryEventName
from nr_openai_observability.error_handling_decorator import handle_errors


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
            name="AI/Bedrock/Chat/Completions/Create", group=""
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

        (summary, messages) = build_bedrock_events(response, event_dict, time_delta)

        logger.info(f"Bedrock Reported event dictionary:\n{event_dict}\n")
        logger.info(f'Bedrock summary event: {summary}')
        for event in messages:
            monitor.record_event(event, MessageEventName)
        monitor.record_event(summary, SummaryEventName)
    except Exception as error:
        stacks = traceback.format_exception(error)
        logger.error(f'error writing bedrock event summary: {error}')
        logger.error(stacks)


def build_bedrock_events(response, event_dict, time_delta):
    """
    returns (summary_event, list(message_events))
    """
    (input_message, input_tokens, response_tokens, stop_reason) = get_bedrock_info(event_dict)
    summary = {}
    messages = []
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
        model = event_dict['modelId'] or "bedrock-unknown"
        temperature = 0
        vendor = 'bedrock'
        max_tokens = 0
        tokens = input_tokens

        if 'vendor' in event_dict:
            vendor = event_dict['vendor']

        if 'temperature' in event_dict['body']:
            temperature = event_dict['body']['temperature']

        if 'body' in event_dict and 'max_tokens_to_sample' in event_dict['body']:
            max_tokens = event_dict['body']['max_tokens_to_sample']


        if 'titan' in event_dict['modelId']:
            # build out the input and output messages for this request
            message_id = str(uuid.uuid4())
            if 'ResponseMetadata' in response and 'RequestId' in response['ResponseMetadata']:
                # TODO Is this general to all Bedrock LLMs? Can we move it up?
                message_id = response['ResponseMetadata']['RequestId']

            messages.append( # input message
                build_bedrock_result_message(
                    completion_id=completion_id,
                    message_id=message_id,
                    content=input_message[:4095],
                    tokens=input_tokens,
                    role='system',
                    sequence=len(messages),
                    model=model,
                    vendor=vendor
                )
            )

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
                            vendor=vendor
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
                        vendor=vendor
                    )
                )
                logger.info(f"\tresponse_message = {messages[-1]['content'][:30]}")
                logger.info(f"\tcompletion_reason = {messages[-1]['stop_reason']}")

            summary = {
                "id": completion_id,
                "timestamp": datetime.now(),
                "response_time": int(time_delta * 1000),
                "request.model": model,
                "response.model": model,
                "usage.completion_tokens": response_tokens,
                "usage.total_tokens": tokens,
                "usage.prompt_tokens": input_tokens,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "finish_reason": stop_reason,
                "api_type": None,
                "vendor": vendor,
                "ingest_source": "PythonSDK",
                "number_of_messages": len(messages), 
                "trace.id": trace_id,
                "transactionId": transaction_id,
                "response": messages[-1]['content'][:4095],
                # "organization": response.organization,
                # "api_version": response_headers.get("openai-version"),
            }

    return (summary, messages)



def build_bedrock_result_message(completion_id, message_id, content, tokens=None, role=None, sequence=None, stop_reason=None, model=None, vendor=None):
    return {
        "id": message_id,
        "content": content[:4095],
        "tokens": tokens,
        "role": role,
        "completion_id": completion_id,
        "sequence": sequence,
        "stop_reason": stop_reason,
        "model": model,
        "vendor": vendor,
        "ingest_source": "PythonSDK",
    }

def get_bedrock_info(event_dict):
    """
    (input_message, input_tokens, response_tokens, completion_reason) = 
    """
    (input_message, input_tokens, response_tokens, stop_reason) = (None, None, None, None)

    if 'modelId' in event_dict:
        if 'titan' in event_dict['modelId']:
            response_tokens = 0

            if isinstance(event_dict['results'], list):
                for result in event_dict['results']:
                    response_tokens += result['tokenCount']
                    stop_reason = result['completionReason'] # keep the last one

            input_message = event_dict['body']['inputText']
            input_tokens = event_dict['inputTextTokenCount']
        if 'claude' in event_dict['modelId']:
            pass
        if 'ai21.j2' in event_dict['modelId']:
            pass

    return (input_message, input_tokens, response_tokens, stop_reason)


def bind__create_api_method(py_operation_name, operation_name, service_model,
        *args, **kwargs):
    return (py_operation_name, service_model)
