import json
import logging
import newrelic.agent
import sys
import time
import uuid

from datetime import datetime

from nr_openai_observability.build_events import compat_fields, get_trace_details
from nr_openai_observability.monitor import monitor
from nr_openai_observability.patcher import patched_call
from nr_openai_observability.consts import (
    EmbeddingEventName,
    MessageEventName,
    SummaryEventName,
    TransactionBeginEventName,
)
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.call_vars import (
    set_ai_message_ids,
    create_ai_message_id,
    get_conversation_id,
)


logger = logging.getLogger("nr_openai_observability")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def patcher_aws_create_api(original_fn, *args, **kwargs):
    # We are in the AWS API to create a new object, we need to invoke that first!
    try:
        from botocore.model import ServiceModel

        response = original_fn(*args, **kwargs)

        cls = type(response)
        name = cls.__qualname__
        if cls.__module__ is not None and cls.__module__ != "__builtin__":
            name = f"{cls.__module__}.{cls.__qualname__}"

        if name == "botocore.client.BedrockRuntime":
            bedrock_method = "invoke_model"
            original_invoke_model = getattr(response, bedrock_method)

            if original_invoke_model is not None:
                setattr(
                    response,
                    bedrock_method,
                    patched_call(
                        response, bedrock_method, patcher_bedrock_create_completion
                    ),
                )
            else:
                logger.error(f"failed to find method '{bedrock_method}' on {response}")

        return response

    except BaseException as error:
        logger.error(f"caught exception trying to invoke original function: {error}")
        raise error


def patcher_bedrock_create_completion(original_fn, *args, **kwargs):
    timestamp = time.time()
    result, time_delta = None, None
    model = kwargs.get("modelId") or ""
    embedding = "titan-embed" in model or "cohere.embed" in model

    completion_id = str(uuid.uuid4())

    try:
        trace_name = "AI/Bedrock/Chat/Completions/Create"

        if embedding:
            trace_name = "AI/Bedrock/Embeddings/Create"

        with newrelic.agent.FunctionTrace(
            name=trace_name, group="", terminal=True
        ) as trace:
            trace.add_custom_attribute("completion_id", completion_id)
            monitor.record_library("botocore", "Bedrock")
            result = original_fn(*args, **kwargs)
            time_delta = time.time() - timestamp
    except Exception as error:
        build_bedrock_completion_summary_for_error(error, completion_id)
        raise result

    # print the HTTP body
    from botocore.response import StreamingBody
    from io import BytesIO

    contents = result["body"].read()

    if embedding:
        handle_bedrock_embedding(result, contents, time_delta, **kwargs)
    else:
        handle_bedrock_create_completion(
            result, contents, completion_id, time_delta, **kwargs
        )

    # we have to reset the body after we read it. The handle function is going to read the contents,
    # and we'll have to apply the body again before returning back.
    bio = BytesIO(contents)
    result["body"] = StreamingBody(bio, len(contents))
    return result


def build_bedrock_completion_summary_for_error(error, completion_id):
    logger.error(f"error invoking bedrock function: {error}")

    completion = {
        "id": completion_id,
        "vendor": "bedrock",
        "ingest_source": "PythonSDK",
        "timestamp": datetime.now(),
        "error_status": error.http_status,
        "error_message": error.error.message,
        "error_type": error.error.type,
        "error_code": error.error.code,
        "error_param": error.error.param,
    }

    monitor.record_event(completion, SummaryEventName)


@handle_errors
def handle_bedrock_create_completion(
    response, contents, completion_id, time_delta, **kwargs
):
    from nr_openai_observability.patcher import flatten_dict

    try:
        response_body = json.loads(contents)

        event_dict = {
            **kwargs,
            "response_time": time_delta,
            **flatten_dict(response_body, separator="."),
            "vendor": "bedrock",
        }

        if "credentials" in event_dict:
            event_dict.pop("credentials")

        # convert body from a str to a json dictionary
        if "body" in event_dict and type(event_dict["body"]) is str:
            body = json.loads(event_dict["body"])
            event_dict["body"] = body

        (summary, messages, transaction_begin_event) = build_bedrock_events(
            response, event_dict, completion_id, time_delta
        )

        for event in messages:
            monitor.record_event(event, MessageEventName)
        monitor.record_event(summary, SummaryEventName)
        monitor.record_event(transaction_begin_event, TransactionBeginEventName)

    except BaseException as error:
        build_bedrock_completion_summary_for_error(error, completion_id)
        raise error


def handle_bedrock_embedding(result, contents, time_delta, **kwargs):
    embedding_id = str(uuid.uuid4())
    input_body = json.loads(kwargs.get("body"))
    model = kwargs.get("modelId")
    request_id = None
    if None != result.get("ResponseMetadata") and None != result.get(
        "ResponseMetadata"
    ).get("RequestId"):
        request_id = result.get("ResponseMetadata").get("RequestId")

    text = ""

    if "titan-embed" in model:
        text = input_body.get("inputText")
    elif "cohere.embed" in model:
        for t in input_body.get("texts"):
            text += f"{t}\n"

    embedding = {
        "id": embedding_id,
        "request_id": request_id,
        "input": text[:4095],
        "timestamp": datetime.now(),
        "request.model": kwargs.get("modelId"),
        "response.model": kwargs.get("modelId"),
        "vendor": "Bedrock",
        "ingest_source": "PythonSDK",
        **compat_fields(["response_time", "duration"], int(time_delta * 1000)),
        **get_trace_details(),
    }

    monitor.record_event(embedding, EmbeddingEventName)


def build_bedrock_events(response, event_dict, completion_id, time_delta):
    """
    returns (summary_event, list(message_events), transaction_event)
    """
    (
        input_message,
        input_tokens,
        response_tokens,
        stop_reason,
        temperature,
        max_tokens,
    ) = get_bedrock_info(event_dict)

    request_id = None
    if None != response.get("ResponseMetadata") and None != response.get(
        "ResponseMetadata"
    ).get("RequestId"):
        request_id = response.get("ResponseMetadata").get("RequestId")

    summary = {}
    messages = []
    model = "bedrock-unknown"

    if "modelId" in event_dict:
        model = event_dict["modelId"]
        vendor = "bedrock"
        tokens = input_tokens or 0

        if tokens and response_tokens:
            tokens += response_tokens

        if "vendor" in event_dict:
            vendor = event_dict["vendor"]

        if "temperature" in event_dict["body"]:
            temperature = event_dict["body"]["temperature"]

        if "body" in event_dict and "max_tokens_to_sample" in event_dict["body"]:
            max_tokens = event_dict["body"]["max_tokens_to_sample"]

        # input message
        messages.append(
            build_bedrock_result_message(
                completion_id=completion_id,
                message_id=str(uuid.uuid4()),
                content=input_message,
                tokens=input_tokens,
                role="user",
                sequence=len(messages),
                model=model,
                vendor=vendor,
            )
        )

        response_message_id = str(uuid.uuid4())

        if (
            "ResponseMetadata" in response
            and "RequestId" in response["ResponseMetadata"]
        ):
            response_message_id = response["ResponseMetadata"]["RequestId"]

        if "titan" in model:
            if isinstance(event_dict["results"], list):
                for result in event_dict["results"]:
                    messages.append(
                        build_bedrock_result_message(
                            completion_id=completion_id,
                            message_id=response_message_id,
                            content=result["outputText"],
                            tokens=result["tokenCount"],
                            role="assistant",
                            sequence=len(messages),
                            stop_reason=result["completionReason"],
                            model=model,
                            vendor=vendor,
                        )
                    )
            else:  # TODO Is this actually a case? Or is it always a list?
                result = event_dict["results"]
                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=response_message_id,
                        content=result["outputText"],
                        tokens=result["tokenCount"],
                        role="assistant",
                        sequence=len(messages),
                        stop_reason=result["completionReason"],
                        model=model,
                        vendor=vendor,
                    )
                )
        elif "claude" in model:
            messages.append(
                build_bedrock_result_message(
                    completion_id=completion_id,
                    message_id=response_message_id,
                    content=event_dict["completion"],
                    tokens=response_tokens,
                    role="assistant",
                    sequence=len(messages),
                    stop_reason=event_dict["stop_reason"],
                    model=model,
                    vendor=vendor,
                )
            )
        elif "ai21.j2" in model:
            for result in event_dict["completions"]:
                message_tokens = 0
                if "data" in result and "tokens" in result["data"]:
                    message_tokens = len(result["data"]["tokens"])

                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=response_message_id,
                        content=result["data"]["text"],
                        tokens=message_tokens,
                        role="assistant",
                        sequence=len(messages),
                        stop_reason=result["finishReason"]["reason"],
                        model=model,
                        vendor=vendor,
                    )
                )
        elif "cohere.command" in model:
            for result in event_dict["generations"]:
                messages.append(
                    build_bedrock_result_message(
                        completion_id=completion_id,
                        message_id=response_message_id,
                        content=result["text"],
                        role="assistant",
                        sequence=len(messages),
                        model=model,
                        vendor=vendor,
                    )
                )
        elif "meta.llama2" in model:
            messages.append(
                build_bedrock_result_message(
                    completion_id=completion_id,
                    message_id=response_message_id,
                    content=event_dict["generation"],
                    tokens=response_tokens,
                    role="assistant",
                    sequence=len(messages),
                    stop_reason=event_dict["stop_reason"],
                    model=model,
                    vendor=vendor,
                )
            )

        if len(messages) > 0:
            messages[-1]["is_final_response"] = True
            ai_message_id = create_ai_message_id(messages[-1].get("id"), request_id)
            set_ai_message_ids([ai_message_id])

        summary = {
            "id": completion_id,
            "conversation_id": get_conversation_id(),
            "timestamp": datetime.now(),
            "response_time": int(time_delta * 1000),
            "model": model,
            "request.model": model,
            "response.model": model,
            **compat_fields(["temperature", "request.temperature"], temperature),
            **compat_fields(["api_type", "response.api_type"], None),
            "vendor": vendor,
            "ingest_source": "PythonSDK",
            **compat_fields(
                ["number_of_messages", "response.number_of_messages"], len(messages)
            ),
            **get_trace_details(),
        }

        if stop_reason:
            summary.update(
                compat_fields(
                    ["finish_reason", "response.choices.finish_reason"], stop_reason
                )
            )
        if response_tokens:
            summary.update(
                compat_fields(
                    ["usage.completion_tokens", "response.usage.completion_tokens"],
                    response_tokens,
                )
            )
        if tokens:
            summary.update(
                compat_fields(
                    ["usage.total_tokens", "response.usage.total_tokens"], tokens
                )
            )
        if input_tokens:
            summary.update(
                compat_fields(
                    ["usage.prompt_tokens", "response.usage.prompt_tokens"],
                    input_tokens,
                )
            )
        if max_tokens:
            summary.update(
                compat_fields(["max_tokens", "request.max_tokens"], max_tokens)
            )

        transaction_begin_event = {
            "human_prompt": messages[0]["content"],
            "vendor": vendor,
            "ingest_source": "PythonAgentHybrid",
            **get_trace_details(),
        }

    return (summary, messages, transaction_begin_event)


def build_bedrock_result_message(
    completion_id,
    message_id,
    content,
    tokens=None,
    role=None,
    sequence=None,
    stop_reason=None,
    model=None,
    vendor=None,
):
    message = {
        "id": message_id,
        "content": content[:4095],
        "content_length": len(content),
        "conversation_id": get_conversation_id(),
        "role": role,
        "completion_id": completion_id,
        "sequence": sequence,
        **compat_fields(["model", "response.model"], model),
        "vendor": vendor,
        "ingest_source": "PythonSDK",
        **get_trace_details(),
    }

    if tokens:
        message["tokens"] = tokens
    if stop_reason:
        message["stop_reason"] = stop_reason

    return message


def get_bedrock_info(event_dict):
    """
    (input_message, input_tokens, response_tokens, completion_reason, default_temp, max_tokens) =

    default temperature and max tokens per model was found at https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
    """
    (
        input_message,
        input_tokens,
        response_tokens,
        stop_reason,
        default_temp,
        default_max_tokens,
    ) = (None, None, None, None, 0, 0)

    if "modelId" in event_dict:
        model = event_dict["modelId"]

        if "titan" in model:
            response_tokens = 0
            default_temp = 0
            default_max_tokens = 512

            if isinstance(event_dict["results"], list):
                for result in event_dict["results"]:
                    response_tokens += result["tokenCount"]
                    stop_reason = result["completionReason"]  # keep the last one

            input_message = event_dict["body"]["inputText"]
            input_tokens = event_dict["inputTextTokenCount"]
        elif "claude" in model:
            from anthropic import _tokenizers

            input_message = event_dict["body"]["prompt"]
            stop_reason = event_dict["stop_reason"]
            default_temp = 0.5
            default_max_tokens = 200

            tokenizer = _tokenizers.sync_get_tokenizer()
            encoded = tokenizer.encode(input_message)
            input_tokens = len(encoded)

            encoded = tokenizer.encode(event_dict["completion"])
            response_tokens = len(encoded)
        elif "ai21.j2" in model:
            input_message = event_dict["prompt.text"]
            input_tokens = len(event_dict["prompt.tokens"])
            response_tokens = 0
            default_temp = 0.5
            default_max_tokens = 200

            for result in event_dict["completions"]:
                stop_reason = result["finishReason"]["reason"]  # keep the last one

                if "data" in result and "tokens" in result["data"]:
                    response_tokens += len(result["data"]["tokens"])
        elif "cohere.command" in model:
            input_message = event_dict["prompt"]
            default_temp = 0.9
            default_max_tokens = 20
        elif "meta.llama2" in model:
            input_message = event_dict["body"]["prompt"]
            input_tokens = event_dict["prompt_token_count"]
            response_tokens = event_dict["generation_token_count"]
            stop_reason = event_dict["stop_reason"]
            default_temp = 0.5
            default_max_tokens = 512

    return (
        input_message,
        input_tokens,
        response_tokens,
        stop_reason,
        default_temp,
        default_max_tokens,
    )


def bind__create_api_method(
    py_operation_name, operation_name, service_model, *args, **kwargs
):
    return (py_operation_name, service_model)


def perform_patch_bedrock():
    from newrelic.common.package_version_utils import get_package_version_tuple

    botocore_version = get_package_version_tuple("botocore")
    if botocore_version == None:
        return

    (major, minor, revision) = botocore_version
    if major < 1 or minor < 31 or (minor == 31 and revision < 57):
        logger.warning(f"minimum version of botocore that supports Bedrock is 1.31.57")
        return

    boto3_version = get_package_version_tuple("boto3")
    if boto3_version == None:
        return

    (major, minor, revision) = boto3_version
    if major < 1 or minor < 28 or (minor == 28 and revision < 57):
        logger.warning("minimum version of boto3 that supports Bedrock is 1.28.57")
        return

    import botocore

    try:
        botocore.client.ClientCreator.create_client = patched_call(
            botocore.client.ClientCreator, "create_client", patcher_aws_create_api
        )
    except AttributeError as error:
        logger.debug(
            f"failed to instrument botocore.client.ClientCreator.create_client: {error}"
        )
