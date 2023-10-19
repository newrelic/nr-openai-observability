import logging
import uuid
from datetime import datetime

import openai
import newrelic.agent
import tiktoken

logger = logging.getLogger("nr_openai_observability")


def build_messages_events(messages, model, tags={}, start_seq_num=0):
    completion_id = newrelic.agent.current_span_id()
    trace_id = newrelic.agent.current_trace_id()

    events = []
    for index, message in enumerate(messages):
        currMessage = {
            "id": str(uuid.uuid4()),
            "content": (message.get("content") or "")[:4095],
            "role": message.get("role"),
            "completion_id": completion_id,
            "trace.id": trace_id,
            "sequence": index + start_seq_num,
            "model": model,
            "vendor": "openAI",
            "ingest_source": "PythonSDK",
        }
        currMessage.update(tags)

        events.append(currMessage)

    return events


def _get_rate_limit_data(response_headers):
    def _get_numeric_header(name):
        header = response_headers.get(name)
        return int(header) if header and header.isdigit() else None

    return {
        "ratelimit_limit_requests": _get_numeric_header("ratelimit_limit_requests"),
        "ratelimit_limit_tokens": _get_numeric_header("ratelimit_limit_tokens"),
        "ratelimit_reset_tokens": response_headers.get("x-ratelimit-reset-tokens"),
        "ratelimit_reset_requests": response_headers.get("x-ratelimit-reset-requests"),
        "ratelimit_remaining_tokens": _get_numeric_header("ratelimit_remaining_tokens"),
        "ratelimit_remaining_requests": _get_numeric_header(
            "ratelimit_remaining_requests"
        ),
    }


def calc_completion_tokens(model, message_content):
    try:
        encoding = tiktoken.encoding_for_model(model)
        # could not find encoding for model
    except KeyError:
        return None

    return len(encoding.encode(message_content))


def calc_prompt_tokens(model, messages):
    """
    calculate prompt tokens based on this document
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        # could not find encoding for model
    except KeyError:
        return None

    num_of_tokens_per_msg = 3
    num_of_tokens_per_name = 1

    if model == "gpt-3.5-turbo-0301":
        num_of_tokens_per_msg = 4
        num_of_tokens_per_name = -1

    if "gpt-4" not in model and "gpt-3" not in model:
        logger.warn(f"model:{model} is unsupported for streaming token calculation")
        return None

    num_of_tokens = 3  # this is based on the link in the docstring, every reply contains base 3 tokens that are added tp the prompt

    for message in messages:
        num_of_tokens += num_of_tokens_per_msg
        for key, value in message.items():
            num_of_tokens += len(encoding.encode(value))
            if key == "name":
                num_of_tokens += num_of_tokens_per_name

    return num_of_tokens


def build_stream_completion_events(
    last_chunk, request, response_headers, message, response_time
):
    completion_id = newrelic.agent.current_span_id()
    trace_id = newrelic.agent.current_trace_id()
    transaction_id = (
        newrelic.agent.current_transaction().guid
        if newrelic.agent.current_transaction() != None
        else None
    )

    request_messages = request.get("messages", [])

    prompt_tokens = calc_prompt_tokens(last_chunk.model, request_messages)
    completion_tokens = calc_completion_tokens(
        last_chunk.model, message.get("content", "")
    )
    total_tokens = (
        completion_tokens + prompt_tokens
        if completion_tokens and prompt_tokens
        else None
    )

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{last_chunk.api_key[-4:]}",
        "response_time": int(response_time * 1000),
        "request.model": request.get("model") or request.get("engine"),
        "response.model": last_chunk.model,
        "usage.completion_tokens": completion_tokens,
        "usage.total_tokens": total_tokens,
        "usage.prompt_tokens": prompt_tokens,
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "finish_reason": last_chunk.choices[0].finish_reason,
        "api_type": last_chunk.api_type,
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        "number_of_messages": len(request.get("messages", [])) + 1,
        "organization": last_chunk.organization,
        "api_version": response_headers.get("openai-version"),
        "trace.id": trace_id,
        "transactionId": transaction_id,
        "response": (message.get("content") or "")[:4095],
        "stream": True,
    }

    completion.update(_get_rate_limit_data(response_headers))

    return completion


def build_completion_summary(
    response, request, response_headers, response_time, final_message
):
    completion_id = newrelic.agent.current_span_id()
    trace_id = newrelic.agent.current_trace_id()
    transaction_id = (
        newrelic.agent.current_transaction().guid
        if newrelic.agent.current_transaction() != None
        else None
    )

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "response_time": int(response_time * 1000),
        "request.model": request.get("model") or request.get("engine"),
        "response.model": response.model,
        "usage.completion_tokens": response.usage.completion_tokens,
        "usage.total_tokens": response.usage.total_tokens,
        "usage.prompt_tokens": response.usage.prompt_tokens,
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "finish_reason": response.choices[0].finish_reason,
        "api_type": response.api_type,
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        "number_of_messages": len(request.get("messages", [])) + len(response.choices),
        "organization": response.organization,
        "api_version": response_headers.get("openai-version"),
        "trace.id": trace_id,
        "transactionId": transaction_id,
        "response": (final_message.get("content") or "")[:4095],
        "stream": False,
    }

    completion.update(_get_rate_limit_data(response_headers))

    return completion


def build_completion_summary_for_error(request, error, isStream=False):
    completion_id = str(uuid.uuid4())

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{openai.api_key[-4:]}",
        "request.model": request.get("model") or request.get("engine"),
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        "organization": error.organization,
        "number_of_messages": len(request.get("messages", [])),
        "error_status": error.http_status,
        "error_message": error.error.message,
        "error_type": error.error.type,
        "error_code": error.error.code,
        "error_param": error.error.param,
        "stream": isStream,
    }

    return completion


def build_embedding_event(response, request, response_headers, response_time):
    embedding_id = str(uuid.uuid4())

    embedding = {
        "id": embedding_id,
        "input": request.get("input")[:4095],
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "timestamp": datetime.now(),
        "response_time": int(response_time * 1000),
        "request.model": request.get("model") or request.get("engine"),
        "response.model": response.model,
        "usage.total_tokens": response.usage.total_tokens,
        "usage.prompt_tokens": response.usage.prompt_tokens,
        "api_type": response.api_type,
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        "organization": response.organization,
        "api_version": response_headers.get("openai-version"),
    }

    embedding.update(_get_rate_limit_data(response_headers))
    return embedding


def build_embedding_error_event(request, error):
    embedding_id = str(uuid.uuid4())

    embedding = {
        "id": embedding_id,
        "api_key_last_four_digits": f"sk-{openai.api_key[-4:]}",
        "timestamp": datetime.now(),
        "request.model": request.get("model") or request.get("engine"),
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        "organization": error.organization,
        "error_status": error.http_status,
        "error_message": error.error.message,
        "error_type": error.error.type,
        "error_code": error.error.code,
        "error_param": error.error.param,
    }

    return embedding
