import logging
import uuid
from datetime import datetime

import openai
import newrelic.agent
import tiktoken

logger = logging.getLogger("nr_openai_observability")


def build_messages_events(messages, model, completion_id, tags={}, start_seq_num=0):
    events = []
    for index, message in enumerate(messages):
        currMessage = {
            "id": str(uuid.uuid4()),
            "completion_id": completion_id,
            "content": (message.get("content") or "")[:4095],
            "role": message.get("role"),
            "sequence": index + start_seq_num,
            **compat_fields(["model", "response.model"], model),
            "vendor": "openAI",
            "ingest_source": "PythonSDK",
            **get_trace_details(),
        }
        currMessage.update(tags)

        events.append(currMessage)

    return events


def _get_rate_limit_data(response_headers):
    def _get_numeric_header(name):
        header = response_headers.get(name)
        return int(header) if header and header.isdigit() else None

    return {
        **compat_fields(
            ["ratelimit_limit_requests", "response.headers.ratelimitLimitRequests"],
            _get_numeric_header("ratelimit_limit_requests"),
        ),
        **compat_fields(
            ["ratelimit_limit_tokens", "response.headers.ratelimitLimitTokens"],
            _get_numeric_header("ratelimit_limit_tokens"),
        ),
        **compat_fields(
            ["ratelimit_reset_tokens", "response.headers.ratelimitResetTokens"],
            response_headers.get("x-ratelimit-reset-tokens"),
        ),
        **compat_fields(
            ["ratelimit_reset_requests", "response.headers.ratelimitResetRequests"],
            response_headers.get("x-ratelimit-reset-requests"),
        ),
        **compat_fields(
            ["ratelimit_remaining_tokens", "response.headers.ratelimitRemainingTokens"],
            _get_numeric_header("ratelimit_remaining_tokens"),
        ),
        **compat_fields(
            [
                "ratelimit_remaining_requests",
                "response.headers.ratelimitRemainingRequests",
            ],
            _get_numeric_header("ratelimit_remaining_requests"),
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
    last_chunk, request, response_headers, message, response_time, completion_id
):
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
        **compat_fields(
            ["usage.completion_tokens", "response.usage.completion_tokens"],
            completion_tokens,
        ),
        **compat_fields(
            ["usage.total_tokens", "response.usage.total_tokens"], total_tokens
        ),
        **compat_fields(
            ["usage.prompt_tokens", "response.usage.prompt_tokens"], prompt_tokens
        ),
        **compat_fields(
            ["temperature", "request.temperature"], request.get("temperature")
        ),
        **compat_fields(
            ["max_tokens", "request.max_tokens"], request.get("max_tokens")
        ),
        **compat_fields(
            ["finish_reason", "response.choices.finish_reason"],
            last_chunk.choices[0].finish_reason,
        ),
        **compat_fields(["api_type", "response.api_type"], last_chunk.api_type),
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        **compat_fields(
            ["number_of_messages", "response.number_of_messages"],
            len(request.get("messages", [])) + 1,
        ),
        **compat_fields(
            ["organization", "response.organization"], last_chunk.organization
        ),
        **compat_fields(
            ["api_version", "response.headers.llmVersion"],
            response_headers.get("openai-version"),
        ),
        "response": (message.get("content") or "")[:4095],
        "stream": True,
        **get_trace_details(),
    }

    completion.update(_get_rate_limit_data(response_headers))

    return completion


def build_completion_summary(
    response, request, response_headers, response_time, final_message, completion_id
):
    completion = {
        "id": completion_id,
        "request_id": response_headers.get("x-request-id", ""),
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "response_time": int(response_time * 1000),
        "request.model": request.get("model") or request.get("engine"),
        "response.model": response.model,
        **compat_fields(
            ["organization", "response.organization"], response.organization
        ),
        **compat_fields(
            ["usage.completion_tokens", "response.usage.completion_tokens"],
            response.usage.completion_tokens,
        ),
        **compat_fields(
            ["usage.total_tokens", "response.usage.total_tokens"],
            response.usage.total_tokens,
        ),
        **compat_fields(
            ["usage.prompt_tokens", "response.usage.prompt_tokens"],
            response.usage.prompt_tokens,
        ),
        **compat_fields(
            ["temperature", "request.temperature"], request.get("temperature")
        ),
        **compat_fields(
            ["max_tokens", "request.max_tokens"], request.get("max_tokens")
        ),
        **compat_fields(
            ["finish_reason", "response.choices.finish_reason"],
            response.choices[0].finish_reason,
        ),
        **compat_fields(["api_type", "response.api_type"], response.api_type),
        **compat_fields(
            ["api_version", "response.headers.llmVersion"],
            response_headers.get("openai-version"),
        ),
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        **compat_fields(
            ["number_of_messages", "response.number_of_messages"],
            len(request.get("messages", [])) + len(response.choices),
        ),
        "response": (final_message.get("content") or "")[:4095],
        "stream": False,
        **get_trace_details(),
    }

    completion.update(_get_rate_limit_data(response_headers))

    return completion


def build_completion_summary_for_error(request, error, completion_id, isStream=False):
    nested_error = getattr(error, "error", None)

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{openai.api_key[-4:]}",
        "request.model": request.get("model") or request.get("engine"),
        **compat_fields(
            ["temperature", "request.temperature"], request.get("temperature")
        ),
        **compat_fields(
            ["max_tokens", "request.max_tokens"], request.get("max_tokens")
        ),
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        **compat_fields(
            ["organization", "response.organization"],
            getattr(error, "organization", None),
        ),
        **compat_fields(
            ["number_of_messages", "response.number_of_messages"],
            len(request.get("messages", [])),
        ),
        "error_status": getattr(error, "http_status", None),
        "error_message": nested_error.message if nested_error else str(error),
        "error_type": getattr(nested_error, "type", None) if nested_error else error.__class__.__name__,
        "error_code": getattr(nested_error, "code", None) if nested_error else None,
        "error_param": getattr(nested_error, "param", None) if nested_error else None,
        "stream": isStream,
        **get_trace_details(),
    }

    return completion


def build_embedding_event(response, request, response_headers, response_time):
    embedding_id = str(uuid.uuid4())

    embedding = {
        "id": embedding_id,
        "request_id": response_headers.get("x-request-id", ""),
        "input": request.get("input")[:4095],
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "timestamp": datetime.now(),
        "request.model": request.get("model") or request.get("engine"),
        "response.model": response.model,
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        **compat_fields(["response_time", "duration"], int(response_time * 1000)),
        **compat_fields(
            ["usage.total_tokens", "response.usage.total_tokens"],
            response.usage.total_tokens,
        ),
        **compat_fields(
            ["usage.prompt_tokens", "response.usage.prompt_tokens"],
            response.usage.prompt_tokens,
        ),
        **compat_fields(["api_type", "response.api_type"], response.api_type),
        **compat_fields(
            ["api_version", "response.headers.llmVersion"],
            response_headers.get("openai-version"),
        ),
        **compat_fields(
            ["organization", "response.organization"], response.organization
        ),
        **get_trace_details(),
    }

    embedding.update(_get_rate_limit_data(response_headers))
    return embedding


def build_embedding_error_event(request, error):
    embedding_id = str(uuid.uuid4())
    nested_error = getattr(error, "error", None)

    embedding = {
        "id": embedding_id,
        "api_key_last_four_digits": f"sk-{openai.api_key[-4:]}",
        "timestamp": datetime.now(),
        "request.model": request.get("model") or request.get("engine"),
        "vendor": "openAI",
        "ingest_source": "PythonSDK",
        **compat_fields(
            ["organization", "response.organization"],
            getattr(error, "organization", None),
        ),
        "error_status": getattr(error, "http_status", None),
        "error_message": nested_error.message if nested_error else str(error),
        "error_type": getattr(nested_error, "type", None) if nested_error else error.__class__.__name__,
        "error_code": getattr(nested_error, "code", None) if nested_error else None,
        "error_param": getattr(nested_error, "param", None) if nested_error else None,
        **get_trace_details(),
    }

    return embedding


def get_trace_details():
    span_id = newrelic.agent.current_span_id()
    trace_id = newrelic.agent.current_trace_id()
    transaction_id = (
        newrelic.agent.current_transaction().guid
        if newrelic.agent.current_transaction() != None
        else None
    )
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "trace.id": trace_id,  # Legacy value from SDK
        "transaction_id": transaction_id,
        "transactionId": transaction_id,  # Legacy value from SDK
    }


def compat_fields(keys, value):
    return dict.fromkeys(keys, value)
