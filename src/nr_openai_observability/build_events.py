import uuid
from datetime import datetime

import openai

HEADER_PREFIX_TO_BE_COLLECTED = "metadata."

def filter_dict_by_prefix(dictionary, prefix):
    return {k: v for k, v in dictionary.items() if k.startswith(prefix)}

def _build_messages_events(messages, completion_id, model):
    events = []
    for index, message in enumerate(messages):
        currMessage = {
            "id": str(uuid.uuid4()),
            "content": message.get("content", "")[:4095],
            "role": message.get("role"),
            "completion_id": completion_id,
            "sequence": index,
            "model": model,
            "vendor": "openAI",
            "ingest_source": "PythonSDK",
        }

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


def build_completion_events(response, request, response_headers, response_time):
    completion_id = str(uuid.uuid4())

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "timestamp": datetime.now(),
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
        **filter_dict_by_prefix(request.get("headers", {}), HEADER_PREFIX_TO_BE_COLLECTED),
    }

    completion.update(_get_rate_limit_data(response_headers))

    messages = _build_messages_events(
        request.get("messages", []) + [response.choices[0].message],
        completion_id,
        response.model,
    )

    return {"messages": messages, "completion": completion}


def build_completion_error_events(request, error):
    completion_id = str(uuid.uuid4())

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{openai.api_key[-4:]}",
        "timestamp": datetime.now(),
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
        **filter_dict_by_prefix(request.get("headers", {}), HEADER_PREFIX_TO_BE_COLLECTED),
    }

    messages = _build_messages_events(
        request.get("messages", []),
        completion_id,
        request.get("model") or request.get("engine"),
    )

    return {"messages": messages, "completion": completion}


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
        **filter_dict_by_prefix(request.get("headers", {}), HEADER_PREFIX_TO_BE_COLLECTED),
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
        **filter_dict_by_prefix(request.get("headers", {}), HEADER_PREFIX_TO_BE_COLLECTED),
    }

    return embedding
