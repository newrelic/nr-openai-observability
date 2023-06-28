import uuid
from datetime import datetime

import openai


def _build_messages_events(messages, completion_id, model):
    events = []
    for index, message in enumerate(messages):
        currMessage = {
            "content": message.get("content"),
            "role": message.get("role"),
            "completion_id": completion_id,
            "sequence": index,
            "model": model,
            "vendor": "openAI",
        }

        events.append(currMessage)

    return events


def _get_rate_limit_data(response_headers):
    ratelimit_limit_requests = response_headers.get("x-ratelimit-limit-requests")
    ratelimit_limit_tokens = response_headers.get("x-ratelimit-limit-tokens")
    ratelimit_reset_tokens = response_headers.get("x-ratelimit-reset-tokens")
    ratelimit_reset_requests = response_headers.get("x-ratelimit-reset-requests")
    ratelimit_remaining_tokens = response_headers.get("x-ratelimit-remaining-tokens")
    ratelimit_remaining_requests = response_headers.get(
        "x-ratelimit-remaining-requests"
    )

    return {
        "ratelimit_limit_requests": int(ratelimit_limit_requests)
        if ratelimit_limit_requests.isdigit()
        else None,
        "ratelimit_limit_tokens": int(ratelimit_limit_tokens)
        if ratelimit_limit_tokens.isdigit()
        else None,
        "ratelimit_reset_tokens": ratelimit_reset_tokens,
        "ratelimit_reset_requests": ratelimit_reset_requests,
        "ratelimit_remaining_tokens": int(ratelimit_remaining_tokens)
        if ratelimit_remaining_tokens.isdigit()
        else None,
        "ratelimit_remaining_requests": int(ratelimit_remaining_requests)
        if ratelimit_remaining_requests.isdigit()
        else None,
    }


def build_completion_events(response, request, response_headers):
    completion_id = str(uuid.uuid4())

    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "timestamp": datetime.now(),
        "response_time": response.response_ms,
        "request.model": request.get("model"),
        "response.model": response.model,
        "usage.completion_tokens": response.usage.completion_tokens,
        "usage.total_tokens": response.usage.total_tokens,
        "usage.prompt_tokens": response.usage.prompt_tokens,
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "finish_reason": response.choices[0].finish_reason,
        "api_type": response.api_type,
        "vendor": "openAI",
        "number_of_messages": len(request.get("messages", [])) + len(response.choices),
        "organization": response.organization,
        "api_version": response_headers.get("openai-version"),
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
        "request.model": request.get("model"),
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "vendor": "openAI",
        "organization": error.organization,
        "number_of_messages": len(request.get("messages", [])),
        "error_status": error.http_status,
        "error_message": error.error.message,
        "error_type": error.error.type,
        "error_code": error.error.code,
        "error_param": error.error.param,
    }

    messages = _build_messages_events(
        request.get("messages", []),
        completion_id,
        request.get("model"),
    )

    return {"messages": messages, "completion": completion}


def build_embedding_event(response, request, response_headers):
    embedding_id = str(uuid.uuid4())

    embedding = {
        "id": embedding_id,
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "timestamp": datetime.now(),
        "response_time": response.response_ms,
        "request.model": request.get("model"),
        "response.model": response.model,
        "usage.total_tokens": response.usage.total_tokens,
        "usage.prompt_tokens": response.usage.prompt_tokens,
        "api_type": response.api_type,
        "vendor": "openAI",
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
        "request.model": request.get("model"),
        "vendor": "openAI",
        "organization": error.organization,
        "error_status": error.http_status,
        "error_message": error.error.message,
        "error_type": error.error.type,
        "error_code": error.error.code,
        "error_param": error.error.param,
    }

    return embedding
