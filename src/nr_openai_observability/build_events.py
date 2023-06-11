import uuid
from datetime import datetime


def build_events(response, request, response_headers):
    completion_id = str(uuid.uuid4())
    completion = {
        "id": completion_id,
        "api_key_last_four_digits": f"sk-{response.api_key[-4:]}",
        "timestamp": datetime.now(),
        "response_time": response.response_ms,
        "model": response.model,
        "usage.completion_tokens": response.usage.completion_tokens,
        "usage.total_tokens": response.usage.total_tokens,
        "usage.prompt_tokens": response.usage.prompt_tokens,
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        "finish_reason": response.choices[0].finish_reason,
        "api_type": response.api_type,
        "vendor": "openAI",
        "number_of_messages": len(request.get("messages", [])) + len(response.choices),
        "ratelimit_limit_requests": response_headers.get("x-ratelimit-limit-requests"),
        "ratelimit_limit_tokens": response_headers.get("x-ratelimit-limit-tokens"),
        "ratelimit_reset_tokens": response_headers.get("x-ratelimit-reset-tokens"),
        "ratelimit_reset_requests": response_headers.get("x-ratelimit-reset-requests"),
        "ratelimit_remaining_tokens": response_headers.get(
            "x-ratelimit-remaining-tokens"
        ),
        "ratelimit_remaining_requests": response_headers.get(
            "x-ratelimit-remaining-requests"
        ),
        "organization": response.organization,
        "api_version": response_headers.get("openai-version"),
    }

    messages = []

    for index, message in enumerate(
        request.get("messages", []) + [response.choices[0].message]
    ):
        currMessage = {
            "content": message.get("content"),
            "role": message.get("role"),
            "completion_id": completion_id,
            "sequence": index,
            "model": response.model,
            "vendor": "openAI",
        }

        messages.append(currMessage)

    return {"messages": messages, "completion": completion}
