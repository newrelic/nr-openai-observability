import logging
import openai
import inspect
import logging
import sys
import time
import uuid
from argparse import ArgumentError

import newrelic.agent

import nr_openai_observability.consts as consts
from nr_openai_observability.build_events import (
    build_completion_summary,
    build_completion_summary_for_error,
    build_embedding_error_event,
    build_embedding_event,
    build_messages_events,
    get_trace_details,
)
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.openai_monitoring import monitor
from nr_openai_observability.patcher import (
    flatten_dict,
    patched_call,
    patched_call_async,
)
from nr_openai_observability.patchers.openai_streaming import (
    patcher_create_chat_completion_stream,
    patcher_create_chat_completion_stream_async,
)
from nr_openai_observability.call_vars import (
    create_ai_message_id,
    get_ai_message_ids,
    set_ai_message_ids,
)

logger = logging.getLogger("nr_openai_observability")


def patcher_convert_to_openai_object(original_fn, *args, **kwargs):
    response = original_fn(*args, **kwargs)

    if isinstance(args[0], openai.openai_response.OpenAIResponse):
        setattr(response, "_nr_response_headers", getattr(args[0], "_headers", {}))

    return response


def patcher_create_chat_completion(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        completion_id = str(uuid.uuid4())
        timestamp = time.time()
        with newrelic.agent.FunctionTrace(
            name="AI/OpenAI/Chat/Completions/Create",
            group="",
            terminal=True,
        ) as trace:
            trace.add_custom_attribute("completion_id", completion_id)
            monitor.record_library("openai", "OpenAI")
            handle_start_completion(kwargs, completion_id)
            result = original_fn(*args, **kwargs)
            time_delta = time.time() - timestamp
            logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

            return handle_finish_chat_completion(
                result, kwargs, time_delta, completion_id
            )
    except Exception as ex:
        monitor.record_event(
            build_completion_summary_for_error(kwargs, ex, completion_id),
            consts.SummaryEventName,
        )
        raise ex


async def patcher_create_chat_completion_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )
    result, time_delta = None, None
    try:
        completion_id = str(uuid.uuid4())
        timestamp = time.time()
        with newrelic.agent.FunctionTrace(
            name="AI/OpenAI/Chat/Completions/Create", group="", terminal=True
        ) as trace:
            trace.add_custom_attribute("completion_id", completion_id)
            monitor.record_library("openai", "OpenAI")
            handle_start_completion(kwargs, completion_id)
            result = await original_fn(*args, **kwargs)

            time_delta = time.time() - timestamp
            logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

            return handle_finish_chat_completion(
                result, kwargs, time_delta, completion_id
            )
    except Exception as ex:
        monitor.record_event(
            build_completion_summary_for_error(kwargs, ex, completion_id),
            consts.SummaryEventName,
        )
        raise ex


@handle_errors
def handle_start_completion(request, completion_id):
    transaction = newrelic.agent.current_transaction()
    if transaction and getattr(transaction, "_traceHasHadCompletions", None) == None:
        transaction._traceHasHadCompletions = True
        messages = request.get("messages", [])
        human_message = next((m for m in messages if m["role"] == "user"), None)
        if human_message:
            monitor.record_event(
                {
                    "human_prompt": human_message["content"],
                    "vendor": "openAI",
                    "ingest_source": "PythonAgentHybrid",
                    **get_trace_details(),
                },
                consts.TransactionBeginEventName,
            )

    message_events = build_messages_events(
        request.get("messages", []),
        request.get("model") or request.get("engine"),
        completion_id,
        vendor="openAI",
    )
    for event in message_events:
        monitor.record_event(event, consts.MessageEventName)


@handle_errors
def handle_finish_chat_completion(response, request, response_time, completion_id):
    initial_messages = request.get("messages", [])
    final_message = response.choices[0].message
    response_headers = getattr(response, "_nr_response_headers")

    completion = build_completion_summary(
        response,
        request,
        response_headers,
        response_time,
        final_message,
        completion_id,
    )
    delattr(response, "_nr_response_headers")

    response_message = build_messages_events(
        [final_message],
        response.model,
        completion_id,
        None,
        response.id,
        {"is_final_response": True},
        len(initial_messages),
        vendor="openAI",
    )[0]

    ai_message_ids = get_ai_message_ids(response.get("id"))

    ai_message_ids.append(
        create_ai_message_id(
            response_message.get("id"), response_headers.get("x-request-id", "")
        )
    )

    set_ai_message_ids(ai_message_ids, response.get("id"))

    monitor.record_event(response_message, consts.MessageEventName)

    monitor.record_event(completion, consts.SummaryEventName)

    return response


async def patcher_create_completion_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    result = await original_fn(*args, **kwargs)
    time_delta = time.time() - timestamp

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_completion(result, time_delta, **kwargs)


def patcher_create_completion(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    result = original_fn(*args, **kwargs)
    time_delta = time.time() - timestamp

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_completion(result, time_delta, **kwargs)


@handle_errors
def handle_create_completion(response, time_delta, **kwargs):
    choices_payload = {}
    for i, choice in enumerate(response.get("choices")):
        choices_payload.update(flatten_dict(choice, prefix="choices", index=str(i)))

    logger.debug(dict(**kwargs))

    event_dict = {
        **kwargs,
        "response_time": time_delta,
        **flatten_dict(response.to_dict_recursive(), separator="."),
        **choices_payload,
    }
    event_dict.pop("choices")

    if "messages" in event_dict:
        event_dict["messages"] = str(kwargs.get("messages"))

    logger.debug(f"Reported event dictionary:\n{event_dict}")
    monitor.record_event(event_dict)

    return response


def patcher_create_embedding(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        with newrelic.agent.FunctionTrace(
            name="AI/OpenAI/Embeddings/Create", group="", terminal=True
        ):
            timestamp = time.time()
            result = original_fn(*args, **kwargs)
            time_delta = time.time() - timestamp
    except Exception as ex:
        handle_create_embedding(result, kwargs, ex, time_delta)
        raise ex

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_embedding(result, kwargs, None, time_delta)


async def patcher_create_embedding_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        timestamp = time.time()
        result = await original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        handle_create_embedding(result, kwargs, ex, time_delta)
        raise ex

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_embedding(result, kwargs, None, time_delta)


@handle_errors
def handle_create_embedding(response, request, error, response_time):
    event = None
    if error:
        event = build_embedding_error_event(request, error)
    else:
        event = build_embedding_event(
            response, request, getattr(response, "_nr_response_headers"), response_time
        )
        delattr(response, "_nr_response_headers")

    monitor.record_event(event, consts.EmbeddingEventName)

    return response


MIN_MAJOR_VERSION = 0
MIN_MINOR_VERSION = 26

MAX_MAJOR_VERSION = 1
MIN_MAJOR_VERSION = 0


def perform_patch_openai():
    from newrelic.common.package_version_utils import get_package_version_tuple

    openai_version = get_package_version_tuple("openai")
    newrelic_version = get_package_version_tuple("newrelic")
    if openai_version == None or newrelic_version == None:
        return

    (major, minor, revision) = openai_version
    too_new = major > 1
    requires_agent = major == 1
    agent_or_plugin = major == 0 and minor >= 26
    too_old = major == 0 and minor < MIN_MINOR_VERSION

    supported_versions_msg = f"Versions between v{MIN_MAJOR_VERSION}.${MIN_MINOR_VERSION}.0 and <{MAX_MAJOR_VERSION + 1}.0 are supported"
    if too_new:
        logger.error(
            f"Detected OpenAI v{major}.{minor}.{revision} which is not supported yet. {supported_versions_msg}"
        )
        return

    if too_old:
        logger.error(
            f"Detected OpenAI v{major}.{minor}.{revision} which too old. {supported_versions_msg}"
        )
        return

    if requires_agent:
        # TODO - Check if _anything_ is instrumented. If not, raise a different error. The customer is probably trying to instrument but failing because of a
        # mismatch between the agent version and plugin
        logger.warn(f"OpenAI streaming is not yet supported in v1.0")
        return

    try:
        openai.Embedding.create = patched_call(
            openai.Embedding, "create", patcher_create_embedding
        )
    except AttributeError:
        pass

    try:
        openai.Embedding.acreate = patched_call_async(
            openai.Embedding, "acreate", patcher_create_embedding_async
        )
    except AttributeError:
        pass

    try:
        openai.Completion.create = patched_call(
            openai.Completion, "create", patcher_create_completion
        )
    except AttributeError:
        pass

    try:
        openai.Completion.acreate = patched_call_async(
            openai.Completion, "acreate", patcher_create_completion_async
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.create = patched_call(
            openai.ChatCompletion,
            "create",
            patcher_create_chat_completion,
            patcher_create_chat_completion_stream,
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.acreate = patched_call_async(
            openai.ChatCompletion,
            "acreate",
            patcher_create_chat_completion_async,
            patcher_create_chat_completion_stream_async,
        )
    except AttributeError:
        pass

    try:
        openai.util.convert_to_openai_object = patched_call(
            openai.util, "convert_to_openai_object", patcher_convert_to_openai_object
        )
    except AttributeError:
        pass
