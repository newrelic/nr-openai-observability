import inspect
import logging
import os
import sys
import time
import uuid
from argparse import ArgumentError
from typing import Any, Dict, Optional

import openai

import newrelic.agent

from nr_openai_observability.build_events import (
    build_completion_summary_for_error,
    build_completion_summary,
    build_embedding_error_event,
    build_embedding_event,
    build_messages_events,
    build_ai_feedback_event,
)
from nr_openai_observability.call_vars import (
    get_message_id,
    create_ai_message_id,
    get_ai_message_ids,
    set_ai_message_ids
)
from nr_openai_observability.error_handling_decorator import handle_errors

logger = logging.getLogger("nr_openai_observability")

EventName = "LlmCompletion"
MessageEventName = "LlmChatCompletionMessage"
SummaryEventName = "LlmChatCompletionSummary"
EmbeddingEventName = "LlmEmbedding"
FeedbackEventName = "LlmChatFeedback"
VectorSearchEventName = "LlmVectorSearch"
VectorSearchResultsEventName = "LlmVectorSearchResult"
TransactionBeginEventName = "LlmTransactionBegin"


def _patched_call(original_fn, patched_fn):
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    def _inner_patch(*args, **kwargs):
        if kwargs.get("stream") is True:
            logger.warning(
                "stream = True is not supported by nr_openai_observability. Ignoring monitoring for this function call"
            )
            return original_fn(*args, **kwargs)

        try:
            return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def _patched_call_async(original_fn, patched_fn):
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    async def _inner_patch(*args, **kwargs):
        if kwargs.get("stream") is True:
            logger.warning(
                "stream = True is not supported by nr_openai_observability. Ignoring monitoring for this function call"
            )
            return await original_fn(*args, **kwargs)
        try:
            return await patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


class OpenAIMonitoring:
    # this class uses the telemetry SDK to record metrics to new relic, please see https://github.com/newrelic/newrelic-telemetry-sdk-python
    def __init__(
        self,
        use_logger: Optional[bool] = None,
    ):
        self.use_logger = use_logger if use_logger else False
        self.headers_by_id: dict = {}
        self.initialized = False

    def _set_metadata(
        self,
        metadata: Dict[str, Any] = {},
    ):
        self.metadata = metadata

        if not isinstance(metadata, Dict) and metadata is not None:
            raise TypeError("metadata instance type must be Dict[str, Any]")

    def _log(self, msg: str):
        if self.use_logger:
            logger.info(msg)
        else:
            print(msg)

    def start(
        self,
        application_name: str,
        metadata: Dict[str, Any] = {},
        metadata_callback: Optional[callable] = None,
    ):
        if not self.initialized:
            self.application_name = application_name
            self._set_metadata(metadata)
            self.metadata_callback = metadata_callback
            self._start()
            self.initialized = True

    # initialize event thread
    def _start(self):
        None

    def record_event(
        self,
        event_dict: dict,
        table: str = EventName,
    ):
        event_dict.update(self.metadata)
        if self.metadata_callback:
            try:
                metadata = self.metadata_callback(event_dict)
                if metadata:
                    event_dict.update(metadata)
            except Exception as ex:
                logger.warning(f"Failed to run metadata callback: {ex}")
        newrelic.agent.record_custom_event(table, event_dict)


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
        timestamp = time.time()
        with newrelic.agent.FunctionTrace(
            name="AI/OpenAI/Chat/Completions/Create", group="", terminal=True
        ):
            handle_start_completion(kwargs)
            result = original_fn(*args, **kwargs)
            time_delta = time.time() - timestamp
            logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

            return handle_finish_chat_completion(result, kwargs, time_delta)
    except Exception as ex:
        build_completion_summary_for_error(kwargs, ex)
        raise ex


async def patcher_create_chat_completion_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )
    result, time_delta = None, None
    try:
        timestamp = time.time()
        with newrelic.agent.FunctionTrace(
            name="AI/OpenAI/Chat/Completions/Create", group="", terminal=True
        ):
            handle_start_completion(kwargs)
            result = await original_fn(*args, **kwargs)

            time_delta = time.time() - timestamp
            logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

            return handle_finish_chat_completion(result, kwargs, time_delta)
    except Exception as ex:
        build_completion_summary_for_error(kwargs, ex)
        raise ex


@handle_errors
def handle_start_completion(request):
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
                    "trace.id": transaction.trace_id,
                    "ingest_source": "PythonAgentHybrid",
                },
                TransactionBeginEventName,
            )

    # completion_id = newrelic.agent.current_trace_id()
    message_events = build_messages_events(
        request.get("messages", []),
        request.get("model") or request.get("engine"),
    )
    for event in message_events:
        monitor.record_event(event, MessageEventName)


@handle_errors
def handle_finish_chat_completion(response, request, response_time):
    initial_messages = request.get("messages", [])
    final_message = response.choices[0].message

    completion = build_completion_summary(
        response,
        request,
        getattr(response, "_nr_response_headers"),
        response_time,
        final_message,
    )
    delattr(response, "_nr_response_headers")

    response_message = build_messages_events(
        [final_message],
        response.model,
        get_message_id(),
        response.get("id"),
        {"is_final_response": True},
        len(initial_messages),
    )[0]

    ai_message_ids = []

    ai_message_ids.append(
        create_ai_message_id(response_message.get("id"), response.get("id"))
    )

    set_ai_message_ids(ai_message_ids, response.get("id"))

    monitor.record_event(response_message, MessageEventName)

    monitor.record_event(completion, SummaryEventName)

    return response


def get_arg_value(
    args,
    kwargs,
    pos,
    kw,
):
    try:
        return kwargs[kw]
    except KeyError:
        try:
            return args[pos]
        except IndexError:
            raise ArgumentError("Missing required argument: %s" % (kw,))


@handle_errors
def handle_similarity_search(
    original_fn, response, request_args, request_kwargs, error, response_time
):
    vendor = inspect.getmodule(original_fn).__name__.split(".")[-1]
    query = get_arg_value(request_args, request_kwargs, 1, "query")
    k = request_kwargs.get("k")
    if k is None and len(request_args) >= 3:
        k = request_args[1]

    event_dict = {
        "provider": vendor,
        "query": query,
        "k": k,
        "response_time": response_time,
    }

    if error:
        event_dict["error"] = str(error)
    else:
        documents = response
        event_dict["search_id"] = str(uuid.uuid4())
        event_dict["document_count"] = len(documents)
        for idx, document in enumerate(documents):
            result_event_dict = {
                "search_id": event_dict["search_id"],
                "result_rank": idx,
                "document_page_content": str(document.page_content),
            }
            for kwarg_key, v in document.metadata.items():
                result_event_dict[f"document_metadata_{kwarg_key}"] = str(v)

            result_event_dict.update(**event_dict)

            monitor.record_event(result_event_dict, VectorSearchResultsEventName)

    monitor.record_event(event_dict, VectorSearchEventName)

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


def flatten_dict(dd, separator=".", prefix="", index=""):
    if len(index):
        index = index + separator
    return (
        {
            prefix + separator + index + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


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


def patcher_similarity_search(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        timestamp = time.time()
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        handle_similarity_search(original_fn, result, args, kwargs, ex, time_delta)
        raise ex

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_similarity_search(original_fn, result, args, kwargs, None, time_delta)


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

    monitor.record_event(event, EmbeddingEventName)

    return response


monitor = OpenAIMonitoring()


def initialization(
    application_name: str,
    metadata: Dict[str, Any] = {},
    metadata_callback: Optional[callable] = None,
):
    monitor.start(
        application_name,
        metadata,
        metadata_callback,
    )
    perform_patch()
    return monitor


def perform_patch_langchain_vectorstores():
    import langchain.vectorstores
    from langchain.vectorstores import __all__ as langchain_vectordb_list

    for vector_store in langchain_vectordb_list:
        try:
            cls = getattr(langchain.vectorstores, vector_store)
            cls.similarity_search = _patched_call(
                cls.similarity_search, patcher_similarity_search
            )
        except AttributeError:
            pass


def perform_patch():
    try:
        openai.Embedding.create = _patched_call(
            openai.Embedding.create, patcher_create_embedding
        )
    except AttributeError:
        pass

    try:
        openai.Embedding.acreate = _patched_call_async(
            openai.Embedding.acreate, patcher_create_embedding_async
        )
    except AttributeError:
        pass

    try:
        openai.Completion.create = _patched_call(
            openai.Completion.create, patcher_create_completion
        )
    except AttributeError:
        pass

    try:
        openai.Completion.acreate = _patched_call_async(
            openai.Completion.acreate, patcher_create_completion_async
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.create = _patched_call(
            openai.ChatCompletion.create, patcher_create_chat_completion
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.acreate = _patched_call_async(
            openai.ChatCompletion.acreate, patcher_create_chat_completion_async
        )
    except AttributeError:
        pass

    try:
        openai.util.convert_to_openai_object = _patched_call(
            openai.util.convert_to_openai_object, patcher_convert_to_openai_object
        )
    except AttributeError:
        pass

    from nr_openai_observability.bedrock import perform_patch_bedrock
    perform_patch_bedrock()

    if "langchain" in sys.modules:
        perform_patch_langchain_vectorstores()

def record_ai_feedback_event(category, rating, message_id, conversation_id = None, request_id = None, message = None):
    event = build_ai_feedback_event(category, rating, message_id, conversation_id, request_id, message)

    monitor.record_event(event, FeedbackEventName)
