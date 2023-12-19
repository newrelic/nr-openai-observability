import inspect
import logging
import time
import uuid

import nr_openai_observability.consts as consts
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.openai_monitoring import monitor
from nr_openai_observability.patcher import get_arg_value, patched_call

logger = logging.getLogger("nr_openai_observability")


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

            monitor.record_event(result_event_dict, consts.VectorSearchResultsEventName)

    monitor.record_event(event_dict, consts.VectorSearchEventName)

    return response


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


def perform_patch_langchain_vectorstores():
    import langchain.vectorstores
    from langchain.vectorstores import __all__ as langchain_vectordb_list

    for vector_store in langchain_vectordb_list:
        try:
            cls = getattr(langchain.vectorstores, vector_store)
            cls.similarity_search = patched_call(
                cls, "similarity_search", patcher_similarity_search
            )
        except AttributeError:
            pass
