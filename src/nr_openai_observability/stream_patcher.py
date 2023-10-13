import time

import nr_openai_observability.consts as consts
from nr_openai_observability.build_events import (
    build_completion_summary_for_error,
    build_messages_events,
    build_stream_completion_events,
)
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.openai_monitoring import monitor
import newrelic.agent


def patcher_create_chat_completion_stream(original_fn, *args, **kwargs):
    def wrap_stream_generator(stream_gen):
        role, time_delta, content = None, None, ""
        try:
            timestamp = time.time()
            with newrelic.agent.FunctionTrace(
                name="AI/OpenAI/Chat/Completions/Create", group="", terminal=True
            ):
                handle_start_completion(kwargs)
                for chunk in stream_gen:
                    content += chunk.choices[0].delta.get("content", "")
                    if hasattr(chunk.choices[0].delta, "role"):
                        role = chunk.choices[0].delta.role
                    yield chunk
                time_delta = time.time() - timestamp
        except Exception as ex:
            build_completion_summary_for_error(kwargs, ex, True)
            raise ex

        handle_finish_chat_completion(
            chunk, kwargs, time_delta, {"role": role, "content": content}
        )

    try:
        result = original_fn(*args, **kwargs)
    except Exception as ex:
        build_completion_summary_for_error(kwargs, ex, True)
        raise ex

    wrapped_result = wrap_stream_generator(result)

    return wrapped_result


async def patcher_create_chat_completion_stream_async(original_fn, *args, **kwargs):
    async def wrap_stream_generator(stream_gen):
        role, time_delta, content = None, None, ""
        try:
            timestamp = time.time()
            with newrelic.agent.FunctionTrace(
                name="AI/OpenAI/Chat/Completions/Create", group="", terminal=True
            ):
                handle_start_completion(kwargs)
                async for chunk in await stream_gen:
                    content += chunk.choices[0].delta.get("content", "")
                    if hasattr(chunk.choices[0].delta, "role"):
                        role = chunk.choices[0].delta.role
                    yield chunk
                time_delta = time.time() - timestamp
        except Exception as ex:
            build_completion_summary_for_error(kwargs, ex, True)
            raise ex

        handle_finish_chat_completion(
            chunk, kwargs, time_delta, {"role": role, "content": content}
        )

    try:
        result = original_fn(*args, **kwargs)
    except Exception as ex:
        build_completion_summary_for_error(kwargs, ex, True)
        raise ex

    wrapped_result = wrap_stream_generator(result)

    return wrapped_result


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
                consts.TransactionBeginEventName,
            )

    # completion_id = newrelic.agent.current_trace_id()
    message_events = build_messages_events(
        request.get("messages", []),
        request.get("model") or request.get("engine"),
    )
    for event in message_events:
        monitor.record_event(event, consts.MessageEventName)


@handle_errors
def handle_finish_chat_completion(last_chunk, request, response_time, final_message):
    initial_messages = request.get("messages", [])

    completion = build_stream_completion_events(
        last_chunk,
        request,
        getattr(last_chunk, "_nr_response_headers"),
        response_time,
        final_message,
    )
    delattr(last_chunk, "_nr_response_headers")

    response_message = build_messages_events(
        [final_message],
        last_chunk.model,
        {"is_final_response": True},
        len(initial_messages),
    )[0]

    monitor.record_event(response_message, consts.MessageEventName)

    monitor.record_event(completion, consts.SummaryEventName)