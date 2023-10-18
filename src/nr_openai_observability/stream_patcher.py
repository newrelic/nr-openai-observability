import time

import nr_openai_observability.consts as consts
from nr_openai_observability.build_events import (
    build_completion_error_events,
    build_stream_completion_events,
)
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.openai_monitoring import monitor


def patcher_create_chat_completion_stream(original_fn, *args, **kwargs):
    def wrap_stream_generator(stream_gen):
        role, time_delta, content = None, None, ""
        span = monitor.create_span()
        try:
            timestamp = time.time()
            for chunk in stream_gen:
                content += chunk.choices[0].delta.get("content", "")
                if hasattr(chunk.choices[0].delta, "role"):
                    role = chunk.choices[0].delta.role
                yield chunk
            time_delta = time.time() - timestamp
        except Exception as ex:
            handle_stream_completed(
                None,
                kwargs,
                ex,
                time_delta,
                {"role": role, "content": content},
            )
            raise ex
        finally:
            span.finish()

        handle_stream_completed(
            chunk, kwargs, None, time_delta, {"role": role, "content": content}
        )

    try:
        result = original_fn(*args, **kwargs)
    except Exception as ex:
        handle_stream_completed(
            None,
            kwargs,
            ex,
            None,
            None,
        )
        raise ex

    wrapped_result = wrap_stream_generator(result)

    return wrapped_result


async def patcher_create_chat_completion_stream_async(original_fn, *args, **kwargs):
    async def wrap_stream_generator(stream_gen):
        role, time_delta, content = None, None, ""
        span = monitor.create_span()
        try:
            timestamp = time.time()
            async for chunk in await stream_gen:
                content += chunk.choices[0].delta.get("content", "")
                if hasattr(chunk.choices[0].delta, "role"):
                    role = chunk.choices[0].delta.role
                yield chunk
            time_delta = time.time() - timestamp
        except Exception as ex:
            handle_stream_completed(
                None,
                kwargs,
                ex,
                time_delta,
                {"role": role, "content": content},
            )
            raise ex
        finally:
            span.finish()

        handle_stream_completed(
            chunk, kwargs, None, time_delta, {"role": role, "content": content}
        )

    try:
        result = original_fn(*args, **kwargs)
    except Exception as ex:
        handle_stream_completed(
            None,
            kwargs,
            ex,
            None,
            None,
        )
        raise ex

    wrapped_result = wrap_stream_generator(result)

    return wrapped_result


@handle_errors
def handle_stream_completed(last_chunk, request, error, response_time, message):
    events = None
    if error:
        events = build_completion_error_events(request, error, True)
    else:
        events = build_stream_completion_events(
            last_chunk,
            request,
            getattr(last_chunk, "_nr_response_headers"),
            message,
            response_time,
        )
        delattr(last_chunk, "_nr_response_headers")

    for event in events["messages"]:
        monitor.record_event(event, consts.MessageEventName)

    monitor.record_event(events["completion"], consts.SummaryEventName)
