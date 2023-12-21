import logging
import sys
from argparse import ArgumentError

logger = logging.getLogger("nr_openai_observability")


def patched_call(
    parent_class,
    method_name,
    patched_fn,
    stream_patched_fn=None,
    expect_agent_instrumentation=False,
):
    original_fn = getattr(parent_class, method_name)
    patched_by_plugin = hasattr(original_fn, "is_patched_by_monitor")
    patched_by_agent = getattr(parent_class, "_nr_wrapped", False)
    if expect_agent_instrumentation and not patched_by_agent:
        logger.warn(
            f"{parent_class.__name__}{method_name} should be instrumented by the python agent but it is not. Falling back to plugin instrumentation"
        )

    if patched_by_plugin:
        return original_fn

    def _inner_patch(*args, **kwargs):
        try:
            if kwargs.get("stream") is True and stream_patched_fn is not None:
                # The agent doesn't support streaming yet. Try to use the plugin. This will need to change once the agent patches streaming calls
                return stream_patched_fn(original_fn, *args, **kwargs)
            else:
                if patched_by_agent:
                    return original_fn(*args, **kwargs)
                else:
                    return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def patched_call_async(
    parent_class,
    method_name,
    patched_fn,
    stream_patched_fn=None,
    expect_agent_instrumentation=False,
):
    original_fn = getattr(parent_class, method_name)
    patched_by_plugin = hasattr(original_fn, "is_patched_by_monitor")
    patched_by_agent = getattr(parent_class, "_nr_wrapped", False)

    if expect_agent_instrumentation and not patched_by_agent:
        logger.warn(
            f"{str(parent_class)}{method_name} should be instrumented by the python agent but it is not. Falling back to plugin instrumentation"
        )

    if patched_by_plugin:
        return original_fn

    async def _inner_patch(*args, **kwargs):
        try:
            if kwargs.get("stream") is True and stream_patched_fn is not None:
                return await stream_patched_fn(original_fn, *args, **kwargs)
            else:
                if patched_by_agent:
                    return await original_fn(*args, **kwargs)
                else:
                    return await patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


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


def perform_patch():
    from nr_openai_observability.patchers.bedrock import (
        perform_patch_bedrock,
    )
    from nr_openai_observability.patchers.langchain import (
        perform_patch_langchain_vectorstores,
    )
    from nr_openai_observability.patchers.openai import perform_patch_openai

    perform_patch_bedrock()
    perform_patch_openai()

    if "langchain" in sys.modules:
        perform_patch_langchain_vectorstores()
