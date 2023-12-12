import logging
import sys
from argparse import ArgumentError

logger = logging.getLogger("nr_openai_observability")


def patched_call(original_fn, patched_fn, stream_patched_fn=None):
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    def _inner_patch(*args, **kwargs):
        try:
            if kwargs.get("stream") is True and stream_patched_fn is not None:
                return stream_patched_fn(original_fn, *args, **kwargs)
            else:
                return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def patched_call_async(original_fn, patched_fn, stream_patched_fn=None):
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    async def _inner_patch(*args, **kwargs):
        try:
            if kwargs.get("stream") is True and stream_patched_fn is not None:
                return await stream_patched_fn(original_fn, *args, **kwargs)
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
