import openai
import pytest
from openai.util import convert_to_openai_object

from nr_openai_observability.monitor import patcher_create, _patched_call


@pytest.fixture(autouse=True)
def openai_patch(openai_object):
    def get_openapi_object(*args, **kwargs):
        return openai_object

    openai.Completion.create = _patched_call(
        get_openapi_object, patcher_create
    )


@pytest.fixture(autouse=True)
def openai_dict():
    return {
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "logprobs": None,
                "text": "\n\nIt depends on the situation",
            }
        ],
        "created": 1676917710,
        "id": "some-test-id-123456789",
        "model": "text-davinci-003",
        "object": "text_completion",
        "usage": {"completion_tokens": 7, "prompt_tokens": 3, "total_tokens": 10},
    }


@pytest.fixture(autouse=True)
def openai_object(openai_dict):
    return convert_to_openai_object(openai_dict)
