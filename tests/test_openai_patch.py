import openai

from nr_openai_observability import monitor


def test_patch_response(openai_object):
    monitor.initialization(
        metadata={"test": "test"}
    )
    openai.api_key = (
        "some-key"  # os.getenv("OPENAI_API_KEY")
    )
    response = openai.Completion.create(
        model="text-davinci-003", prompt="Is it?", max_tokens=7, temperature=0
    )
    assert openai_object == response
