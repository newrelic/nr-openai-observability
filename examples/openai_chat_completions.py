import newrelic.agent
import openai
import os


"""
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])
"""

@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="chat-completion-image-input")
def runChatCompletionWithImageInput():
    # change to 'gpt-4' if you have access to it
    model = "gpt-4-vision-preview" # "gpt-4-1106-preview" # "gpt-3.5-turbo"

    print(f"Running OpenAI with model {model}")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ],
    )

    print("Response from ChatGPT\n")
    print(response["choices"][0]["message"]["content"])
    print("\n\n")


if __name__ == "__main__":
    
    # Enable New Relic Python agent. You must make sure your application name is either defined in the ini file below
    # or in the environment variable NEW_RELIC_APP_NAME
    newrelic.agent.initialize("newrelic.ini")
    newrelic.agent.register_application(timeout=10)

    # Enable New Relic observability for LLMs
    from nr_openai_observability import monitor

    monitor.initialization()

    runChatCompletionWithImageInput()

    # Allow the New Relic agent to send final messages as part of shutdown
    # The agent by default can send data up to a minute later
    newrelic.agent.shutdown_agent()