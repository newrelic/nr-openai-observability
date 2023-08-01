import os

import openai
from nr_openai_observability import monitor

monitor.initialization(
   application_name="OpenAI observability example", metadata={"environment": "development"}
)

openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a rhythm about observability",
        },
    ],
    headers={
        "metadata.conversion_id": "1",
    }
)
print(response["choices"][0]["message"]["content"])
