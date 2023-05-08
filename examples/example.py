import os

import openai
from nr_openai_observability import monitor

monitor.initialization(
    environment="development",
)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Completion.create(
    model="text-davinci-003",
    prompt="What is Observability?",
    max_tokens=20,
    temperature=0 
)
