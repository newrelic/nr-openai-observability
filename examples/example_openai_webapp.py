import os
import newrelic.agent

import openai
from flask import Flask

from nr_openai_observability import monitor

# The New Relic Agent should be initialized before the nr_openai_observability library. You can
# Explicitly initialize it in your application or you can use the newrelic-admin tool to start
# your application
newrelic.agent.initialize("newrelic.ini")
monitor.initialization()

app = Flask(__name__)


# When you make openai calls within your web application endpoints, New Relic will associate
# the calls with individual web transactions. This allows you to correlate individual LLM requests with
# other operations such as vector database calls, external API calls, and even other LLM responses
@app.post("/chat")
def createChat():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is Observability?"}],
    )
    return result.choices[0].message.content


app.run(port=5000)
