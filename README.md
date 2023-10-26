[![Community Project header](https://github.com/newrelic/open-source-office/raw/master/examples/categories/images/Community_Project.png)](https://github.com/newrelic/open-source-office/blob/master/examples/categories/index.md#category-community-project)

# OpenAI Observability

This library extends the [New Relic Python Agent](https://github.com/newrelic/newrelic-python-agent) to enable monitoring for Large Language Models.

This library currently supports monitoring for:

- The following features from the official [Python OpenAI SDK](https://github.com/openai/openai-python):
  - Sync and async calls to the Chat Completion API in either streaming or standard mode
  - Sync and async Calls to create Embeddings
- AWS Bedrock through [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html) including `invoke_model` calls to the following models:
  - Titan (Amazon)
  - Claude (Anthropic)
  - Jurassic (AI21 Studio)
  - Command (Cohere)

## Installation

**With `pip`**

```bash
pip install git+https://github.com/newrelic/nr-openai-observability@staging
```

**With `poetry` (in `pyproject.toml`)**

```toml
[tool.poetry.dependencies]
nr-openai-observability = { git = "https://github.com/newrelic/nr-openai-observability.git", branch = "staging" }
```

## Getting Started

### STEP 1: Install the New Relic Python Agent

- You should set up the New Relic Python Agent for your application. This will help us gather general application performance monitoring (APM) information about your application like API throughput, response time, error rates, and more. We will correlate this application data with AI-specific data.

There are a number of ways you can set up the Python agent depending on your application architecture. You can follow the workflow [here](https://docs.newrelic.com/install/python/) for help configuring your agent.

_Limited Preview only_ - While the nr-openai-observability library is in Limited Preview, you should install the following version of the New Relic agent:

**With `pip`**

```bash
pip install git+https://github.com/newrelic/newrelic-python-agent@3b6273ee65d92fb24cf803d4023fe45b3620ee93
```

**With `poetry` (in `pyproject.toml`)**

```toml
[tool.poetry.dependencies]
newrelic = { git = "https://github.com/newrelic/newrelic-python-agent.git", rev = "3b6273ee65d92fb24cf803d4023fe45b3620ee93" }
```

### Step 2: Configure your New Relic Python Agent

After you have installed the Python agent, you will need to turn on two configuration fields

**Using environment variables**
`NEW_RELIC_MACHINE_LEARNING_ENABLED=true`
`NEW_RELIC_ML_INSIGHTS_EVENTS_ENABLED=true`

**Using the agent configuration file**

```toml
machine_learning.enabled = true
ml_insights_events.enabled = true
```

We also strongly recommend that you have the following configurations enabled:

- [`application_logging.enabled`](https://docs.newrelic.com/docs/apm/agents/python-agent/configuration/python-agent-configuration/#application_logging.enabled) - Allows us to capture application logs and correlate them with LLM calls
- [`application_logging.forwarding.enabled](https://docs.newrelic.com/docs/apm/agents/python-agent/configuration/python-agent-configuration/#application_logging.forwarding.enabled) - Automatically forwards application logs to New Relic (if you have another log forwarding setup, you can use that instead)
- [`distributed_tracing.enabled`](https://docs.newrelic.com/docs/apm/agents/python-agent/configuration/python-agent-configuration/#distributed-tracing-enabled) - Enables distributed tracing so that we can understand your application's LLM usage in the context of your overall architecture

### STEP 3: Initialize AI-specific monitoring

Include the following lines in your application code:

```python
from nr_openai_observability import monitor

monitor.initialization()
```

It is important that the `monitor.initialization()` line runs after the `openai`or `boto3` library is imported. In most cases, the only thing you'll need to do to ensure this happens is to run the initialization call after your main block of imports in your entrypoint file.

#### Code example:

```python

import os

import openai
from nr_openai_observability import monitor

monitor.initialization()

openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a rhythm about observability",
        },
    ],
)
print(response["choices"][0]["message"]["content"])
```

## Support

New Relic hosts and moderates an online forum where customers can interact with New Relic employees as well as other customers to get help and share best practices. Like all official New Relic open source projects, there's a related Community topic in the New Relic Explorers Hub. You can find this project's topic/threads here:

## Contribute

We encourage your contributions to improve nr-openai-observability! Keep in mind that when you submit your pull request, you'll need to sign the CLA via the click-through using CLA-Assistant. You only have to sign the CLA one time per project.

If you have any questions, or to execute our corporate CLA (which is required if your contribution is on behalf of a company), drop us an email at opensource@newrelic.com.

**A note about vulnerabilities**

As noted in our [security policy](../../security/policy), New Relic is committed to the privacy and security of our customers and their data. We believe that providing coordinated disclosure by security researchers and engaging with the security community are important means to achieve our security goals.

If you believe you have found a security vulnerability in this project or any of New Relic's products or websites, we welcome and greatly appreciate you reporting it to New Relic through [HackerOne](https://hackerone.com/newrelic).

If you would like to contribute to this project, review [these guidelines](./CONTRIBUTING.md).

To all contributors, we thank you! Without your contribution, this project would not be what it is today.

## License

nr-openai-observability is licensed under the [Apache 2.0](http://apache.org/licenses/LICENSE-2.0.txt) License.
The nr-openai-observability also uses source code from third-party libraries. You can find full details on which libraries are used and the terms under which they are licensed in the third-party notices document.
