from typing import Any, Dict, List

import boto3
import newrelic.agent
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI, BedrockChat
from langchain.llms import Bedrock
from langchain.llms.base import LLM
from langchain.tools import Tool

# When testing changes within the SDK, we need to load the changes from a local
# directory. These lines allow for this. Unless you are testing changes to the
# SDK itself, leave these lines commented out.
#
# Add vendor directory to module search path
import os
import sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
vendor_dir = os.path.join(parent_dir, '../src')

sys.path.append(vendor_dir)
# End adding SDK

from nr_openai_observability.langchain_callback import NewRelicCallbackHandler

@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="langchain-openai")
def runLangchainOpenAI(prompt, app_name):
    new_relic_monitor = NewRelicCallbackHandler(app_name)

    openai_llm = ChatOpenAI(temperature=0)

    openai_agent = get_agent(openai_llm)
    print('Langchain with OpenAI')
    result = openai_agent.run(prompt, callbacks=[new_relic_monitor])
    print(f'prompt: {prompt}')
    print(f'result: {result}')
    print("\n\n")


@newrelic.agent.background_task()
@newrelic.agent.function_trace(name="langchain-bedrock")
def runLangchainBedrock(prompt, app_name):
    new_relic_monitor = NewRelicCallbackHandler(app_name)

    boto_client = boto3.client("bedrock-runtime", "us-east-1")
    bedrock_llm = Bedrock(
        model_id="anthropic.claude-instant-v1", # "anthropic.claude-v2",
        client=boto_client,
    )


    bedrock_agent = get_agent(bedrock_llm)
    print('Langchain with Bedrock')
    result = bedrock_agent.run(prompt, callbacks=[new_relic_monitor])

    print(f'prompt: {prompt}')
    print(f'result: {result}')
    print("\n\n")


def get_agent(llm: LLM):
    return initialize_agent(
        [
            Tool(
                func=math,
                name="Calculator",
                description="useful for when you need to answer questions about math",
            )
        ],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        stop='stop_sequence'
    )


def math(x):
    print(f'Running math tool with input {x}. Returning 4.')
    return 4


if __name__ == "__main__":
    app_name = os.getenv('NEW_RELIC_APP_NAME')
    if not app_name:
        print("You must set the NEW_RELIC_APP_NAME environment variable.")
        exit(1)


    # Enable New Relic Python agent
    newrelic.agent.initialize()
    newrelic.agent.register_application(name=app_name, timeout=10)

    prompt = "What is 2 + 2?"

    runLangchainOpenAI(prompt, app_name)
    runLangchainBedrock(prompt, app_name)

    # Allow the New Relic agent to send final messages as part of shutdown
    # The agent by default can send data up to a minute later
    newrelic.agent.shutdown_agent(60)

    print("Agent run finished!")
