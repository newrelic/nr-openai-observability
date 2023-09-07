from typing import Any, Dict, List

import boto3
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from langchain.llms.bedrock import Bedrock
from langchain.tools import Tool

from nr_openai_observability.langchain_callback import NewRelicCallbackHandler


def metadata_callback(ocj: Any) -> Dict[str, Any]:
    return {"conversation_id": 123}


new_relic_monitor = NewRelicCallbackHandler(
    "LangChain observability trace", metadata_callback=metadata_callback
)


def math(x):
    return 4


tools = []
tools.append(
    Tool(
        func=math,
        name="Calculator",
        description="useful for when you need to answer questions about math",
    )
)


openai_llm = ChatOpenAI(temperature=0)


boto_client = boto3.client("bedrock", "us-east-1")
bedrock_llm = Bedrock(
    model_id="anthropic.claude-v2",  # "amazon.titan-tg1-large"
    client=boto_client,
)


def get_agent(llm: LLM, tools: List[Tool]):
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )


openai_agent = get_agent(openai_llm, tools)
openai_agent.run("What is 2 + 2?", callbacks=[new_relic_monitor])

bedrock_agent = get_agent(bedrock_llm, tools)
bedrock_agent.run("What is 2 + 2?", callbacks=[new_relic_monitor])


print("Agent run successfully!")
