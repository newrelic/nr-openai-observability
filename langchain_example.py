from typing import Any, Dict
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from nr_openai_observability.langchain_callback import NewRelicCallbackHandler

def metadata_callback(ocj: Any) -> Dict[str, Any]:
    return {
        "conversation_id": 123
    }

new_relic_monitor = NewRelicCallbackHandler("LangChain observability trace", metadata_callback=metadata_callback)

def math(x):
    return 4

llm = ChatOpenAI(temperature=0, callbacks=[new_relic_monitor])
tools = []
tools.append(
    Tool(
        func=math,
        name="Calculator",
        description="useful for when you need to answer questions about math",
    )
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[new_relic_monitor],
)

agent.run("What is 2 + 2?", callbacks=[new_relic_monitor])
print("Agent run successfully!")
