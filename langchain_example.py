from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from nr_openai_observability.langchain_callback import NewRelicCallbackHandler

new_relic_monitor = NewRelicCallbackHandler("LangChain observability example")

llm = ChatOpenAI(temperature=0)
tools = []
tools.append(
    Tool.from_function(
        func=lambda question: "4",
        name="Calculator",
        description="useful for when you need to answer questions about math",
        # coroutine= ... <- you can specify an async method if desired as well
    )
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[new_relic_monitor],
)

agent.run("What is 2 + 2?")
