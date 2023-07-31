from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

from nr_openai_observability import monitor
from nr_openai_observability.monitor import monitor as new_relic_monitor


class NewRelicCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        application_name: str,
    ) -> None:
        """Initialize callback handler."""
        self.application_name = application_name

        monitor.initialization(application_name=application_name)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        new_relic_monitor.record_event({}, "LlmOnLLMStart")
        print(f"on_llm_start {serialized}", prompts, kwargs)
        print()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        new_relic_monitor.record_event(
            {
                "inputs": inputs,
                "run_id": str(kwargs.get("run_id")),
                "tags": str(kwargs.get("tags")),
                "metadata": str(kwargs.get("metadata")),
            },
            table="LlmOnChainStart",
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        new_relic_monitor.record_event(
            {
                "outputs": outputs.get("output"),
                "run_id": str(kwargs.get("run_id")),
                "tags": str(kwargs.get("tags")),
            },
            table="LlmOnChainEnd",
        )

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        new_relic_monitor.record_event({}, "LlmOnChainError")
        print(f"on_chain_error")
        print("error", error)
        print("kwargs", kwargs)
        print()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        new_relic_monitor.record_event({}, "LlmOnToolStart")
        print(f"on_tool_start")
        print("serialized", serialized)
        print("input_str", input_str)
        print("kwargs", kwargs)
        print()

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        new_relic_monitor.record_event({}, "LlmOnToolEnd")
        print(f"on_tool_end")
        print("output", output)
        print("kwargs", kwargs)
        print()

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        new_relic_monitor.record_event({}, "LlmOnToolError")
        print(f"on_tool_error")
        print("output", error)
        print("kwargs", kwargs)
        print()

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        print(f"on_text")
        print("text", text)
        print("kwargs", kwargs)
        print()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        new_relic_monitor.record_event(
            {
                "tool_name": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
                "run_id": str(kwargs.get("run_id")),
                "tags": str(kwargs.get("tags")),
                "metadata": str(kwargs.get("metadata")),
            },
            table="LlmOnAgentAction",
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        new_relic_monitor.record_event(
            {
                "return_values": finish.return_values.get("output"),
                "log": finish.log,
                "run_id": str(kwargs.get("run_id")),
                "tags": str(kwargs.get("tags")),
                "metadata": str(kwargs.get("metadata")),
            },
            table="LlmOnAgentFinish",
        )
