from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult
from newrelic_telemetry_sdk import Span

from nr_openai_observability import monitor
from nr_openai_observability.monitor import monitor as new_relic_monitor

from collections import deque


class NewRelicCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        application_name: str,
    ) -> None:
        """Initialize callback handler."""
        self.application_name = application_name

        monitor.initialization(application_name=application_name)
        self.spans_stack = deque()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        new_relic_monitor.record_event({}, "LlmOnLLMStart")
        print("on_llm_start")
        print("serialized", serialized)
        print("prompts", prompts)
        print("kwargs", kwargs)
        print()

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        print("on_chat_model_start")
        print("serialized", serialized)
        print("messages", messages)
        print("kwargs", kwargs)
        print()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print("on_llm_end")
        print("response", response)
        print("LLMResult", LLMResult)
        print("kwargs", kwargs)
        print()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        print("on_llm_error")
        print("error", error)
        print("kwargs", kwargs)
        print()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        tags = {
                "input": inputs.get("input"),
                "start_run_id": str(kwargs.get("run_id")),
                "start_tags": str(kwargs.get("tags")),
                "start_metadata": str(kwargs.get("metadata")),
            }
        new_relic_monitor.record_event(
            tags,
            table="LlmOnChainStart",
        )
        parent_span = self.spans_stack[-1] if self.spans_stack else None
        parent_span_id = parent_span["id"] if parent_span else None
        self.spans_stack.append(Span(name="LlmChain", tags=tags, parent_id=parent_span_id))
        

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        tags = {
                "outputs": outputs.get("output"),
                "end_run_id": str(kwargs.get("run_id")),
                "end_tags": str(kwargs.get("tags")),
            }
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmChain"
        new_relic_monitor.record_span(span)

        new_relic_monitor.record_event(
            tags,
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
        tags = {}
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmChain"
        new_relic_monitor.record_span(span)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        tags = {
                "tool_name": serialized.get("name"),
                "tool_description": serialized.get("description"),
                "tool_input": input_str,
        }   
        new_relic_monitor.record_event(tags, "LlmOnToolStart")
        parent_span = self.spans_stack[-1] if self.spans_stack else None
        parent_span_id = parent_span["id"] if parent_span else None
        self.spans_stack.append(Span(name="LlmTool", tags=tags, parent_id=parent_span_id))

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        tags = {
            "tool_output": output,
        }
        new_relic_monitor.record_event(tags, "LlmOnToolEnd")

        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmTool"
        tool_name = kwargs.get("name")
        assert span["attributes"]["tool_name"] == tool_name
        new_relic_monitor.record_span(span)

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
        tags = {
                "tool_name": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
                "run_id": str(kwargs.get("run_id")),
                "tags": str(kwargs.get("tags")),
                "metadata": str(kwargs.get("metadata")),
            }
        new_relic_monitor.record_event(
            tags,
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
