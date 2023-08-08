from collections import deque
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult
from newrelic_telemetry_sdk import Span

from nr_openai_observability import monitor


class NewRelicCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        application_name: str,
        **kwargs: Any,
    ) -> None:
        """Initialize callback handler."""
        self.application_name = application_name

        self.new_relic_monitor = monitor.initialization(
            application_name=application_name,
            parent_span_id_callback=self.parent_id_callback,
            **kwargs,
        )
        self.spans_stack = deque()
        self.tool_invocation_counter = 0

    def get_and_update_tool_invocation_counter(self):
        self.tool_invocation_counter += 1
        return self.tool_invocation_counter

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print("on_llm_start")
        print("serialized", serialized)
        print("prompts", prompts)
        print("kwargs", kwargs)
        print()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        invocation_params = kwargs.get("invocation_params", {})
        tags = {
            "model": invocation_params.get("model"),
            "model_name": invocation_params.get("model_name"),
            "temperature": invocation_params.get("temperature"),
            "request_timeout": invocation_params.get("request_timeout"),
            "max_tokens": invocation_params.get("max_tokens"),
            "stream": invocation_params.get("stream"),
            "n": invocation_params.get("n"),
            "temperature": invocation_params.get("temperature"),
        }

        parent_span = self.spans_stack[-1] if self.spans_stack else None
        parent_span_id = parent_span["id"] if parent_span else None
        self.spans_stack.append(
            Span(name="LlmCompletion", tags=tags, parent_id=parent_span_id)
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        tags = {}
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmCompletion"
        self.new_relic_monitor.record_span(span)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        tags = {"error": str(error)}
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmCompletion"
        self.new_relic_monitor.record_span(span)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        tags = {
            "input": inputs.get("input"),
            "run_id": str(kwargs.get("run_id")),
            "start_tags": str(kwargs.get("tags")),
            "start_metadata": str(kwargs.get("metadata")),
        }
        parent_span = self.spans_stack[-1] if self.spans_stack else None
        parent_span_id = parent_span["id"] if parent_span else None
        self.spans_stack.append(
            Span(name="LlmChain", tags=tags, parent_id=parent_span_id)
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        tags = {
            "outputs": outputs.get("output"),
            "run_id": str(kwargs.get("run_id")),
            "end_tags": str(kwargs.get("tags")),
        }
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmChain"
        self.new_relic_monitor.record_span(span)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        tags = {error: str(error)}
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmChain"
        self.new_relic_monitor.record_span(span)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        tags = {
            "tool_name": serialized.get("name"),
            "tool_description": serialized.get("description"),
            "tool_input": input_str,
        }
        parent_span = self.spans_stack[-1] if self.spans_stack else None
        parent_span_id = parent_span["id"] if parent_span else None
        self.spans_stack.append(
            Span(name="LlmTool", tags=tags, parent_id=parent_span_id)
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        tags = {
            "tool_output": output,
            "tool_invocation_counter": self.get_and_update_tool_invocation_counter(),
        }
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmTool"
        tool_name = kwargs.get("name")
        assert span["attributes"]["tool_name"] == tool_name
        self.new_relic_monitor.record_span(span)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        tags = {
            "error": error,
        }
        span = self.spans_stack.pop()
        span["attributes"].update(tags)
        span.finish()
        assert span["attributes"]["name"] == "LlmTool"
        tool_name = kwargs.get("name")
        assert span["attributes"]["tool_name"] == tool_name
        self.new_relic_monitor.record_span(span)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

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
        self.new_relic_monitor.record_event(
            tags,
            table="LlmOnAgentAction",
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.new_relic_monitor.record_event(
            {
                "return_values": finish.return_values.get("output"),
                "log": finish.log,
                "run_id": str(kwargs.get("run_id")),
                "tags": str(kwargs.get("tags")),
                "metadata": str(kwargs.get("metadata")),
            },
            table="LlmOnAgentFinish",
        )

    def parent_id_callback(self) -> str:
        parent_span = self.spans_stack[-1] if self.spans_stack else None
        return parent_span["id"] if parent_span else None
