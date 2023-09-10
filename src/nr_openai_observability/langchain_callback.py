import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult
from newrelic_telemetry_sdk import Span

from nr_openai_observability import monitor
from nr_openai_observability.build_events import span_to_event


class NewRelicCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        application_name: str,
        langchain_callback_metadata: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize callback handler."""
        self.application_name = application_name

        self.new_relic_monitor = monitor.initialization(
            application_name=application_name,
            parent_span_id_callback=self.parent_id_callback,  # may not be asyncio safe
            **kwargs,
        )
        self.langchain_callback_metadata = langchain_callback_metadata
        self.spans_stack = deque()
        self.tool_invocation_counter = 0
        self.trace_id = "%016x" % random.getrandbits(64)

    def get_and_update_tool_invocation_counter(self):
        self.tool_invocation_counter += 1
        return self.tool_invocation_counter

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        tags = {
            "messages": "\n".join(prompts),
            "model_name": kwargs.get("invocation_params", {}).get("_type", ""),
        }
        self.spans_stack.append(self.create_span(name="LlmCompletion", tags=tags))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        invocation_params = kwargs.get("invocation_params", {})
        tags = {
            "messages": "\n".join([f"{x.type}: {x.content}" for x in messages[0]]),
            "model": invocation_params.get("model"),
            "model_name": invocation_params.get("model_name"),
            "temperature": invocation_params.get("temperature"),
            "request_timeout": invocation_params.get("request_timeout"),
            "max_tokens": invocation_params.get("max_tokens"),
            "stream": invocation_params.get("stream"),
            "n": invocation_params.get("n"),
            "temperature": invocation_params.get("temperature"),
        }

        self.spans_stack.append(self.create_span(name="LlmCompletion", tags=tags))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

        tags = {
            "response": response.generations[0][0].text,
        }

        llm_output = response.llm_output
        if llm_output:
            tags.update(
                {
                    "prompt_tokens": response.llm_output.get(
                        "token_usage"
                    ).prompt_tokens,
                    "completion_tokens": response.llm_output.get(
                        "token_usage"
                    ).completion_tokens,
                    "total_tokens": response.llm_output.get("token_usage").total_tokens,
                }
            )
        span = self.spans_stack.pop()
        assert span["attributes"]["name"] == "LlmCompletion"
        self.finish_and_record_span(span, tags)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        tags = {"error": str(error)}
        span = self.spans_stack.pop()
        assert span["attributes"]["name"] == "LlmCompletion"
        self.finish_and_record_span(span, tags)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

        key = "chat_history" if inputs.get("chat_history") else "memory"
        chat_history = (
            "\n".join([f"{x.type}: {x.content}" for x in inputs.get(key, [])])
            if inputs.get(key)
            else ""
        )

        tags = {
            "input": inputs.get("input") or inputs.get("human_input") or "",
            "chat_history": chat_history,
            "run_id": str(kwargs.get("run_id")),
            "start_tags": str(kwargs.get("tags")),
            "start_metadata": str(kwargs.get("metadata")),
        }
        self.spans_stack.append(self.create_span(name="LlmChain", tags=tags))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        tags = {
            "outputs": outputs.get("output"),
            "run_id": str(kwargs.get("run_id")),
            "end_tags": str(kwargs.get("tags")),
        }
        span = self.spans_stack.pop()
        assert span["attributes"]["name"] == "LlmChain"
        self.finish_and_record_span(span, tags)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        tags = {"error": str(error)}
        span = self.spans_stack.pop()
        assert span["attributes"]["name"] == "LlmChain"
        self.finish_and_record_span(span, tags)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        tags = {
            "tool_name": serialized.get("name"),
            "tool_description": serialized.get("description"),
            "tool_input": input_str,
        }
        self.spans_stack.append(self.create_span(name="LlmTool", tags=tags))

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        tags = {
            "tool_output": output,
            "tool_invocation_counter": self.get_and_update_tool_invocation_counter(),
        }
        span = self.spans_stack.pop()

        assert span["attributes"]["name"] == "LlmTool"
        tool_name = kwargs.get("name")
        assert span["attributes"]["tool_name"] == tool_name

        self.finish_and_record_span(span, tags)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        tags = {"error": str(error)}
        span = self.spans_stack.pop()
        assert span["attributes"]["name"] == "LlmTool"
        tool_name = kwargs.get("name")
        assert span["attributes"]["tool_name"] == tool_name

        self.finish_and_record_span(span, tags)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""

    def parent_id_callback(self) -> str:
        parent_span = self.spans_stack[-1] if self.spans_stack else None
        return parent_span["id"] if parent_span else None

    def create_span(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        guid: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ):
        if parent_id is None:
            parent_span = self.spans_stack[-1] if self.spans_stack else None
            parent_id = parent_span["id"] if parent_span else None

        if self.langchain_callback_metadata:
            tags = tags or {}
            tags.update(self.langchain_callback_metadata)

        if not trace_id and "newrelic" in sys.modules:
            import newrelic.agent  # type: ignore

            trace_id = getattr(newrelic.agent.current_transaction(), "trace_id", None)

        trace_id = trace_id or self.trace_id

        span = Span(
            name,
            tags,
            guid,
            trace_id,
            parent_id,
            start_time_ms,
            duration_ms,
        )
        return span

    def finish_and_record_span(self, span: Span, tags: Optional[Dict[str, Any]] = None):
        span["attributes"].update(tags or {})
        span.finish()
        self.new_relic_monitor.record_span(span)
        self.new_relic_monitor.record_event(**span_to_event(span))
