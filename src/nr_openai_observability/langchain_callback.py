import logging
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult

from nr_openai_observability import monitor
import newrelic.agent


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
            **kwargs,
        )
        self.langchain_callback_metadata = langchain_callback_metadata
        self.tool_invocation_counter = 0
        self.trace_stacks = {}

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
        trace = newrelic.agent.FunctionTrace(name="AI/LangChain/RunLLM", terminal=False)
        self._start_segment(kwargs["run_id"], trace, tags)

    # TODO - Why is there no corresponding end method for this callback? How do we set up spans without this?
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
        trace = newrelic.agent.FunctionTrace(
            name="AI/LangChain/RunChatModel", terminal=False
        )
        self._start_segment(kwargs["run_id"], trace, tags)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

        tags = {
            "response": response.generations[0][0].text,
        }

        llm_output = response.llm_output
        if llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            tags.update(
                {
                    "prompt_tokens": token_usage.get("prompt_tokens", None),
                    "completion_tokens": token_usage.get("completion_tokens", None),
                    "total_tokens": token_usage.get("total_tokens", None),
                }
            )
        self._finish_segment(kwargs["run_id"])

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        tags = {"error": str(error)}
        self._finish_segment(kwargs["run_id"], tags)

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

        trace = newrelic.agent.FunctionTrace(
            name="AI/LangChain/RunChain", terminal=False
        )
        tags = {
            "input": inputs.get("input") or inputs.get("human_input") or "",
            "chat_history": chat_history,
            "run_id": str(kwargs.get("run_id")),
            "start_tags": str(kwargs.get("tags")),
            "start_metadata": str(kwargs.get("metadata")),
        }
        self._start_segment(kwargs["run_id"], trace, tags)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        tags = {
            "outputs": outputs.get("output"),
            "run_id": str(kwargs.get("run_id")),
            "end_tags": str(kwargs.get("tags")),
        }
        self._finish_segment(kwargs["run_id"], tags)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        tags = {"error": str(error)}
        self._finish_segment(kwargs["run_id"], tags)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        tool_name = serialized.get("name")
        trace = newrelic.agent.FunctionTrace(
            name=f"AI/LangChain/Tool/{tool_name}", terminal=False
        )
        tags = {
            "tool_name": tool_name,
            "tool_description": serialized.get("description"),
            "tool_input": input_str,
        }
        self._start_segment(kwargs["run_id"], trace, tags)

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        tags = {
            "tool_output": output,
            "tool_invocation_counter": self.get_and_update_tool_invocation_counter(),
        }
        self._finish_segment(kwargs["run_id"], tags)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        tags = {
            "error": str(error),
        }
        newrelic.agent.notice_error()
        self._finish_segment(kwargs["run_id"], tags)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        # self._finish_segment(kwargs["run_id"])

    def _start_segment(self, run_id, trace, tags={}):
        trace.__enter__()
        if self.langchain_callback_metadata:
            tags = tags or {}
            tags.update(self.langchain_callback_metadata)

        for key, val in tags.items():
            trace.add_custom_attribute(key, val)

        stack = self.trace_stacks.get(run_id, [])
        stack.append(trace)

        self.trace_stacks[run_id] = stack

    def _finish_segment(self, run_id, tags={}):
        stack = self.trace_stacks.get(run_id, [])
        if stack != None and len(stack) != 0:
            trace = stack.pop()

            if len(stack) == 0:
                self.trace_stacks.pop(run_id, None)

            for key, val in tags.items():
                trace.add_custom_attribute(key, val)

            trace.__exit__(None, None, None)

            return trace
