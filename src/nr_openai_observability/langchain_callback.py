import re
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult

from nr_openai_observability import monitor
from nr_openai_observability.consts import CompletionEventName, ChainEventName, ToolEventName
import newrelic.agent
from nr_openai_observability.build_events import build_messages_events
from nr_openai_observability.consts import MessageEventName
from nr_openai_observability.call_vars import (
    set_conversation_id,
    get_response_model,
    get_completion_id,
    set_message_id,
    get_message_id,
)

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
        self.new_relic_monitor.record_library('langchain', 'LangChain')


    def get_and_update_tool_invocation_counter(self):
        self.tool_invocation_counter += 1
        return self.tool_invocation_counter

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self._save_metadata(kwargs.get("metadata", {}))
        model = self._get_model(serialized, **kwargs)

        tags = {
            "messages": "\n".join(prompts),
            "model_name": model,
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
        self._save_metadata(kwargs.get("metadata", {}))
        invocation_params = kwargs.get("invocation_params", {})
        model = self._get_model(serialized, **kwargs)

        tags = {
            "messages": "\n".join([f"{x.type}: {x.content}" for x in messages[0]]),
            "model": model,
            "model_name": invocation_params.get("model_name") or model,
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
        self._finish_segment(kwargs["run_id"], tags, CompletionEventName)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        tags = {"error": str(error)}
        self._finish_segment(kwargs["run_id"], tags, CompletionEventName)

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
        self._finish_segment(kwargs["run_id"], tags, ChainEventName)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        tags = {"error": str(error)}
        self._finish_segment(kwargs["run_id"], tags, ChainEventName)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        # we don't know if the tool is actually for Pinecone, but this is a best guess if 
        # the module is in scope.
        self.new_relic_monitor.record_library('pinecone-client', 'Pinecone')
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
        self._finish_segment(kwargs["run_id"], tags, ToolEventName)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        tags = {
            "error": str(error),
        }
        newrelic.agent.notice_error()
        self._finish_segment(kwargs["run_id"], tags, ToolEventName)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        final_message = {
            "role": "assistant",
            "content": finish.return_values.get("output"),
        }
        response_message = build_messages_events(
            [final_message],
            get_response_model(),
            get_completion_id(),
            get_message_id(),
            None,
            {"is_returned_langchain_message": True},
        )[0]

        self.new_relic_monitor.record_event(response_message, MessageEventName)
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

    def _finish_segment(self, run_id, tags={}, event_name=None):
        stack = self.trace_stacks.get(run_id, [])
        if stack != None and len(stack) != 0:
            trace = stack.pop()

            if len(stack) == 0:
                self.trace_stacks.pop(run_id, None)

            for key, val in tags.items():
                trace.add_custom_attribute(key, val)

            trace.__exit__(None, None, None)

            if event_name:
                attrs = trace.user_attributes

                if tags:
                    attrs.update(tags)

                attrs['trace.id'] = getattr(newrelic.agent.current_transaction(), "trace_id", None) or trace.guid
                attrs['guid'] = trace.guid
                attrs['parent.id'] = None
                attrs['duration.ms'] = trace.duration * 1000

                self.new_relic_monitor.record_event(attrs, event_name)

            return trace

    def _get_model(self, serialized: Dict[str, Any], **kwargs: Any) -> str:
        invocation_params = kwargs.get("invocation_params", {})
        model = invocation_params.get("model")
        if not model:
            model = invocation_params.get("model_id")
        if not model:
            if 'repr' in serialized:
                match = re.match(".*model_id='([^']+)'.*", serialized['repr'])
                if match:
                    model = match.group(1)
        if not model:
            model = invocation_params.get("_type", "")

        return model

    def _save_metadata(self, metadata):
        set_conversation_id(metadata.get("conversation_id", None))
        set_message_id(metadata.get("message_id", None))