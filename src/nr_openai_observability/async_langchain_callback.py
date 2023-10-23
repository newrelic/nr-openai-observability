from typing import Any, Dict, List, Union

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult

from nr_openai_observability.langchain_callback import NewRelicCallbackHandler


class NewRelicAsyncCallbackHandler(AsyncCallbackHandler):
    def __init__(
        self,
        application_name: str,
        langchain_callback_metadata: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize callback handler."""
        self.sync_handler = NewRelicCallbackHandler(
            application_name, langchain_callback_metadata, **kwargs
        )

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.sync_handler.on_llm_start(serialized, prompts, **kwargs)

    # TODO - Why is there no corresponding end method for this callback? How do we set up spans without this?
    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self.sync_handler.on_chat_model_start(serialized, messages, **kwargs)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.sync_handler.on_llm_new_token(token, **kwargs)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.sync_handler.on_llm_end(response, **kwargs)

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.sync_handler.on_llm_error(error, **kwargs)

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        self.sync_handler.on_chain_start(serialized, inputs, **kwargs)

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.sync_handler.on_chain_end(outputs, **kwargs)

    async def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        self.sync_handler.on_chain_error(error, **kwargs)

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.sync_handler.on_tool_start(serialized, input_str, **kwargs)

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.sync_handler.on_tool_end(output, **kwargs)

    async def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.sync_handler.on_tool_error(error, **kwargs)

    async def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        self.sync_handler.on_text(text, **kwargs)

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.sync_handler.on_agent_action(action, **kwargs)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.sync_handler.on_agent_finish(finish, **kwargs)
