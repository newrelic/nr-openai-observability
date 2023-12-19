import contextvars
import logging
import newrelic.agent
from newrelic.api.transaction import current_transaction

logger = logging.getLogger("nr_openai_observability")

conversation_id = contextvars.ContextVar("conversation_id")


def get_ai_message_ids(response_id=None):
    if getattr(newrelic.agent, "get_llm_message_ids", None):
        message_ids = newrelic.agent.get_llm_message_ids(response_id)
        if len(message_ids) > 0:
            return message_ids

    transaction = current_transaction()

    if transaction is None:
        logger.debug("AI message IDs need a transaction to be retrieved.")

    ai_message_ids = getattr(transaction, "_ai_sdk_message_ids", {})
    if response_id is not None:
        # OpenAI
        return ai_message_ids.get(response_id, [])
    else:
        # Bedrock
        return ai_message_ids


def set_ai_message_ids(message_ids, response_id=None):
    transaction = current_transaction()
    if transaction is None:
        logger.debug("AI message IDs need a transaction to be stored.")
        return

    if response_id is not None:
        # OpenAI
        current_ids = getattr(transaction, "_ai_sdk_message_ids", {})
        current_ids[response_id] = message_ids
        transaction._ai_sdk_message_ids = current_ids
    else:
        # Bedrock
        transaction._ai_sdk_message_ids = message_ids


def set_conversation_id(id):
    conversation_id.set(id)


def get_conversation_id():
    return conversation_id.get(None)


def create_ai_message_id(message_id, request_id=None):
    return {
        "conversation_id": get_conversation_id(),
        "request_id": request_id,
        "message_id": message_id,
    }
