import contextvars

conversation_id = contextvars.ContextVar('conversation_id')
ai_message_ids = contextvars.ContextVar('ai_message_ids')

def get_ai_message_ids(response_id=None):
    if response_id is not None:
        #OpenAI
        return ai_message_ids.get({}).get(response_id, [])
    else:
        #Bedrock
        return ai_message_ids.get([])

def set_ai_message_ids(message_ids, response_id=None):
    if response_id is not None:
        #OpenAI
        current_ids = ai_message_ids.get({})
        current_ids[response_id] = message_ids
        ai_message_ids.set(current_ids)
    else:
        #Bedrock
        ai_message_ids.set(message_ids)

def set_conversation_id(id):
    conversation_id.set(id)

def get_conversation_id():
    return conversation_id.get(None)

def create_ai_message_id(message_id, response_id=None):
    return {
        "conversation_id": get_conversation_id(),
        "response_id": response_id,
        "message_id": message_id,
    }