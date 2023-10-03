import logging
from typing import Any, Callable, Dict, Optional

from nr_openai_observability.patcher import perform_patch
from nr_openai_observability.openai_monitoring import monitor
from nr_openai_observability.build_events import build_ai_feedback_event
from nr_openai_observability.consts import FeedbackEventName

logger = logging.getLogger("nr_openai_observability")


def initialization(
    application_name: str,
    metadata: Dict[str, Any] = {},
    metadata_callback: Optional[Callable] = None,
):
    monitor.start(
        application_name,
        metadata,
        metadata_callback,
    )
    perform_patch()
    return monitor

def record_ai_feedback_event(rating, message_id, category = None, conversation_id = None, request_id = None, message = None):
    event = build_ai_feedback_event(category, rating, message_id, conversation_id, request_id, message)

    monitor.record_event(event, FeedbackEventName)
