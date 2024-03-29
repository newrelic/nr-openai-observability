import logging
from typing import Any, Callable, Dict, Optional

from nr_openai_observability.patcher import perform_patch
from nr_openai_observability.openai_monitoring import monitor

logger = logging.getLogger("nr_openai_observability")


def initialization(
    application_name: str,
    license_key: Optional[str] = None,
    metadata: Dict[str, Any] = {},
    event_client_host: Optional[str] = None,
    parent_span_id_callback: Optional[Callable] = None,
    metadata_callback: Optional[Callable] = None,
):
    monitor.start(
        application_name,
        license_key,
        metadata,
        event_client_host,
        parent_span_id_callback,
        metadata_callback,
    )
    perform_patch()
    return monitor
