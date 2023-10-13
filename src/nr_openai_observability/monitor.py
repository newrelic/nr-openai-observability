import logging
from typing import Any, Callable, Dict, Optional

from nr_openai_observability.patcher import perform_patch
from nr_openai_observability.openai_monitoring import monitor

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
