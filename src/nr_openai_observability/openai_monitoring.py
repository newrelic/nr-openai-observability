import os
import logging
from typing import Any, Callable, Dict, Optional

import nr_openai_observability.consts as consts

import newrelic.agent

logger = logging.getLogger("nr_openai_observability")


class OpenAIMonitoring:
    # this class uses the telemetry SDK to record metrics to new relic, please see https://github.com/newrelic/newrelic-telemetry-sdk-python       
    def __init__(
        self,
        use_logger: Optional[bool] = None,
    ):
        self.use_logger = use_logger if use_logger else False
        self.headers_by_id: dict = {}
        self.initialized = False

    def _set_metadata(
        self,
        metadata: Dict[str, Any] = {},
    ):
        self.metadata = metadata

        if not isinstance(metadata, Dict) and metadata is not None:
            raise TypeError("metadata instance type must be Dict[str, Any]")

    def _log(self, msg: str):
        if self.use_logger:
            logger.info(msg)
        else:
            print(msg)

    def start(
        self,
        application_name: str,
        metadata: Dict[str, Any] = {},
        metadata_callback: Optional[Callable] = None,
    ):
        if not self.initialized:
            self.application_name = application_name
            self._set_metadata(metadata)
            self.metadata_callback = metadata_callback
            self._start()
            self.initialized = True

    # initialize event thread
    def _start(self):
        None

    def record_event(
        self,
        event_dict: dict,
        table: str = consts.EventName,
    ):
        event_dict.update(self.metadata)
        if self.metadata_callback:
            try:
                metadata = self.metadata_callback(event_dict)
                if metadata:
                    event_dict.update(metadata)
            except Exception as ex:
                logger.warning(f"Failed to run metadata callback: {ex}")
        newrelic.agent.record_custom_event(table, event_dict)

monitor = OpenAIMonitoring()
