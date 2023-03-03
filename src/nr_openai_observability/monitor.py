import atexit
import logging
import os
import time
from typing import Optional

import openai
from newrelic_telemetry_sdk import Event, EventBatch, EventClient, Harvester

logger = logging.getLogger("nr_openai_observability")

EventName = "OpenAICompletion"


def _patched_call(original_fn, patched_fn):
    def _inner_patch(*args, **kwargs):
        try:
            return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    return _inner_patch


class OpenAIMonitoring:
    # this class uses the telemetry SDK to record metrics to new relic, please see https://github.com/newrelic/newrelic-telemetry-sdk-python
    def __init__(
        self,
        event_client_host: Optional[str] = None,
        use_logger: Optional[bool] = None,
    ):
        if not isinstance(event_client_host, str) and event_client_host is not None:
            raise TypeError("event_client_host instance type must be str or None")

        self.event_client_host = event_client_host or os.getenv(
            "EVENT_CLIENT_HOST", EventClient.HOST
        )

        self.use_logger = use_logger if use_logger else False

    def _set_license_key(
        self,
        license_key: Optional[str] = None,
    ):
        self.license_key = (
            license_key
            or os.getenv("NEW_RELIC_LICENSE_KEY")
            or os.getenv("NEW_RELIC_INSERT_KEY")
        )  # type: ignore

        if (
            not isinstance(self.license_key, str) and self.license_key is not None
        ) or self.license_key is None:
            raise TypeError("license_key instance type must be str and not None")

    def _log(self, msg: str):
        if self.use_logger:
            logger.info(msg)
        else:
            print(msg)

    def start(
        self,
        license_key: Optional[str] = None,
    ):
        self._set_license_key(license_key)
        self._start()

    # initialize event thread
    def _start(self):
        self.event_client = EventClient(
            self.license_key,
            host=self.event_client_host,
        )
        self.event_batch = EventBatch()

        # Background thread that flushes the batch
        self.event_harvester = Harvester(self.event_client, self.event_batch)

        # This starts the thread
        self.event_harvester.start()

        # When the process exits, run the harvester.stop() method before terminating the process
        # Why? To send the remaining data...
        atexit.register(self.event_harvester.stop)

    def record_event(self, event_dict: dict, table: str = EventName):
        event = Event(table, event_dict)
        self.event_batch.record(event)


def patcher_create(original_fn, *args, **kwargs):
    def flatten_dict(dd, separator=".", prefix="", index=""):
        if len(index):
            index = index + separator
        return (
            {
                prefix + separator + index + k if prefix else k: v
                for kk, vv in dd.items()
                for k, v in flatten_dict(vv, separator, kk).items()
            }
            if isinstance(dd, dict)
            else {prefix: dd}
        )

    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    result = original_fn(*args, **kwargs)
    time_delta = time.time() - timestamp

    logger.debug(
        f"Finished running function: '{original_fn.__qualname__}'. result: {result}"
    )

    choices_payload = {}
    for i, choice in enumerate(result.get("choices")):
        choices_payload.update(flatten_dict(choice, prefix="choices", index=str(i)))

    logger.debug(dict(**kwargs))

    event_dict = {
        **kwargs,
        "response_time": time_delta,
        **flatten_dict(result.to_dict_recursive(), separator="."),
        **choices_payload,
    }
    event_dict.pop("choices")

    if "messages" in event_dict:
        event_dict["messages"] = str(kwargs.get("messages"))

    logger.debug(f"Reported event dictionary:\n{event_dict}")

    monitor.record_event(event_dict)

    return result


monitor = OpenAIMonitoring()


def initialization(license_key: Optional[str] = None):
    monitor.start(license_key)
    perform_patch()


def perform_patch():
    try:
        openai.Completion.create = _patched_call(
            openai.Completion.create, patcher_create
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.create = _patched_call(
            openai.ChatCompletion.create, patcher_create
        )
    except AttributeError:
        pass
