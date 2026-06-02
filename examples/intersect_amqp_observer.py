"""Run a long-lived Flowcept AMQP observer for INTERSECT traffic."""

import signal
import time

from flowcept import Flowcept
from flowcept.configs import DB_FLUSH_MODE, INSERTION_BUFFER_TIME, MONGO_ENABLED, MQ_ENABLED, SETTINGS_PATH
from flowcept.commons.flowcept_logger import FlowceptLogger

running = True
logger = FlowceptLogger()


def stop_handler(*_):
    """Handle container shutdown signals."""
    global running
    running = False


def validate_persistence_settings():
    """Fail fast when the observer cannot stream provenance into Mongo."""
    if not MQ_ENABLED or not MONGO_ENABLED or DB_FLUSH_MODE != "online" or INSERTION_BUFFER_TIME is None:
        raise RuntimeError(
            "INTERSECT AMQP observation requires Flowcept Redis MQ, MongoDB, and online "
            "flush mode with db_buffer.insertion_buffer_time_secs set for live persistence. "
            f"Current settings path: {SETTINGS_PATH}; "
            f"MQ_ENABLED={MQ_ENABLED}; MONGO_ENABLED={MONGO_ENABLED}; DB_FLUSH_MODE={DB_FLUSH_MODE!r}; "
            f"INSERTION_BUFFER_TIME={INSERTION_BUFFER_TIME!r}. "
            "Set FLOWCEPT_SETTINGS_PATH to an observer settings file with db_buffer.insertion_buffer_time_secs, "
            "or export MQ_ENABLED=true MONGO_ENABLED=true DB_FLUSH_MODE=online and use settings that define "
            "a db_buffer flush interval before starting."
        )


signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)

validate_persistence_settings()

logger.info(
    "Starting INTERSECT AMQP observer "
    f"settings_path={SETTINGS_PATH} db_flush_mode={DB_FLUSH_MODE} "
    f"mongo_enabled={MONGO_ENABLED} mq_enabled={MQ_ENABLED} "
    f"db_flush_interval_secs={INSERTION_BUFFER_TIME}"
)

fc = Flowcept(
    interceptors=["broker_amqp"],
    workflow_name="intersect-amqp-provenance-observer",
    workflow_subtype="broker_observer",
    start_persistence=True,
    check_safe_stops=False,
)
fc.start()
logger.info("INTERSECT AMQP observer is running. Press Ctrl+C to stop.")

try:
    while running:
        time.sleep(5)
finally:
    logger.info("Stopping INTERSECT AMQP observer.")
    fc.stop()
    logger.info("INTERSECT AMQP observer stopped.")
