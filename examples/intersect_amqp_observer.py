"""Run a long-lived Flowcept AMQP observer for INTERSECT traffic."""

import signal
import time

from flowcept import Flowcept

running = True


def stop_handler(*_):
    """Handle container shutdown signals."""
    global running
    running = False


signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)

fc = Flowcept(
    interceptors=["broker_amqp"],
    workflow_name="intersect-amqp-provenance-observer",
    workflow_subtype="broker_observer",
    start_persistence=True,
    check_safe_stops=False,
)
fc.start()

try:
    while running:
        time.sleep(5)
finally:
    fc.stop()
