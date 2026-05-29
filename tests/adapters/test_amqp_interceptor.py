"""Tests for AMQP broker interception."""

# ruff: noqa: D103

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from flowcept.commons.sanitization import REDACTED
from flowcept.flowcept_api.flowcept_controller import Flowcept
from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor
from flowcept.flowceptor.adapters.brokers.amqp_interceptor import AMQPBrokerInterceptor


def _interceptor():
    interceptor = AMQPBrokerInterceptor()
    interceptor._max_payload_bytes = 128
    return interceptor


def _method(routing_key="org.fac.sys.sub.service.request", exchange="intersect-messages", redelivered=False):
    return SimpleNamespace(
        exchange=exchange,
        routing_key=routing_key,
        redelivered=redelivered,
        delivery_tag=42,
    )


def _properties(**kwargs):
    defaults = {
        "message_id": None,
        "headers": {},
        "content_type": "application/json",
        "correlation_id": None,
        "reply_to": None,
        "delivery_mode": None,
        "priority": None,
        "timestamp": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_prepare_task_msg_uses_message_id_precedence_and_sanitizes_json_preview(monkeypatch):
    monkeypatch.setattr(Flowcept, "current_workflow_id", "wf-1")
    monkeypatch.setattr(Flowcept, "campaign_id", "camp-1")
    interceptor = _interceptor()

    task = interceptor.prepare_task_msg(
        method=_method(),
        properties=_properties(
            message_id="prop-id",
            headers={"messageId": "header-id", "Authorization": "Bearer secret"},
            correlation_id="corr",
        ),
        body=b'{"safe": 1, "password": "pw"}',
    )
    task_dict = task.to_dict()

    assert task_dict["task_id"] == "prop-id"
    assert task_dict["workflow_id"] == "wf-1"
    assert task_dict["campaign_id"] == "camp-1"
    assert task_dict["activity_id"] == "org.fac.sys.sub.service.request"
    assert task_dict["used"]["payload_preview"]["safe"] == 1
    assert task_dict["used"]["payload_preview"]["password"] == REDACTED
    assert task_dict["custom_metadata"]["headers"]["Authorization"] == REDACTED
    assert task_dict["custom_metadata"]["organization"] == "org"
    assert task_dict["custom_metadata"]["message_type"] == "request"


def test_prepare_task_msg_uses_header_message_id_then_deterministic_fallback():
    interceptor = _interceptor()
    body = b"hello"

    header_task = interceptor.prepare_task_msg(
        method=_method(routing_key="org.fac"),
        properties=_properties(headers={"messageId": "header-id"}, content_type="text/plain"),
        body=body,
    )
    fallback_1 = interceptor.prepare_task_msg(
        method=_method(routing_key="org.fac"),
        properties=_properties(content_type="text/plain", correlation_id="corr", timestamp=123),
        body=body,
    )
    fallback_2 = interceptor.prepare_task_msg(
        method=_method(routing_key="org.fac"),
        properties=_properties(content_type="text/plain", correlation_id="corr", timestamp=123),
        body=body,
    )

    assert header_task.task_id == "header-id"
    assert fallback_1.task_id == fallback_2.task_id
    assert fallback_1.task_id != "header-id"


def test_prepare_task_msg_uses_json_payload_message_id_and_operation_id():
    interceptor = _interceptor()

    task = interceptor.prepare_task_msg(
        method=_method(routing_key="org.fac.sys.sub.service.event"),
        properties=_properties(message_id=None, headers={}),
        body=b'{"messageId": "payload-id", "operationId": "IntersectChess.collect"}',
    )
    task_dict = task.to_dict()

    assert task_dict["task_id"] == "payload-id"
    assert task_dict["activity_id"] == "IntersectChess.collect"
    assert task_dict["custom_metadata"]["intersect_message_id"] == "payload-id"
    assert task_dict["custom_metadata"]["intersect_operation_id"] == "IntersectChess.collect"


def test_payload_preview_size_text_and_binary_handling():
    interceptor = _interceptor()
    interceptor._max_payload_bytes = 4

    large = interceptor.prepare_task_msg(
        method=_method(),
        properties=_properties(content_type="text/plain"),
        body=b"12345",
    )
    text = interceptor.prepare_task_msg(
        method=_method(),
        properties=_properties(content_type="text/plain"),
        body=b"hey",
    )
    binary = interceptor.prepare_task_msg(
        method=_method(),
        properties=_properties(content_type="application/octet-stream"),
        body=b"\xff\xfe",
    )

    assert "payload_preview" not in large.used
    assert text.used["payload_preview"] == "hey"
    assert binary.used["payload_preview"] == "fffe"


def test_parse_routing_key_is_defensive():
    interceptor = _interceptor()

    short = interceptor._parse_routing_key("org.fac")
    long = interceptor._parse_routing_key("org.fac.sys.sub.service.message.extra")

    assert short["routing_parts"] == ["org", "fac"]
    assert short["organization"] == "org"
    assert short["facility"] == "fac"
    assert "system" not in short
    assert long["message_type"] == "message"
    assert long["routing_parts"][-1] == "extra"


def test_setup_observer_queue_passive_exchange_generated_exclusive_queue():
    interceptor = _interceptor()
    channel = MagicMock()
    channel.queue_declare.return_value = SimpleNamespace(method=SimpleNamespace(queue="observer-queue"))
    interceptor._channel = channel

    interceptor._setup_observer_queue()

    channel.exchange_declare.assert_called_once_with(
        exchange="intersect-messages",
        exchange_type="topic",
        durable=True,
        passive=True,
    )
    channel.queue_declare.assert_called_once_with(
        queue="",
        durable=False,
        exclusive=True,
        auto_delete=True,
        arguments={},
    )
    channel.queue_bind.assert_called_once_with(
        exchange="intersect-messages",
        queue="observer-queue",
        routing_key="#",
    )
    channel.basic_qos.assert_called_once_with(prefetch_count=100)


def test_setup_observer_queue_passes_queue_arguments():
    interceptor = _interceptor()
    interceptor._observer_queue["arguments"] = {"x-message-ttl": 86400000, "x-max-length": 100000}
    channel = MagicMock()
    channel.queue_declare.return_value = SimpleNamespace(method=SimpleNamespace(queue="observer-queue"))
    interceptor._channel = channel

    interceptor._setup_observer_queue()

    channel.queue_declare.assert_called_once_with(
        queue="",
        durable=False,
        exclusive=True,
        auto_delete=True,
        arguments={"x-message-ttl": 86400000, "x-max-length": 100000},
    )


def test_process_data_events_loop_exits_when_stopping_is_set():
    interceptor = _interceptor()
    connection = MagicMock()
    connection.process_data_events.side_effect = lambda time_limit: interceptor._stopping.set()
    interceptor._connection = connection

    interceptor._process_data_events_until_stopped()

    connection.process_data_events.assert_called_once_with(time_limit=1.0)


def test_observe_consumes_only_observer_queue_and_exits_on_stop():
    interceptor = _interceptor()
    channel = MagicMock()
    connection = MagicMock()
    connection.process_data_events.side_effect = lambda time_limit: interceptor._stopping.set()
    channel.queue_declare.return_value = SimpleNamespace(method=SimpleNamespace(queue="observer-queue"))
    channel.is_open = True
    connection.is_open = True
    interceptor._connect = MagicMock(side_effect=lambda: setattr(interceptor, "_connection", connection))
    interceptor._channel = channel
    interceptor._connect.side_effect = lambda: (
        setattr(interceptor, "_connection", connection),
        setattr(interceptor, "_channel", channel),
    )

    interceptor.observe()

    channel.basic_consume.assert_called_once_with(
        queue="observer-queue",
        on_message_callback=interceptor.callback,
        auto_ack=False,
    )
    assert channel.basic_consume.call_args.kwargs["queue"] != "production-queue"


def test_observe_retries_after_recoverable_amqp_exception(monkeypatch):
    import flowcept.flowceptor.adapters.brokers.amqp_interceptor as amqp_module

    interceptor = _interceptor()
    channel = MagicMock()
    connection = MagicMock()
    connection.process_data_events.side_effect = lambda time_limit: interceptor._stopping.set()
    channel.queue_declare.return_value = SimpleNamespace(method=SimpleNamespace(queue="observer-queue"))
    channel.is_open = True
    connection.is_open = True
    attempts = {"count": 0}

    def connect():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise amqp_module.pika.exceptions.StreamLostError("connection dropped")
        interceptor._connection = connection
        interceptor._channel = channel

    interceptor._connect = MagicMock(side_effect=connect)
    interceptor._sleep_before_reconnect = MagicMock()

    interceptor.observe()

    assert interceptor._connect.call_count == 2
    interceptor._sleep_before_reconnect.assert_called_once()
    channel.basic_consume.assert_called_once_with(
        queue="observer-queue",
        on_message_callback=interceptor.callback,
        auto_ack=False,
    )


def test_callback_acks_after_successful_intercept(monkeypatch):
    interceptor = _interceptor()
    channel = MagicMock()
    interceptor.intercept = MagicMock()
    monkeypatch.setattr(Flowcept, "current_workflow_id", "wf")
    monkeypatch.setattr(Flowcept, "campaign_id", "camp")

    interceptor.callback(
        channel,
        _method(),
        _properties(message_id="id"),
        b"{}",
    )

    interceptor.intercept.assert_called_once()
    channel.basic_ack.assert_called_once_with(delivery_tag=42)
    channel.basic_nack.assert_not_called()


def test_callback_nacks_without_requeue_on_conversion_failure():
    interceptor = _interceptor()
    channel = MagicMock()
    interceptor.prepare_task_msg = MagicMock(side_effect=ValueError("bad message"))

    interceptor.callback(channel, _method(), _properties(), b"{}")

    channel.basic_ack.assert_not_called()
    channel.basic_nack.assert_called_once_with(delivery_tag=42, requeue=False)


def test_missing_pika_error_is_explicit(monkeypatch):
    import flowcept.flowceptor.adapters.brokers.amqp_interceptor as amqp_module

    monkeypatch.setattr(amqp_module, "pika", None)
    interceptor = _interceptor()

    with pytest.raises(ModuleNotFoundError, match="flowcept\\[amqp\\]"):
        interceptor._connect()


def test_stop_does_not_close_amqp_from_main_thread(monkeypatch):
    interceptor = _interceptor()
    channel = MagicMock()
    connection = MagicMock()
    interceptor._channel = channel
    interceptor._connection = connection
    monkeypatch.setattr(BaseInterceptor, "stop", lambda *args, **kwargs: None)

    interceptor.stop()

    assert interceptor._stopping.is_set()
    channel.close.assert_not_called()
    connection.close.assert_not_called()
