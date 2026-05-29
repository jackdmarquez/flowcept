"""Tests for AMQP broker interception."""

# ruff: noqa: D103

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from flowcept.commons.sanitization import REDACTED
from flowcept.flowcept_api.flowcept_controller import Flowcept
from flowcept.flowceptor.adapters.brokers.amqp_interceptor import AMQPBrokerInterceptor


def _interceptor():
    interceptor = AMQPBrokerInterceptor()
    interceptor._max_payload_bytes = 64
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
    channel.queue_declare.assert_called_once_with(queue="", durable=False, exclusive=True, auto_delete=True)
    channel.queue_bind.assert_called_once_with(
        exchange="intersect-messages",
        queue="observer-queue",
        routing_key="#",
    )
    channel.basic_qos.assert_called_once_with(prefetch_count=100)


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
