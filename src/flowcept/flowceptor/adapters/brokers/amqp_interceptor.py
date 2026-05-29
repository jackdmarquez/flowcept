"""AMQP broker interceptor for observing INTERSECT traffic."""

from __future__ import annotations

import json
import os
import threading
from hashlib import sha256
from time import sleep
from typing import Any, Dict

try:
    import pika
except ModuleNotFoundError:  # pragma: no cover - exercised when optional dependency is absent
    pika = None
from omegaconf import DictConfig, OmegaConf

from flowcept.commons.flowcept_dataclasses.task_object import TaskObject
from flowcept.commons.sanitization import sanitize_value
from flowcept.flowcept_api.flowcept_controller import Flowcept
from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor


class AMQPBrokerInterceptor(BaseInterceptor):
    """Observe RabbitMQ AMQP 0.9.1 messages without consuming production queues."""

    _ROUTING_FIELDS = ("organization", "facility", "system", "subsystem", "service", "message_type")

    def __init__(self, plugin_key="broker_amqp"):
        super().__init__(plugin_key)
        self._settings = (
            OmegaConf.to_container(self.settings, resolve=False)
            if isinstance(self.settings, DictConfig)
            else dict(self.settings or {})
        )
        self._host = self._resolve_setting(self._settings.get("host", "localhost"))
        self._port = int(self._resolve_setting(self._settings.get("port", 5672)))
        self._virtual_host = self._resolve_setting(self._settings.get("virtual_host", "/"))
        self._username = self._resolve_setting(self._settings.get("username", "guest"))
        self._password = self._resolve_setting(self._settings.get("password", "guest"))
        self._exchanges = list(self._settings.get("exchanges", [{"name": "intersect-messages"}]))
        self._exchange = self._exchanges[0] if self._exchanges else {"name": "intersect-messages"}
        self._exchange_name = self._exchange.get("name", "intersect-messages")
        self._exchange_type = self._exchange.get("type", "topic")
        self._exchange_durable = bool(self._exchange.get("durable", True))
        self._exchange_passive = bool(self._exchange.get("passive", True))
        self._routing_keys = list(self._exchange.get("routing_keys", ["#"]))
        self._observer_queue = dict(self._settings.get("observer_queue", {}))
        self._prefetch_count = int(
            self._settings.get("prefetch_count", self._observer_queue.get("prefetch_count", 100))
        )
        self._task_subtype = self._settings.get("task_subtype", "intersect_amqp_msg")
        payload_policy = self._settings.get("payload_policy", {}) or {}
        self._max_payload_bytes = int(payload_policy.get("max_payload_bytes", 65536))
        self._parse_json_preview = bool(payload_policy.get("parse_json_preview", True))
        self._retry_attempts = int(self._settings.get("connection_retry_attempts", 12))
        self._retry_initial_delay = float(self._settings.get("connection_retry_initial_delay_secs", 1))
        self._retry_max_delay = float(self._settings.get("connection_retry_max_delay_secs", 30))
        self._consumer_tag = None
        self._connection = None
        self._channel = None
        self._queue_name = None
        self._observer_thread: threading.Thread | None = None
        self._stopping = threading.Event()

    @staticmethod
    def _resolve_setting(value):
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                return os.getenv(value[2:-1], value)
            if value.startswith("$") and len(value) > 1:
                return os.getenv(value[1:], value)
        return value

    def _connect(self):
        if pika is None:
            raise ModuleNotFoundError("pika is required for AMQP observation. Install flowcept[amqp].")
        credentials = pika.PlainCredentials(self._username, self._password)
        params = pika.ConnectionParameters(
            host=self._host,
            port=self._port,
            virtual_host=self._virtual_host,
            credentials=credentials,
        )
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()

    def _setup_observer_queue(self):
        self._channel.exchange_declare(
            exchange=self._exchange_name,
            exchange_type=self._exchange_type,
            durable=self._exchange_durable,
            passive=self._exchange_passive,
        )
        queue_result = self._channel.queue_declare(
            queue=self._observer_queue.get("name", ""),
            durable=bool(self._observer_queue.get("durable", False)),
            exclusive=bool(self._observer_queue.get("exclusive", True)),
            auto_delete=bool(self._observer_queue.get("auto_delete", True)),
        )
        self._queue_name = queue_result.method.queue
        for routing_key in self._routing_keys:
            self._channel.queue_bind(
                exchange=self._exchange_name,
                queue=self._queue_name,
                routing_key=routing_key,
            )
        self._channel.basic_qos(prefetch_count=self._prefetch_count)

    def _connect_with_retry(self):
        delay = self._retry_initial_delay
        last_error = None
        for attempt in range(1, self._retry_attempts + 1):
            if self._stopping.is_set():
                return
            try:
                self._connect()
                self._setup_observer_queue()
                self.logger.info(
                    f"AMQP observer bound to exchange '{self._exchange_name}' queue '{self._queue_name}'."
                )
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    f"AMQP observer setup failed on attempt {attempt}/{self._retry_attempts}: {exc}"
                )
                self._close_amqp()
                if attempt < self._retry_attempts:
                    sleep(delay)
                    delay = min(delay * 2, self._retry_max_delay)
        raise RuntimeError("AMQP observer could not connect after configured retries.") from last_error

    def start(self, bundle_exec_id, check_safe_stops: bool = True) -> "AMQPBrokerInterceptor":
        """Start the observer thread."""
        super().start(bundle_exec_id, check_safe_stops=check_safe_stops)
        self._stopping.clear()
        self._observer_thread = threading.Thread(target=self.observe, daemon=True)
        self._observer_thread.start()
        return self

    def observe(self):
        """Connect and consume from the private observer queue."""
        try:
            self._connect_with_retry()
            if self._stopping.is_set() or self._channel is None:
                return
            self._consumer_tag = self._channel.basic_consume(
                queue=self._queue_name,
                on_message_callback=self.callback,
                auto_ack=False,
            )
            self._channel.start_consuming()
        except Exception as exc:
            if not self._stopping.is_set():
                self.logger.exception(exc)
                raise
        finally:
            self._close_amqp()

    def callback(self, channel, method, properties, body):
        """Convert one AMQP delivery and ack only after buffering succeeds."""
        delivery_tag = method.delivery_tag
        try:
            task_obj = self.prepare_task_msg(method=method, properties=properties, body=body)
            self.intercept(task_obj.to_dict())
        except Exception as exc:
            self.logger.error(f"Failed to convert AMQP message; rejecting delivery {delivery_tag}: {exc}")
            self._reject_without_requeue(channel, delivery_tag)
            return
        channel.basic_ack(delivery_tag=delivery_tag)

    @staticmethod
    def _reject_without_requeue(channel, delivery_tag):
        if hasattr(channel, "basic_nack"):
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
        else:
            channel.basic_reject(delivery_tag=delivery_tag, requeue=False)

    def prepare_task_msg(self, method=None, properties=None, body: bytes = b"") -> TaskObject:
        """Convert an AMQP delivery into a Flowcept task."""
        exchange = getattr(method, "exchange", None) or self._exchange_name
        routing_key = getattr(method, "routing_key", None) or ""
        headers = dict(getattr(properties, "headers", None) or {})
        content_type = getattr(properties, "content_type", None)
        payload_hash = sha256(body).hexdigest()
        task_id = (
            getattr(properties, "message_id", None)
            or headers.get("messageId")
            or self._fallback_task_id(exchange, routing_key, body, properties)
        )
        used = {
            "payload_sha256": payload_hash,
            "payload_size_bytes": len(body),
            "content_type": content_type,
        }
        preview = self._payload_preview(body, content_type)
        if preview is not None:
            used["payload_preview"] = preview

        custom_metadata = {
            "exchange": exchange,
            "routing_key": routing_key,
            "correlation_id": getattr(properties, "correlation_id", None),
            "reply_to": getattr(properties, "reply_to", None),
            "content_type": content_type,
            "headers": sanitize_value(headers),
            "delivery_mode": getattr(properties, "delivery_mode", None),
            "priority": getattr(properties, "priority", None),
            "timestamp": getattr(properties, "timestamp", None),
            "redelivered": getattr(method, "redelivered", None),
        }
        custom_metadata.update(self._parse_routing_key(routing_key))

        return TaskObject.from_dict(
            {
                "task_id": task_id,
                "activity_id": routing_key,
                "workflow_id": Flowcept.current_workflow_id,
                "campaign_id": Flowcept.campaign_id,
                "subtype": self._task_subtype,
                "used": sanitize_value(used),
                "custom_metadata": sanitize_value(custom_metadata),
            }
        )

    @staticmethod
    def _fallback_task_id(exchange: str, routing_key: str, body: bytes, properties) -> str:
        discriminator = "|".join(
            str(value or "")
            for value in (
                exchange,
                routing_key,
                getattr(properties, "correlation_id", None),
                getattr(properties, "timestamp", None),
            )
        ).encode()
        return sha256(discriminator + b"|" + body).hexdigest()

    def _payload_preview(self, body: bytes, content_type: str | None):
        if len(body) > self._max_payload_bytes:
            return None
        is_json_content = bool(content_type and "json" in content_type.lower())
        if self._parse_json_preview or is_json_content:
            try:
                return sanitize_value(json.loads(body.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if is_json_content:
                    return None
        try:
            return sanitize_value(body.decode("utf-8"))
        except UnicodeDecodeError:
            return body[:128].hex()

    def _parse_routing_key(self, routing_key: str) -> Dict[str, Any]:
        parts = [part for part in routing_key.split(".") if part] if routing_key else []
        metadata: Dict[str, Any] = {"routing_parts": parts}
        for index, field in enumerate(self._ROUTING_FIELDS):
            if index < len(parts):
                metadata[field] = parts[index]
        return metadata

    def _close_amqp(self):
        try:
            if self._channel is not None and getattr(self._channel, "is_open", True):
                self._channel.close()
        except Exception as exc:
            self.logger.warning(f"Exception while closing AMQP channel: {exc}")
        try:
            if self._connection is not None and getattr(self._connection, "is_open", True):
                self._connection.close()
        except Exception as exc:
            self.logger.warning(f"Exception while closing AMQP connection: {exc}")
        self._channel = None
        self._connection = None

    def stop(self, check_safe_stops: bool = True) -> bool:
        """Stop AMQP consumption and close Flowcept buffering."""
        self.logger.debug("AMQP interceptor stopping...")
        self._stopping.set()
        try:
            if self._channel is not None and getattr(self._channel, "is_open", True):
                if self._consumer_tag:
                    self._channel.basic_cancel(self._consumer_tag)
                self._channel.stop_consuming()
        except Exception as exc:
            self.logger.warning(f"Exception while cancelling AMQP consumer: {exc}")
        self._close_amqp()
        if self._observer_thread and self._observer_thread.is_alive():
            self._observer_thread.join(timeout=10)
        super().stop(check_safe_stops=check_safe_stops)
        self.logger.debug("AMQP interceptor stopped.")
        return True
