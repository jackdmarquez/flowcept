"""AMQP broker interceptor for observing INTERSECT traffic."""

from __future__ import annotations

import json
import os
import threading
from hashlib import sha256
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
        self._log_observed_messages = bool(self._settings.get("log_observed_messages", False))
        self._retry_attempts = int(self._settings.get("connection_retry_attempts", 12))
        self._retry_initial_delay = float(self._settings.get("connection_retry_initial_delay_secs", 1))
        self._retry_max_delay = float(self._settings.get("connection_retry_max_delay_secs", 30))
        self._process_data_events_time_limit = float(self._settings.get("process_data_events_time_limit_secs", 1.0))
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
            arguments=self._observer_queue.get("arguments", None),
        )
        self._queue_name = queue_result.method.queue
        for routing_key in self._routing_keys:
            self._channel.queue_bind(
                exchange=self._exchange_name,
                queue=self._queue_name,
                routing_key=routing_key,
            )
        self._channel.basic_qos(prefetch_count=self._prefetch_count)

    def start(self, bundle_exec_id, check_safe_stops: bool = True) -> "AMQPBrokerInterceptor":
        """Start the observer thread."""
        super().start(bundle_exec_id, check_safe_stops=check_safe_stops)
        self._stopping.clear()
        self._observer_thread = threading.Thread(target=self.observe, daemon=True)
        self._observer_thread.start()
        return self

    def observe(self):
        """Connect, consume, and reconnect from the observer thread."""
        delay = self._retry_initial_delay
        consecutive_failures = 0
        while not self._stopping.is_set():
            try:
                self._connect()
                self._setup_observer_queue()
                self._consumer_tag = self._channel.basic_consume(
                    queue=self._queue_name,
                    on_message_callback=self.callback,
                    auto_ack=False,
                )
                self.logger.info(f"AMQP observer bound to exchange '{self._exchange_name}' queue '{self._queue_name}'.")
                delay = self._retry_initial_delay
                consecutive_failures = 0
                self._process_data_events_until_stopped()
            except ModuleNotFoundError:
                raise
            except self._recoverable_amqp_exceptions() as exc:
                consecutive_failures += 1
                if self._stopping.is_set():
                    break
                self.logger.warning(
                    "AMQP observer connection/setup failed; "
                    f"will retry in {delay:.1f}s after sanitized error: {sanitize_value(str(exc))}"
                )
            except Exception as exc:
                consecutive_failures += 1
                if self._stopping.is_set():
                    break
                self.logger.error(
                    f"Unexpected AMQP observer error; retrying in {delay:.1f}s: {sanitize_value(str(exc))}"
                )
            finally:
                self._close_amqp()

            if self._stopping.is_set():
                break
            if self._retry_attempts > 0 and consecutive_failures >= self._retry_attempts:
                self.logger.warning(
                    f"AMQP observer has failed {consecutive_failures} consecutive times; continuing retries."
                )
            self._sleep_before_reconnect(delay)
            delay = min(delay * 2, self._retry_max_delay)

    def _process_data_events_until_stopped(self):
        while not self._stopping.is_set():
            self._connection.process_data_events(time_limit=self._process_data_events_time_limit)

    @staticmethod
    def _recoverable_amqp_exceptions():
        if pika is None:
            return ()
        names = (
            "AMQPConnectionError",
            "StreamLostError",
            "ChannelClosedByBroker",
            "ConnectionClosedByBroker",
            "AMQPChannelError",
        )
        return tuple(getattr(pika.exceptions, name) for name in names if hasattr(pika.exceptions, name))

    def _sleep_before_reconnect(self, delay):
        if self._stopping.wait(delay):
            return

    def callback(self, channel, method, properties, body):
        """Convert one AMQP delivery and ack only after buffering succeeds."""
        delivery_tag = method.delivery_tag
        try:
            task_obj = self.prepare_task_msg(method=method, properties=properties, body=body)
            self.intercept(task_obj.to_dict())
            self._log_observed_message(task_obj)
        except Exception as exc:
            self.logger.error(
                f"Failed to convert AMQP message; rejecting delivery {delivery_tag}: {sanitize_value(str(exc))}"
            )
            self._reject_without_requeue(channel, delivery_tag)
            return
        channel.basic_ack(delivery_tag=delivery_tag)

    def _log_observed_message(self, task_obj: TaskObject):
        if not self._log_observed_messages:
            return
        custom_metadata = task_obj.custom_metadata or {}
        used = task_obj.used or {}
        self.logger.info(
            "Observed AMQP message "
            f"task_id={task_obj.task_id} "
            f"activity_id={task_obj.activity_id} "
            f"campaign_id={task_obj.campaign_id} "
            f"exchange={custom_metadata.get('exchange')} "
            f"routing_key={custom_metadata.get('routing_key')} "
            f"payload_size_bytes={used.get('payload_size_bytes')} "
            f"content_type={used.get('content_type')}"
        )

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
            "headers": headers,
            "delivery_mode": getattr(properties, "delivery_mode", None),
            "priority": getattr(properties, "priority", None),
            "timestamp": getattr(properties, "timestamp", None),
            "redelivered": getattr(method, "redelivered", None),
        }
        custom_metadata.update(self._parse_routing_key(routing_key))
        intersect_metadata = self._extract_intersect_metadata(headers, preview)
        custom_metadata.update(intersect_metadata)

        task_id = (
            getattr(properties, "message_id", None)
            or intersect_metadata.get("intersect_message_id")
            or self._fallback_task_id(exchange, routing_key, body, properties)
        )
        activity_id = intersect_metadata.get("intersect_operation_id") or routing_key
        campaign_id = intersect_metadata.get("intersect_campaign_id") or Flowcept.campaign_id

        return TaskObject.from_dict(
            {
                "task_id": task_id,
                "activity_id": activity_id,
                "workflow_id": Flowcept.current_workflow_id,
                "campaign_id": campaign_id,
                "subtype": self._task_subtype,
                "used": sanitize_value(used),
                "custom_metadata": sanitize_value(custom_metadata),
            }
        )

    @staticmethod
    def _string_value(value):
        if isinstance(value, (str, int, float)):
            return str(value)
        return None

    @classmethod
    def _extract_first_field(cls, *sources_and_keys):
        for source, keys in sources_and_keys:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = cls._string_value(source.get(key))
                if value is not None:
                    return value
        return None

    @classmethod
    def _extract_intersect_metadata(cls, headers, payload_preview):
        metadata = {
            "intersect_message_id": cls._extract_first_field(
                (headers, ("message_id", "messageId")),
                (payload_preview, ("message_id", "messageId")),
            ),
            "intersect_operation_id": cls._extract_first_field(
                (headers, ("operation_id", "operationId")),
                (payload_preview, ("operation_id", "operationId")),
            ),
            "intersect_campaign_id": cls._extract_first_field(
                (headers, ("campaign_id", "campaignId")),
            ),
            "intersect_request_id": cls._extract_first_field(
                (headers, ("request_id", "requestId")),
                (payload_preview, ("request_id", "requestId")),
            ),
            "intersect_source": cls._extract_first_field(
                (headers, ("source",)),
                (payload_preview, ("source",)),
            ),
            "intersect_destination": cls._extract_first_field(
                (headers, ("destination",)),
                (payload_preview, ("destination",)),
            ),
            "intersect_sdk_version": cls._extract_first_field(
                (headers, ("sdk_version", "sdkVersion")),
                (payload_preview, ("sdk_version", "sdkVersion")),
            ),
            "intersect_created_at": cls._extract_first_field(
                (headers, ("created_at", "createdAt")),
                (payload_preview, ("created_at", "createdAt")),
            ),
            "intersect_lifecycle_type": cls._extract_first_field(
                (headers, ("lifecycle_type", "lifecycleType")),
                (payload_preview, ("lifecycle_type", "lifecycleType", "type")),
            ),
        }
        return {key: value for key, value in metadata.items() if value is not None}

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
                if self._consumer_tag:
                    try:
                        self._channel.basic_cancel(self._consumer_tag)
                    except Exception as exc:
                        self.logger.warning(
                            f"Exception while cancelling AMQP consumer in observer thread: {sanitize_value(str(exc))}"
                        )
                self._channel.close()
        except Exception as exc:
            self.logger.warning(f"Exception while closing AMQP channel: {sanitize_value(str(exc))}")
        try:
            if self._connection is not None and getattr(self._connection, "is_open", True):
                self._connection.close()
        except Exception as exc:
            self.logger.warning(f"Exception while closing AMQP connection: {sanitize_value(str(exc))}")
        self._channel = None
        self._connection = None
        self._consumer_tag = None

    def stop(self, check_safe_stops: bool = True) -> bool:
        """Stop AMQP consumption and close Flowcept buffering."""
        self.logger.debug("AMQP interceptor stopping...")
        self._stopping.set()
        if self._observer_thread and self._observer_thread.is_alive():
            self._observer_thread.join(timeout=10)
        if self._observer_thread and self._observer_thread.is_alive():
            self.logger.warning("AMQP observer thread did not stop within 10 seconds; leaving AMQP close to thread.")
        super().stop(check_safe_stops=check_safe_stops)
        self.logger.debug("AMQP interceptor stopped.")
        return True
