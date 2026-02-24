import json
import unittest
import uuid
from time import sleep

import paho.mqtt.client as mqtt

from flowcept import Flowcept
from flowcept.commons.flowcept_logger import FlowceptLogger
from flowcept.commons.utils import assert_by_querying_tasks_until, get_utc_now_str
from flowcept.configs import settings


class TestBroker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBroker, self).__init__(*args, **kwargs)
        self.logger = FlowceptLogger()

    def test_publish_msg(self):

        host = settings["adapters"]["broker_mqtt"]["host"]
        port = settings["adapters"]["broker_mqtt"]["port"]
        username = settings["adapters"]["broker_mqtt"]["username"]
        password = settings["adapters"]["broker_mqtt"]["password"]
        topic = TestBroker._get_publishable_topic(settings["adapters"]["broker_mqtt"]["queues"][0])
        qos = 2  # settings["adapters"]["broker_mqtt"]["qos"]

        for i in range(2):
            TestBroker.publish_msg(host, port, username, password, qos, topic)

    @staticmethod
    def _get_publishable_topic(topic: str) -> str:
        if "+" not in topic and "#" not in topic:
            return topic
        return topic.replace("+", "flowcept-test").replace("#", "flowcept-test")

    @staticmethod
    def publish_msg(host, port, username, password, qos, topic=None):
        msgId = str(uuid.uuid4())
        if topic is None:
            topic = "s3m-org/s3m-facility/s3m-system/s3m-subsystem/s3m-service/request"
        intersect_msg = {
            "msgId": msgId,
            "messageId": msgId,
            "operationId": "IntersectS3M.test_intersect_message",
            "contentType": "application/json",
            "payload": '"S3M Intersect test!"',
            "headers": {
                "source": "tmp-.tmp-.tmp-.-.tmp-66155c12-843c-4a69-bcec-3813acd3de9d",
                "destination": "s3m-org.s3m-facility.s3m-system.s3m-subsystem.s3m-service",
                "sdk_version": "0.8.2",
                "created_at": get_utc_now_str(),
                "data_handler": 0,
                "has_error": False,
            },
        }
        client = mqtt.Client(client_id="producer", clean_session=False, protocol=mqtt.MQTTv311)
        client.username_pw_set(username, password)

        client.connect(host, port, 60)
        client.loop_start()
        message = json.dumps(intersect_msg)
        msg_info = client.publish(topic, message, qos=qos, retain=True)
        msg_info.wait_for_publish()
        print(f" [x] Sent: {message} to topic '{topic}'")

        client.disconnect()
        client.loop_stop()

        return msgId

    def test_observation(self):
        with Flowcept("broker_mqtt") as f:
            sleep(1)

            host = f._interceptor_instances[0]._host
            port = f._interceptor_instances[0]._port
            username = f._interceptor_instances[0]._username
            password = f._interceptor_instances[0]._password
            qos = f._interceptor_instances[0]._qos

            msg_id = TestBroker.publish_msg(host, port, username, password, qos)

            sleep(5)

        assert assert_by_querying_tasks_until(
            {"custom_metadata.msgId": msg_id},
        )

    @classmethod
    def tearDownClass(cls):
        Flowcept.db.close()


if __name__ == "__main__":
    unittest.main()
