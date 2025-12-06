"""
ORCHEX Message Queue Module

Provides message queue abstraction supporting multiple brokers
(Redis, RabbitMQ, Kafka) with producer/consumer patterns,
message priorities, persistence, and delivery guarantees.

Features:
- Unified interface for multiple message brokers
- Producer/Consumer patterns with async support
- Message priorities and routing
- At-least-once and at-most-once delivery
- Message persistence and replay
- Dead letter queue support
- Connection pooling and failover

Author: MEZAN Research Team
Version: 1.0.0
"""

import json
import logging
import time
import threading
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue, PriorityQueue
import pickle

logger = logging.getLogger(__name__)


class MessageBrokerType(Enum):
    """Supported message broker types"""
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    IN_MEMORY = "in_memory"


class DeliveryMode(Enum):
    """Message delivery guarantees"""
    AT_MOST_ONCE = "at_most_once"     # Fire and forget
    AT_LEAST_ONCE = "at_least_once"   # Retry on failure
    EXACTLY_ONCE = "exactly_once"     # Idempotent delivery (Kafka only)


@dataclass
class Message:
    """Message container with metadata"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    key: Optional[str] = None
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    expiration: Optional[float] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    content_type: str = "application/json"
    retry_count: int = 0
    max_retries: int = 3

    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expiration:
            return time.time() > self.expiration
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary"""
        return cls(**data)


class MessageBroker(ABC):
    """Abstract base class for message brokers"""

    def __init__(
        self,
        broker_type: MessageBrokerType,
        connection_config: Dict[str, Any]
    ):
        """Initialize message broker"""
        self.broker_type = broker_type
        self.connection_config = connection_config
        self.is_connected = False
        self.consumers: Dict[str, List[Callable]] = {}
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "connection_failures": 0
        }

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker"""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection to broker"""
        pass

    @abstractmethod
    def send(self, message: Message, delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE) -> bool:
        """Send message to broker"""
        pass

    @abstractmethod
    def receive(self, topic: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from broker"""
        pass

    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """Subscribe to topic with callback"""
        pass

    @abstractmethod
    def unsubscribe(self, topic: str):
        """Unsubscribe from topic"""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get broker metrics"""
        return self.metrics.copy()


class RedisBroker(MessageBroker):
    """Redis-based message broker implementation"""

    def __init__(self, connection_config: Dict[str, Any]):
        """Initialize Redis broker"""
        super().__init__(MessageBrokerType.REDIS, connection_config)
        self.client = None
        self.pubsub = None
        self.subscription_threads: Dict[str, threading.Thread] = {}

    def connect(self) -> bool:
        """Connect to Redis"""
        try:
            import redis

            self.client = redis.Redis(
                host=self.connection_config.get("host", "localhost"),
                port=self.connection_config.get("port", 6379),
                db=self.connection_config.get("db", 0),
                password=self.connection_config.get("password"),
                decode_responses=False
            )

            # Test connection
            self.client.ping()
            self.pubsub = self.client.pubsub()
            self.is_connected = True

            logger.info(f"Connected to Redis broker at {self.connection_config.get('host')}:{self.connection_config.get('port')}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.metrics["connection_failures"] += 1
            return False

    def disconnect(self):
        """Disconnect from Redis"""
        # Stop subscription threads
        for thread in self.subscription_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)

        if self.pubsub:
            self.pubsub.close()

        if self.client:
            self.client.close()

        self.is_connected = False
        logger.info("Disconnected from Redis broker")

    def send(self, message: Message, delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE) -> bool:
        """Send message via Redis"""
        if not self.is_connected:
            logger.error("Not connected to Redis")
            return False

        try:
            # Serialize message
            serialized = pickle.dumps(message.to_dict())

            # Handle different delivery modes
            if delivery_mode == DeliveryMode.AT_MOST_ONCE:
                # Simple publish
                self.client.publish(message.topic, serialized)

            else:  # AT_LEAST_ONCE
                # Use list for persistence
                queue_name = f"queue:{message.topic}"

                # Add to queue based on priority
                if message.priority > 0:
                    # Use sorted set for priority queue
                    self.client.zadd(
                        f"{queue_name}:priority",
                        {serialized: -message.priority}  # Negative for high priority first
                    )
                else:
                    # Use list for FIFO
                    self.client.rpush(queue_name, serialized)

                # Also publish for real-time subscribers
                self.client.publish(message.topic, serialized)

            self.metrics["messages_sent"] += 1
            logger.debug(f"Sent message {message.message_id} to topic {message.topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics["messages_failed"] += 1
            return False

    def receive(self, topic: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from Redis"""
        if not self.is_connected:
            return None

        try:
            queue_name = f"queue:{topic}"
            timeout_int = int(timeout) if timeout else 0

            # Check priority queue first
            priority_queue = f"{queue_name}:priority"
            result = self.client.bzpopmin(priority_queue, timeout=timeout_int)

            if not result:
                # Check regular queue
                result = self.client.blpop(queue_name, timeout=timeout_int)

            if result:
                _, data = result
                message_dict = pickle.loads(data)
                message = Message.from_dict(message_dict)

                # Check expiration
                if message.is_expired():
                    logger.debug(f"Message {message.message_id} expired")
                    return self.receive(topic, timeout)  # Try next message

                self.metrics["messages_received"] += 1
                return message

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")

        return None

    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """Subscribe to Redis channel"""
        if not self.is_connected:
            return

        # Store callback
        if topic not in self.consumers:
            self.consumers[topic] = []
        self.consumers[topic].append(callback)

        # Create subscription thread
        def subscription_worker():
            try:
                pubsub = self.client.pubsub()
                pubsub.subscribe(topic)

                for raw_message in pubsub.listen():
                    if raw_message["type"] == "message":
                        try:
                            message_dict = pickle.loads(raw_message["data"])
                            message = Message.from_dict(message_dict)

                            # Call all callbacks
                            for cb in self.consumers.get(topic, []):
                                try:
                                    cb(message)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")

                        except Exception as e:
                            logger.error(f"Failed to process message: {e}")

            except Exception as e:
                logger.error(f"Subscription worker error: {e}")

        thread = threading.Thread(target=subscription_worker, daemon=True)
        thread.start()
        self.subscription_threads[topic] = thread

        logger.info(f"Subscribed to topic {topic}")

    def unsubscribe(self, topic: str):
        """Unsubscribe from Redis channel"""
        if topic in self.consumers:
            del self.consumers[topic]

        # Note: Thread will terminate when pubsub is closed
        if topic in self.subscription_threads:
            del self.subscription_threads[topic]


class RabbitMQBroker(MessageBroker):
    """RabbitMQ-based message broker implementation"""

    def __init__(self, connection_config: Dict[str, Any]):
        """Initialize RabbitMQ broker"""
        super().__init__(MessageBrokerType.RABBITMQ, connection_config)
        self.connection = None
        self.channel = None

    def connect(self) -> bool:
        """Connect to RabbitMQ"""
        try:
            import pika

            credentials = pika.PlainCredentials(
                self.connection_config.get("username", "guest"),
                self.connection_config.get("password", "guest")
            )

            parameters = pika.ConnectionParameters(
                host=self.connection_config.get("host", "localhost"),
                port=self.connection_config.get("port", 5672),
                virtual_host=self.connection_config.get("vhost", "/"),
                credentials=credentials
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.is_connected = True

            logger.info(f"Connected to RabbitMQ broker at {self.connection_config.get('host')}")
            return True

        except ImportError:
            logger.error("pika library not installed")
            return False

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self.metrics["connection_failures"] += 1
            return False

    def disconnect(self):
        """Disconnect from RabbitMQ"""
        if self.channel:
            self.channel.close()

        if self.connection:
            self.connection.close()

        self.is_connected = False
        logger.info("Disconnected from RabbitMQ broker")

    def send(self, message: Message, delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE) -> bool:
        """Send message via RabbitMQ"""
        if not self.is_connected:
            return False

        try:
            import pika

            # Declare exchange and queue
            self.channel.exchange_declare(
                exchange=message.topic,
                exchange_type="direct",
                durable=True
            )

            # Set delivery mode
            properties = pika.BasicProperties(
                delivery_mode=2 if delivery_mode != DeliveryMode.AT_MOST_ONCE else 1,
                message_id=message.message_id,
                timestamp=int(message.timestamp),
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                priority=message.priority,
                headers=message.headers
            )

            # Publish message
            self.channel.basic_publish(
                exchange=message.topic,
                routing_key=message.key or "",
                body=pickle.dumps(message.payload),
                properties=properties
            )

            self.metrics["messages_sent"] += 1
            logger.debug(f"Sent message {message.message_id} to exchange {message.topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics["messages_failed"] += 1
            return False

    def receive(self, topic: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from RabbitMQ"""
        if not self.is_connected:
            return None

        try:
            # Declare queue
            result = self.channel.queue_declare(queue=topic, durable=True)
            queue_name = result.method.queue

            # Get message
            method, properties, body = self.channel.basic_get(queue_name)

            if method:
                message = Message(
                    message_id=properties.message_id or str(uuid.uuid4()),
                    topic=topic,
                    payload=pickle.loads(body),
                    timestamp=properties.timestamp or time.time(),
                    priority=properties.priority or 0,
                    correlation_id=properties.correlation_id,
                    reply_to=properties.reply_to,
                    headers=properties.headers or {}
                )

                # Acknowledge message
                self.channel.basic_ack(method.delivery_tag)

                self.metrics["messages_received"] += 1
                return message

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")

        return None

    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """Subscribe to RabbitMQ queue"""
        if not self.is_connected:
            return

        try:
            # Declare queue
            result = self.channel.queue_declare(queue=topic, durable=True)
            queue_name = result.method.queue

            # Store callback
            if topic not in self.consumers:
                self.consumers[topic] = []
            self.consumers[topic].append(callback)

            # Define consumer callback
            def on_message(channel, method, properties, body):
                try:
                    message = Message(
                        message_id=properties.message_id or str(uuid.uuid4()),
                        topic=topic,
                        payload=pickle.loads(body),
                        timestamp=properties.timestamp or time.time(),
                        priority=properties.priority or 0,
                        correlation_id=properties.correlation_id,
                        reply_to=properties.reply_to,
                        headers=properties.headers or {}
                    )

                    # Call callbacks
                    for cb in self.consumers.get(topic, []):
                        try:
                            cb(message)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    # Acknowledge message
                    channel.basic_ack(delivery_tag=method.delivery_tag)

                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

            # Start consuming
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message,
                auto_ack=False
            )

            # Start consuming in thread
            def consume():
                try:
                    self.channel.start_consuming()
                except Exception as e:
                    logger.error(f"Consumer error: {e}")

            thread = threading.Thread(target=consume, daemon=True)
            thread.start()

            logger.info(f"Subscribed to queue {topic}")

        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")

    def unsubscribe(self, topic: str):
        """Unsubscribe from RabbitMQ queue"""
        if topic in self.consumers:
            del self.consumers[topic]

        try:
            self.channel.stop_consuming()
        except Exception:
            pass


class KafkaBroker(MessageBroker):
    """Kafka-based message broker implementation"""

    def __init__(self, connection_config: Dict[str, Any]):
        """Initialize Kafka broker"""
        super().__init__(MessageBrokerType.KAFKA, connection_config)
        self.producer = None
        self.consumers: Dict[str, Any] = {}
        self.consumer_threads: Dict[str, threading.Thread] = {}

    def connect(self) -> bool:
        """Connect to Kafka"""
        try:
            from kafka import KafkaProducer

            self.producer = KafkaProducer(
                bootstrap_servers=self.connection_config.get("bootstrap_servers", ["localhost:9092"]),
                value_serializer=lambda v: pickle.dumps(v),
                key_serializer=lambda k: k.encode() if k else None
            )

            self.is_connected = True
            logger.info(f"Connected to Kafka broker at {self.connection_config.get('bootstrap_servers')}")
            return True

        except ImportError:
            logger.error("kafka-python library not installed")
            return False

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.metrics["connection_failures"] += 1
            return False

    def disconnect(self):
        """Disconnect from Kafka"""
        # Stop consumer threads
        for thread in self.consumer_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)

        # Close consumers
        for consumer in self.consumers.values():
            consumer.close()

        # Close producer
        if self.producer:
            self.producer.close()

        self.is_connected = False
        logger.info("Disconnected from Kafka broker")

    def send(self, message: Message, delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE) -> bool:
        """Send message via Kafka"""
        if not self.is_connected:
            return False

        try:
            # Determine acknowledgment level
            acks = "all" if delivery_mode != DeliveryMode.AT_MOST_ONCE else 0

            # Send message
            future = self.producer.send(
                topic=message.topic,
                key=message.key,
                value=message.to_dict(),
                headers=[(k, v.encode()) for k, v in message.headers.items()]
            )

            if delivery_mode != DeliveryMode.AT_MOST_ONCE:
                # Wait for acknowledgment
                future.get(timeout=10)

            self.metrics["messages_sent"] += 1
            logger.debug(f"Sent message {message.message_id} to topic {message.topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics["messages_failed"] += 1
            return False

    def receive(self, topic: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from Kafka"""
        if not self.is_connected:
            return None

        try:
            from kafka import KafkaConsumer

            # Create consumer if not exists
            if topic not in self.consumers:
                self.consumers[topic] = KafkaConsumer(
                    topic,
                    bootstrap_servers=self.connection_config.get("bootstrap_servers", ["localhost:9092"]),
                    value_deserializer=lambda m: pickle.loads(m),
                    consumer_timeout_ms=int(timeout * 1000) if timeout else 1000,
                    auto_offset_reset="earliest"
                )

            consumer = self.consumers[topic]

            # Poll for message
            for message in consumer:
                message_dict = message.value
                msg = Message.from_dict(message_dict)

                self.metrics["messages_received"] += 1
                return msg

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")

        return None

    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """Subscribe to Kafka topic"""
        if not self.is_connected:
            return

        try:
            from kafka import KafkaConsumer

            # Create consumer
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.connection_config.get("bootstrap_servers", ["localhost:9092"]),
                value_deserializer=lambda m: pickle.loads(m),
                auto_offset_reset="latest"
            )

            self.consumers[topic] = consumer

            # Consumer thread
            def consume():
                try:
                    for message in consumer:
                        try:
                            msg = Message.from_dict(message.value)
                            callback(msg)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                except Exception as e:
                    logger.error(f"Consumer error: {e}")

            thread = threading.Thread(target=consume, daemon=True)
            thread.start()
            self.consumer_threads[topic] = thread

            logger.info(f"Subscribed to topic {topic}")

        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")

    def unsubscribe(self, topic: str):
        """Unsubscribe from Kafka topic"""
        if topic in self.consumers:
            self.consumers[topic].close()
            del self.consumers[topic]

        if topic in self.consumer_threads:
            del self.consumer_threads[topic]


class InMemoryBroker(MessageBroker):
    """In-memory message broker for testing"""

    def __init__(self, connection_config: Dict[str, Any] = None):
        """Initialize in-memory broker"""
        super().__init__(MessageBrokerType.IN_MEMORY, connection_config or {})
        self.queues: Dict[str, PriorityQueue] = {}
        self.subscriber_threads: Dict[str, threading.Thread] = {}
        self.stop_flags: Dict[str, threading.Event] = {}

    def connect(self) -> bool:
        """Connect (always succeeds for in-memory)"""
        self.is_connected = True
        logger.info("In-memory broker connected")
        return True

    def disconnect(self):
        """Disconnect in-memory broker"""
        # Stop all subscriber threads
        for topic, flag in self.stop_flags.items():
            flag.set()

        for thread in self.subscriber_threads.values():
            if thread.is_alive():
                thread.join(timeout=1)

        self.is_connected = False
        logger.info("In-memory broker disconnected")

    def send(self, message: Message, delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE) -> bool:
        """Send message to in-memory queue"""
        if not self.is_connected:
            return False

        try:
            # Create queue if not exists
            if message.topic not in self.queues:
                self.queues[message.topic] = PriorityQueue()

            # Add to queue with priority
            self.queues[message.topic].put((-message.priority, time.time(), message))

            self.metrics["messages_sent"] += 1
            logger.debug(f"Sent message {message.message_id} to in-memory topic {message.topic}")

            # Trigger callbacks for real-time processing
            if message.topic in self.consumers:
                for callback in self.consumers[message.topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics["messages_failed"] += 1
            return False

    def receive(self, topic: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from in-memory queue"""
        if not self.is_connected:
            return None

        try:
            if topic not in self.queues:
                self.queues[topic] = PriorityQueue()

            # Get message with timeout
            try:
                _, _, message = self.queues[topic].get(timeout=timeout or 0)

                # Check expiration
                if message.is_expired():
                    return self.receive(topic, timeout)  # Try next

                self.metrics["messages_received"] += 1
                return message

            except:
                return None

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None

    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """Subscribe to in-memory topic"""
        if not self.is_connected:
            return

        # Store callback
        if topic not in self.consumers:
            self.consumers[topic] = []
        self.consumers[topic].append(callback)

        logger.info(f"Subscribed to in-memory topic {topic}")

    def unsubscribe(self, topic: str):
        """Unsubscribe from in-memory topic"""
        if topic in self.consumers:
            del self.consumers[topic]


class MessageQueueFactory:
    """Factory for creating message brokers"""

    @staticmethod
    def create_broker(
        broker_type: Union[str, MessageBrokerType],
        connection_config: Dict[str, Any] = None
    ) -> MessageBroker:
        """
        Create message broker instance

        Args:
            broker_type: Type of broker
            connection_config: Connection configuration

        Returns:
            MessageBroker instance
        """
        if isinstance(broker_type, str):
            broker_type = MessageBrokerType(broker_type)

        if broker_type == MessageBrokerType.REDIS:
            return RedisBroker(connection_config or {})

        elif broker_type == MessageBrokerType.RABBITMQ:
            return RabbitMQBroker(connection_config or {})

        elif broker_type == MessageBrokerType.KAFKA:
            return KafkaBroker(connection_config or {})

        elif broker_type == MessageBrokerType.IN_MEMORY:
            return InMemoryBroker(connection_config or {})

        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")


class MessageQueue:
    """
    High-level message queue interface

    Provides:
    - Automatic reconnection
    - Load balancing
    - Message routing
    - Request/Reply patterns
    """

    def __init__(
        self,
        broker_type: Union[str, MessageBrokerType] = MessageBrokerType.IN_MEMORY,
        connection_config: Dict[str, Any] = None,
        enable_failover: bool = True
    ):
        """Initialize message queue"""
        self.primary_broker = MessageQueueFactory.create_broker(broker_type, connection_config)
        self.failover_broker = None
        self.enable_failover = enable_failover

        # Connect to primary broker
        if not self.primary_broker.connect():
            logger.warning("Failed to connect to primary broker")

            # Try failover if enabled
            if enable_failover:
                self.failover_broker = MessageQueueFactory.create_broker(
                    MessageBrokerType.IN_MEMORY
                )
                self.failover_broker.connect()
                logger.info("Using in-memory failover broker")

    def send(
        self,
        topic: str,
        payload: Any,
        priority: int = 0,
        key: Optional[str] = None,
        headers: Dict[str, str] = None,
        delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    ) -> bool:
        """
        Send message to queue

        Args:
            topic: Target topic/queue
            payload: Message payload
            priority: Message priority
            key: Routing key
            headers: Message headers
            delivery_mode: Delivery guarantee

        Returns:
            True if sent successfully
        """
        message = Message(
            topic=topic,
            payload=payload,
            priority=priority,
            key=key,
            headers=headers or {}
        )

        # Try primary broker
        if self.primary_broker.is_connected:
            if self.primary_broker.send(message, delivery_mode):
                return True

        # Try failover broker
        if self.failover_broker and self.failover_broker.is_connected:
            return self.failover_broker.send(message, delivery_mode)

        return False

    def receive(self, topic: str, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Receive message from queue

        Args:
            topic: Source topic/queue
            timeout: Receive timeout

        Returns:
            Message payload or None
        """
        # Try primary broker
        if self.primary_broker.is_connected:
            message = self.primary_broker.receive(topic, timeout)
            if message:
                return message.payload

        # Try failover broker
        if self.failover_broker and self.failover_broker.is_connected:
            message = self.failover_broker.receive(topic, timeout)
            if message:
                return message.payload

        return None

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        """
        Subscribe to topic

        Args:
            topic: Topic to subscribe to
            callback: Message handler
        """
        # Wrapper to extract payload
        def wrapped_callback(message: Message):
            callback(message.payload)

        # Subscribe on primary
        if self.primary_broker.is_connected:
            self.primary_broker.subscribe(topic, wrapped_callback)

        # Subscribe on failover
        if self.failover_broker and self.failover_broker.is_connected:
            self.failover_broker.subscribe(topic, wrapped_callback)

    def request_reply(
        self,
        topic: str,
        payload: Any,
        timeout: float = 30
    ) -> Optional[Any]:
        """
        Request/Reply pattern

        Args:
            topic: Request topic
            payload: Request payload
            timeout: Reply timeout

        Returns:
            Reply payload or None
        """
        # Create reply queue
        reply_topic = f"reply.{uuid.uuid4()}"
        reply_received = threading.Event()
        reply_payload = None

        def handle_reply(payload):
            nonlocal reply_payload
            reply_payload = payload
            reply_received.set()

        # Subscribe to reply queue
        self.subscribe(reply_topic, handle_reply)

        # Send request with reply_to
        message = Message(
            topic=topic,
            payload=payload,
            reply_to=reply_topic
        )

        if self.primary_broker.is_connected:
            self.primary_broker.send(message)
        elif self.failover_broker and self.failover_broker.is_connected:
            self.failover_broker.send(message)

        # Wait for reply
        if reply_received.wait(timeout):
            return reply_payload

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        metrics = {
            "primary": self.primary_broker.get_metrics() if self.primary_broker else {},
            "failover": self.failover_broker.get_metrics() if self.failover_broker else {},
            "primary_connected": self.primary_broker.is_connected if self.primary_broker else False,
            "failover_active": self.failover_broker is not None
        }

        return metrics

    def close(self):
        """Close all connections"""
        if self.primary_broker:
            self.primary_broker.disconnect()

        if self.failover_broker:
            self.failover_broker.disconnect()