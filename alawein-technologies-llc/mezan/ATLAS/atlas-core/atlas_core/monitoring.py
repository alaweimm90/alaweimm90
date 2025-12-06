"""
MEZAN Real-time Monitoring and Dashboard System.

This module provides real-time monitoring capabilities including:
- WebSocket-based metric streaming
- Live dashboard data aggregation
- Health check endpoints
- Service discovery integration
- Distributed tracing with OpenTelemetry
- Performance metrics collection
- System status monitoring
"""

import asyncio
import json
import time
import threading
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
import aiohttp
from aiohttp import web
import psutil
import socket
import uuid

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricAggregation(Enum):
    """Metric aggregation types."""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


@dataclass
class ServiceInfo:
    """Service registration information."""
    name: str
    host: str
    port: int
    version: str
    status: ServiceStatus
    metadata: Dict[str, Any]
    last_heartbeat: float
    health_check_url: Optional[str] = None
    metrics_url: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Health check result."""
    service: str
    status: ServiceStatus
    latency_ms: float
    message: str
    checks: Dict[str, bool]
    timestamp: float


@dataclass
class TraceSpan:
    """Distributed trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    service: str
    start_time: float
    end_time: Optional[float]
    status: str
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]


class MetricsAggregator:
    """Real-time metrics aggregation engine."""

    def __init__(self, window_size: int = 60, bucket_size: int = 1):
        self.window_size = window_size
        self.bucket_size = bucket_size
        self._metrics = defaultdict(lambda: deque(maxlen=window_size))
        self._aggregates = {}
        self._lock = threading.RLock()

    def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add metric value."""
        with self._lock:
            key = self._generate_key(name, tags)
            timestamp = time.time()
            self._metrics[key].append((timestamp, value))

    def _generate_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate metric key with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_str}"

    def aggregate(self, name: str, aggregation: MetricAggregation,
                 tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Calculate aggregated metric value."""
        with self._lock:
            key = self._generate_key(name, tags)
            if key not in self._metrics or not self._metrics[key]:
                return None

            values = [v for _, v in self._metrics[key]]

            if aggregation == MetricAggregation.AVG:
                return sum(values) / len(values)
            elif aggregation == MetricAggregation.SUM:
                return sum(values)
            elif aggregation == MetricAggregation.MIN:
                return min(values)
            elif aggregation == MetricAggregation.MAX:
                return max(values)
            elif aggregation == MetricAggregation.COUNT:
                return len(values)
            elif aggregation == MetricAggregation.P50:
                return self._percentile(values, 50)
            elif aggregation == MetricAggregation.P95:
                return self._percentile(values, 95)
            elif aggregation == MetricAggregation.P99:
                return self._percentile(values, 99)

            return None

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100.0)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_time_series(self, name: str, tags: Optional[Dict[str, str]] = None,
                       duration: int = 60) -> List[Tuple[float, float]]:
        """Get time series data for metric."""
        with self._lock:
            key = self._generate_key(name, tags)
            if key not in self._metrics:
                return []

            cutoff_time = time.time() - duration
            return [(t, v) for t, v in self._metrics[key] if t >= cutoff_time]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            result = {}
            for key in self._metrics:
                if self._metrics[key]:
                    latest = self._metrics[key][-1]
                    result[key] = {
                        "value": latest[1],
                        "timestamp": latest[0],
                        "count": len(self._metrics[key])
                    }
            return result


class ServiceRegistry:
    """Service discovery and registration."""

    def __init__(self, ttl_seconds: int = 30):
        self.services: Dict[str, ServiceInfo] = {}
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def register(self, service: ServiceInfo):
        """Register a service."""
        with self._lock:
            service.last_heartbeat = time.time()
            self.services[service.name] = service
            logger.info(f"Registered service: {service.name} at {service.host}:{service.port}")

    def unregister(self, name: str):
        """Unregister a service."""
        with self._lock:
            if name in self.services:
                del self.services[name]
                logger.info(f"Unregistered service: {name}")

    def heartbeat(self, name: str):
        """Update service heartbeat."""
        with self._lock:
            if name in self.services:
                self.services[name].last_heartbeat = time.time()

    def get_service(self, name: str) -> Optional[ServiceInfo]:
        """Get service information."""
        with self._lock:
            return self.services.get(name)

    def get_all_services(self) -> List[ServiceInfo]:
        """Get all registered services."""
        with self._lock:
            return list(self.services.values())

    def get_healthy_services(self) -> List[ServiceInfo]:
        """Get healthy services."""
        with self._lock:
            return [s for s in self.services.values()
                   if s.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]]

    def _cleanup_loop(self):
        """Remove stale services."""
        while True:
            try:
                current_time = time.time()
                with self._lock:
                    stale_services = []
                    for name, service in self.services.items():
                        if current_time - service.last_heartbeat > self.ttl_seconds:
                            stale_services.append(name)

                    for name in stale_services:
                        logger.warning(f"Removing stale service: {name}")
                        del self.services[name]

            except Exception as e:
                logger.error(f"Service cleanup error: {e}")

            time.sleep(self.ttl_seconds / 2)


class HealthChecker:
    """Service health checking."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.RLock()

    async def check_service(self, service: ServiceInfo) -> HealthCheckResult:
        """Check service health."""
        start_time = time.time()
        checks = {}

        try:
            if service.health_check_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        service.health_check_url,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        checks["http_status"] = response.status == 200

                        if response.status == 200:
                            data = await response.json()
                            checks.update(data.get("checks", {}))
            else:
                # Basic connectivity check
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                result = sock.connect_ex((service.host, service.port))
                sock.close()
                checks["connectivity"] = result == 0

            # Determine overall status
            if all(checks.values()):
                status = ServiceStatus.HEALTHY
                message = "All checks passed"
            elif any(checks.values()):
                status = ServiceStatus.DEGRADED
                message = f"Some checks failed: {[k for k, v in checks.items() if not v]}"
            else:
                status = ServiceStatus.UNHEALTHY
                message = "All checks failed"

        except Exception as e:
            status = ServiceStatus.UNHEALTHY
            message = str(e)
            checks["error"] = str(e)

        latency_ms = (time.time() - start_time) * 1000

        result = HealthCheckResult(
            service=service.name,
            status=status,
            latency_ms=latency_ms,
            message=message,
            checks=checks,
            timestamp=time.time()
        )

        with self._lock:
            self.results[service.name] = result

        return result

    async def check_all(self, services: List[ServiceInfo]) -> List[HealthCheckResult]:
        """Check health of all services."""
        tasks = [self.check_service(service) for service in services]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_latest_results(self) -> Dict[str, HealthCheckResult]:
        """Get latest health check results."""
        with self._lock:
            return self.results.copy()


class DistributedTracer:
    """Distributed tracing manager."""

    def __init__(self, service_name: str, otlp_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self._lock = threading.RLock()

        if OTEL_AVAILABLE and otlp_endpoint:
            self._setup_opentelemetry(otlp_endpoint)
        else:
            self.tracer = None

    def _setup_opentelemetry(self, otlp_endpoint: str):
        """Setup OpenTelemetry tracing."""
        resource = Resource.create({"service.name": self.service_name})

        # Setup tracing
        trace_provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        trace_provider.add_span_processor(processor)
        trace.set_tracer_provider(trace_provider)

        self.tracer = trace.get_tracer(__name__)

        # Instrument libraries
        AioHttpClientInstrumentor().instrument()

        logger.info(f"OpenTelemetry tracing configured with endpoint: {otlp_endpoint}")

    def start_span(self, operation: str, parent_span_id: Optional[str] = None,
                  attributes: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """Start a new trace span."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            service=self.service_name,
            start_time=time.time(),
            end_time=None,
            status="in_progress",
            attributes=attributes or {},
            events=[]
        )

        with self._lock:
            self.traces[trace_id].append(span)

        return span

    def end_span(self, span: TraceSpan, status: str = "ok"):
        """End a trace span."""
        span.end_time = time.time()
        span.status = status

    def add_event(self, span: TraceSpan, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        span.events.append(event)

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return self.traces.get(trace_id, [])

    def get_recent_traces(self, limit: int = 100) -> List[str]:
        """Get recent trace IDs."""
        with self._lock:
            trace_ids = list(self.traces.keys())
            return trace_ids[-limit:]


class MonitoringServer:
    """Real-time monitoring HTTP/WebSocket server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8090,
                 metrics_aggregator: Optional[MetricsAggregator] = None,
                 service_registry: Optional[ServiceRegistry] = None,
                 health_checker: Optional[HealthChecker] = None,
                 tracer: Optional[DistributedTracer] = None):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.metrics_aggregator = metrics_aggregator or MetricsAggregator()
        self.service_registry = service_registry or ServiceRegistry()
        self.health_checker = health_checker or HealthChecker()
        self.tracer = tracer
        self.websockets: Set[web.WebSocketResponse] = set()

        self._setup_routes()
        self._start_metric_streamer()

    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get('/health', self.health_endpoint)
        self.app.router.add_get('/metrics', self.metrics_endpoint)
        self.app.router.add_get('/services', self.services_endpoint)
        self.app.router.add_get('/traces', self.traces_endpoint)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_post('/register', self.register_service)
        self.app.router.add_post('/heartbeat', self.heartbeat_endpoint)

    def _start_metric_streamer(self):
        """Start background metric streaming."""
        async def stream_metrics():
            while True:
                try:
                    # Get latest metrics
                    metrics = self.metrics_aggregator.get_all_metrics()

                    # Get service status
                    services = self.service_registry.get_all_services()
                    health_results = await self.health_checker.check_all(services)

                    data = {
                        "type": "metrics_update",
                        "timestamp": time.time(),
                        "metrics": metrics,
                        "services": [asdict(s) for s in services],
                        "health": [asdict(r) for r in health_results]
                    }

                    # Send to all connected websockets
                    disconnected = set()
                    for ws in self.websockets:
                        try:
                            await ws.send_json(data)
                        except ConnectionResetError:
                            disconnected.add(ws)

                    # Remove disconnected websockets
                    self.websockets -= disconnected

                except Exception as e:
                    logger.error(f"Metric streaming error: {e}")

                await asyncio.sleep(1)

        asyncio.create_task(stream_metrics())

    async def health_endpoint(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        # Check internal health
        checks = {
            "database": True,  # Would check actual database
            "cache": True,     # Would check cache connectivity
            "disk_space": psutil.disk_usage('/').percent < 90,
            "memory": psutil.virtual_memory().percent < 90,
            "cpu": psutil.cpu_percent(interval=1) < 90
        }

        status = "healthy" if all(checks.values()) else "degraded"

        return web.json_response({
            "status": status,
            "checks": checks,
            "timestamp": time.time()
        })

    async def metrics_endpoint(self, request: web.Request) -> web.Response:
        """Prometheus-compatible metrics endpoint."""
        metrics = []

        # System metrics
        metrics.append(f"mezan_cpu_usage_percent {psutil.cpu_percent()}")
        metrics.append(f"mezan_memory_usage_percent {psutil.virtual_memory().percent}")
        metrics.append(f"mezan_disk_usage_percent {psutil.disk_usage('/').percent}")

        # Custom metrics
        for key, data in self.metrics_aggregator.get_all_metrics().items():
            metrics.append(f'{key} {data["value"]}')

        return web.Response(text="\n".join(metrics), content_type='text/plain')

    async def services_endpoint(self, request: web.Request) -> web.Response:
        """Service discovery endpoint."""
        services = self.service_registry.get_all_services()
        return web.json_response([asdict(s) for s in services])

    async def traces_endpoint(self, request: web.Request) -> web.Response:
        """Distributed traces endpoint."""
        if not self.tracer:
            return web.json_response({"error": "Tracing not configured"}, status=503)

        trace_id = request.query.get('trace_id')

        if trace_id:
            traces = self.tracer.get_trace(trace_id)
            return web.json_response([asdict(t) for t in traces])
        else:
            recent = self.tracer.get_recent_traces(limit=100)
            return web.json_response({"recent_traces": recent})

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket connection handler."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)

        try:
            # Send initial data
            await ws.send_json({
                "type": "connection",
                "status": "connected",
                "timestamp": time.time()
            })

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Handle different message types
                    if data.get("type") == "subscribe":
                        metrics = data.get("metrics", [])
                        await ws.send_json({
                            "type": "subscribed",
                            "metrics": metrics
                        })

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.websockets.discard(ws)

        return ws

    async def register_service(self, request: web.Request) -> web.Response:
        """Register a new service."""
        data = await request.json()

        service = ServiceInfo(
            name=data['name'],
            host=data['host'],
            port=data['port'],
            version=data.get('version', '1.0.0'),
            status=ServiceStatus(data.get('status', 'healthy')),
            metadata=data.get('metadata', {}),
            last_heartbeat=time.time(),
            health_check_url=data.get('health_check_url'),
            metrics_url=data.get('metrics_url')
        )

        self.service_registry.register(service)

        return web.json_response({"status": "registered", "service": data['name']})

    async def heartbeat_endpoint(self, request: web.Request) -> web.Response:
        """Service heartbeat endpoint."""
        data = await request.json()
        service_name = data.get('service')

        if not service_name:
            return web.json_response({"error": "Service name required"}, status=400)

        self.service_registry.heartbeat(service_name)

        return web.json_response({"status": "ok", "timestamp": time.time()})

    def run(self):
        """Start monitoring server."""
        logger.info(f"Starting monitoring server on {self.host}:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port)


def create_monitoring_stack(service_name: str, otlp_endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Create complete monitoring stack."""

    # Create components
    metrics_aggregator = MetricsAggregator()
    service_registry = ServiceRegistry()
    health_checker = HealthChecker()
    tracer = DistributedTracer(service_name, otlp_endpoint) if otlp_endpoint else None

    # Create server
    server = MonitoringServer(
        metrics_aggregator=metrics_aggregator,
        service_registry=service_registry,
        health_checker=health_checker,
        tracer=tracer
    )

    return {
        "server": server,
        "metrics": metrics_aggregator,
        "registry": service_registry,
        "health": health_checker,
        "tracer": tracer
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create monitoring stack
    stack = create_monitoring_stack("mezan-monitoring")

    # Add some example metrics
    stack["metrics"].add_metric("request_count", 100, {"endpoint": "/api/v1/data"})
    stack["metrics"].add_metric("response_time", 0.125, {"endpoint": "/api/v1/data"})

    # Register example service
    example_service = ServiceInfo(
        name="ORCHEX-core",
        host="localhost",
        port=8080,
        version="1.0.0",
        status=ServiceStatus.HEALTHY,
        metadata={"region": "us-west-2"},
        last_heartbeat=time.time(),
        health_check_url="http://localhost:8080/health"
    )
    stack["registry"].register(example_service)

    # Start server
    stack["server"].run()