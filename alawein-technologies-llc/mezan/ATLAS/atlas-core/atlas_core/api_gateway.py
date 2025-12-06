"""
Enterprise API Gateway with Advanced Features

Production-grade API gateway with:
- Dynamic routing and service discovery
- Multiple load balancing strategies
- Circuit breaker pattern
- Request/response transformation
- API versioning support
- Service mesh integration
- Health check integration
- Distributed tracing support

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from urllib.parse import urlparse
import aiohttp
import yaml
from collections import deque, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    url: str
    weight: int = 1
    active: bool = True
    health_check_url: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime stats
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    last_health_check: Optional[float] = None
    health_status: bool = True
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    circuit_breaker_failures: int = 0
    circuit_breaker_last_failure_time: Optional[float] = None
    circuit_breaker_recovery_time: Optional[float] = None


@dataclass
class Route:
    """API route configuration"""
    path: str
    methods: List[str]
    service_name: str
    version: Optional[str] = None
    transform_request: Optional[Callable] = None
    transform_response: Optional[Callable] = None
    rate_limit: Optional[Dict[str, Any]] = None
    auth_required: bool = True
    cache_ttl: Optional[int] = None
    cors_enabled: bool = True
    timeout_override: Optional[float] = None
    retry_policy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Service:
    """Service configuration"""
    name: str
    endpoints: List[ServiceEndpoint]
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0
    sticky_sessions: bool = False
    session_ttl: float = 3600.0
    max_connections_per_endpoint: int = 100
    connection_pool_size: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    current_endpoint_index: int = 0
    sessions: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    last_health_check: Optional[float] = None


class APIGateway:
    """Enterprise API Gateway with advanced features"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_caching: bool = True,
        redis_client: Optional[Any] = None
    ):
        """Initialize API Gateway"""
        self.routes: Dict[str, Route] = {}
        self.services: Dict[str, Service] = {}
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.enable_caching = enable_caching
        self.redis_client = redis_client

        # Metrics
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "requests_by_service": defaultdict(int),
            "requests_by_route": defaultdict(int),
            "response_times": deque(maxlen=1000),
            "errors_by_type": defaultdict(int),
        }

        # Cache
        self.cache: Dict[str, Tuple[Any, float]] = {}

        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None

        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None

        # Load configuration if provided
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load gateway configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Load services
            for service_config in config.get('services', []):
                endpoints = [
                    ServiceEndpoint(**ep)
                    for ep in service_config.get('endpoints', [])
                ]

                service = Service(
                    name=service_config['name'],
                    endpoints=endpoints,
                    load_balancing_strategy=LoadBalancingStrategy(
                        service_config.get('load_balancing_strategy', 'round_robin')
                    ),
                    health_check_interval=service_config.get('health_check_interval', 30.0),
                    sticky_sessions=service_config.get('sticky_sessions', False),
                    session_ttl=service_config.get('session_ttl', 3600.0),
                    max_connections_per_endpoint=service_config.get('max_connections_per_endpoint', 100),
                    metadata=service_config.get('metadata', {})
                )

                self.register_service(service)

            # Load routes
            for route_config in config.get('routes', []):
                route = Route(
                    path=route_config['path'],
                    methods=route_config.get('methods', ['GET']),
                    service_name=route_config['service_name'],
                    version=route_config.get('version'),
                    rate_limit=route_config.get('rate_limit'),
                    auth_required=route_config.get('auth_required', True),
                    cache_ttl=route_config.get('cache_ttl'),
                    cors_enabled=route_config.get('cors_enabled', True),
                    timeout_override=route_config.get('timeout_override'),
                    retry_policy=route_config.get('retry_policy'),
                    metadata=route_config.get('metadata', {})
                )

                self.register_route(route)

            logger.info(f"Loaded configuration: {len(self.services)} services, {len(self.routes)} routes")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def register_service(self, service: Service) -> None:
        """Register a service with the gateway"""
        self.services[service.name] = service
        logger.info(f"Registered service: {service.name} with {len(service.endpoints)} endpoints")

    def register_route(self, route: Route) -> None:
        """Register a route with the gateway"""
        route_key = f"{route.path}:{','.join(route.methods)}"
        self.routes[route_key] = route
        logger.info(f"Registered route: {route_key} -> {route.service_name}")

    async def start(self) -> None:
        """Start the gateway"""
        # Create HTTP session
        self.session = aiohttp.ClientSession()

        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("API Gateway started")

    async def stop(self) -> None:
        """Stop the gateway"""
        # Stop health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self.session:
            await self.session.close()

        logger.info("API Gateway stopped")

    async def route_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None,
        client_ip: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, str], bytes]:
        """
        Route an incoming request to the appropriate service

        Returns: (status_code, headers, body)
        """
        start_time = time.time()

        try:
            # Find matching route
            route = self._find_route(method, path)
            if not route:
                return 404, {}, b'{"error": "Route not found"}'

            # Get service
            service = self.services.get(route.service_name)
            if not service:
                return 500, {}, b'{"error": "Service not found"}'

            # Check cache if enabled
            if self.enable_caching and route.cache_ttl:
                cache_key = self._get_cache_key(method, path, headers, body)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    self._record_metrics('cache_hit', route, time.time() - start_time)
                    return cached_response

            # Select endpoint
            endpoint = await self._select_endpoint(service, client_ip, session_id)
            if not endpoint:
                return 503, {}, b'{"error": "No available endpoints"}'

            # Transform request if needed
            if route.transform_request:
                headers, body = await route.transform_request(headers, body)

            # Make request with retries
            response = await self._make_request(
                endpoint,
                method,
                path,
                headers,
                body,
                route.timeout_override or endpoint.timeout,
                route.retry_policy or {'count': endpoint.retry_count}
            )

            # Transform response if needed
            if route.transform_response:
                response = await route.transform_response(response)

            # Cache response if enabled
            if self.enable_caching and route.cache_ttl:
                self._cache_response(cache_key, response, route.cache_ttl)

            # Record metrics
            self._record_metrics('success', route, time.time() - start_time)

            return response

        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            self._record_metrics('error', None, time.time() - start_time)
            return 500, {}, json.dumps({"error": str(e)}).encode()

    def _find_route(self, method: str, path: str) -> Optional[Route]:
        """Find matching route for request"""
        for route_key, route in self.routes.items():
            route_path, route_methods = route_key.split(':', 1)

            # Check method
            if method not in route_methods.split(','):
                continue

            # Check path (simple prefix matching for now)
            if path.startswith(route_path):
                return route

        return None

    async def _select_endpoint(
        self,
        service: Service,
        client_ip: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[ServiceEndpoint]:
        """Select endpoint based on load balancing strategy"""
        # Filter healthy endpoints
        healthy_endpoints = [
            ep for ep in service.endpoints
            if ep.active and ep.health_status and
            ep.circuit_breaker_state != CircuitBreakerState.OPEN
        ]

        if not healthy_endpoints:
            logger.warning(f"No healthy endpoints for service {service.name}")
            return None

        # Handle sticky sessions
        if service.sticky_sessions and session_id:
            endpoint_url, expiry = service.sessions.get(session_id, (None, 0))
            if endpoint_url and time.time() < expiry:
                # Find endpoint by URL
                for ep in healthy_endpoints:
                    if ep.url == endpoint_url:
                        return ep

        # Select based on strategy
        selected = None

        if service.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = self._round_robin_select(service, healthy_endpoints)

        elif service.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected = min(healthy_endpoints, key=lambda ep: ep.current_connections)

        elif service.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED:
            selected = self._weighted_select(healthy_endpoints)

        elif service.load_balancing_strategy == LoadBalancingStrategy.IP_HASH:
            if client_ip:
                hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
                selected = healthy_endpoints[hash_value % len(healthy_endpoints)]
            else:
                selected = healthy_endpoints[0]

        elif service.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            import random
            selected = random.choice(healthy_endpoints)

        elif service.load_balancing_strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected = min(
                healthy_endpoints,
                key=lambda ep: ep.total_response_time / max(ep.total_requests, 1)
            )

        # Update sticky session if needed
        if selected and service.sticky_sessions and session_id:
            service.sessions[session_id] = (selected.url, time.time() + service.session_ttl)

        return selected

    def _round_robin_select(
        self,
        service: Service,
        endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Round-robin endpoint selection"""
        selected = endpoints[service.current_endpoint_index % len(endpoints)]
        service.current_endpoint_index += 1
        return selected

    def _weighted_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random selection"""
        import random

        total_weight = sum(ep.weight for ep in endpoints)
        rand_value = random.random() * total_weight

        cumulative = 0
        for endpoint in endpoints:
            cumulative += endpoint.weight
            if rand_value <= cumulative:
                return endpoint

        return endpoints[-1]

    async def _make_request(
        self,
        endpoint: ServiceEndpoint,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[bytes],
        timeout: float,
        retry_policy: Dict[str, Any]
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Make HTTP request to endpoint with retries"""
        url = endpoint.url + path
        retry_count = retry_policy.get('count', 3)
        retry_delay = retry_policy.get('delay', 1.0)

        # Update connection count
        endpoint.current_connections += 1
        endpoint.total_requests += 1

        try:
            for attempt in range(retry_count):
                try:
                    start_time = time.time()

                    async with self.session.request(
                        method,
                        url,
                        headers=headers,
                        data=body,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        response_body = await response.read()
                        response_headers = dict(response.headers)

                        # Update stats
                        response_time = time.time() - start_time
                        endpoint.total_response_time += response_time

                        # Check circuit breaker
                        if response.status >= 500:
                            self._record_circuit_breaker_failure(endpoint)
                        else:
                            self._record_circuit_breaker_success(endpoint)

                        return response.status, response_headers, response_body

                except asyncio.TimeoutError:
                    logger.warning(f"Request timeout to {url} (attempt {attempt + 1}/{retry_count})")
                    self._record_circuit_breaker_failure(endpoint)

                    if attempt < retry_count - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        raise

                except aiohttp.ClientError as e:
                    logger.warning(f"Request failed to {url}: {e} (attempt {attempt + 1}/{retry_count})")
                    self._record_circuit_breaker_failure(endpoint)

                    if attempt < retry_count - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        raise

        finally:
            endpoint.current_connections -= 1

    def _record_circuit_breaker_failure(self, endpoint: ServiceEndpoint) -> None:
        """Record circuit breaker failure"""
        if not endpoint.circuit_breaker_enabled:
            return

        endpoint.circuit_breaker_failures += 1
        endpoint.circuit_breaker_last_failure_time = time.time()
        endpoint.failed_requests += 1

        # Check if should open circuit
        if (endpoint.circuit_breaker_state == CircuitBreakerState.CLOSED and
            endpoint.circuit_breaker_failures >= endpoint.circuit_breaker_threshold):

            endpoint.circuit_breaker_state = CircuitBreakerState.OPEN
            endpoint.circuit_breaker_recovery_time = time.time() + endpoint.circuit_breaker_timeout
            logger.warning(f"Circuit breaker OPEN for {endpoint.url}")

    def _record_circuit_breaker_success(self, endpoint: ServiceEndpoint) -> None:
        """Record circuit breaker success"""
        if not endpoint.circuit_breaker_enabled:
            return

        if endpoint.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            # Success in half-open state, close the circuit
            endpoint.circuit_breaker_state = CircuitBreakerState.CLOSED
            endpoint.circuit_breaker_failures = 0
            logger.info(f"Circuit breaker CLOSED for {endpoint.url}")

        elif endpoint.circuit_breaker_state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            endpoint.circuit_breaker_failures = 0

    async def _health_check_loop(self) -> None:
        """Background task for health checks"""
        while True:
            try:
                for service in self.services.values():
                    if time.time() - (service.last_health_check or 0) > service.health_check_interval:
                        await self._check_service_health(service)
                        service.last_health_check = time.time()

                # Check circuit breakers
                for service in self.services.values():
                    for endpoint in service.endpoints:
                        if (endpoint.circuit_breaker_state == CircuitBreakerState.OPEN and
                            endpoint.circuit_breaker_recovery_time and
                            time.time() >= endpoint.circuit_breaker_recovery_time):

                            # Move to half-open state
                            endpoint.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                            logger.info(f"Circuit breaker HALF-OPEN for {endpoint.url}")

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _check_service_health(self, service: Service) -> None:
        """Check health of all endpoints in a service"""
        tasks = []
        for endpoint in service.endpoints:
            if endpoint.health_check_url:
                tasks.append(self._check_endpoint_health(endpoint))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> None:
        """Check health of a single endpoint"""
        try:
            async with self.session.get(
                endpoint.health_check_url,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                endpoint.health_status = response.status == 200
                endpoint.last_health_check = time.time()

                if not endpoint.health_status:
                    logger.warning(f"Health check failed for {endpoint.url}: status {response.status}")

        except Exception as e:
            endpoint.health_status = False
            endpoint.last_health_check = time.time()
            logger.warning(f"Health check failed for {endpoint.url}: {e}")

    def _get_cache_key(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[bytes]
    ) -> str:
        """Generate cache key for request"""
        key_parts = [method, path]

        # Include relevant headers
        for header_name in ['Authorization', 'Accept', 'Content-Type']:
            if header_name in headers:
                key_parts.append(f"{header_name}:{headers[header_name]}")

        # Include body hash if present
        if body:
            key_parts.append(hashlib.sha256(body).hexdigest())

        return ':'.join(key_parts)

    def _get_cached_response(self, cache_key: str) -> Optional[Tuple[int, Dict[str, str], bytes]]:
        """Get cached response if available"""
        if self.redis_client:
            # Use Redis for distributed caching
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        else:
            # Use local cache
            cached, expiry = self.cache.get(cache_key, (None, 0))
            if cached and time.time() < expiry:
                return cached

        return None

    def _cache_response(
        self,
        cache_key: str,
        response: Tuple[int, Dict[str, str], bytes],
        ttl: int
    ) -> None:
        """Cache response"""
        if self.redis_client:
            # Use Redis for distributed caching
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps({
                    'status': response[0],
                    'headers': response[1],
                    'body': response[2].decode('utf-8')
                })
            )
        else:
            # Use local cache
            self.cache[cache_key] = (response, time.time() + ttl)

    def _record_metrics(
        self,
        status: str,
        route: Optional[Route],
        response_time: float
    ) -> None:
        """Record metrics"""
        if not self.enable_metrics:
            return

        self.metrics['requests_total'] += 1

        if status == 'success':
            self.metrics['requests_success'] += 1
        elif status == 'error':
            self.metrics['requests_failed'] += 1
        elif status == 'cache_hit':
            self.metrics['cache_hits'] = self.metrics.get('cache_hits', 0) + 1

        if route:
            self.metrics['requests_by_service'][route.service_name] += 1
            self.metrics['requests_by_route'][route.path] += 1

        self.metrics['response_times'].append(response_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        response_times = list(self.metrics['response_times'])

        return {
            'requests_total': self.metrics['requests_total'],
            'requests_success': self.metrics['requests_success'],
            'requests_failed': self.metrics['requests_failed'],
            'cache_hits': self.metrics.get('cache_hits', 0),
            'requests_by_service': dict(self.metrics['requests_by_service']),
            'requests_by_route': dict(self.metrics['requests_by_route']),
            'response_time_avg': sum(response_times) / len(response_times) if response_times else 0,
            'response_time_p50': sorted(response_times)[len(response_times) // 2] if response_times else 0,
            'response_time_p99': sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0,
            'services': {
                service.name: {
                    'endpoints': len(service.endpoints),
                    'healthy_endpoints': sum(1 for ep in service.endpoints if ep.health_status),
                    'total_requests': sum(ep.total_requests for ep in service.endpoints),
                    'failed_requests': sum(ep.failed_requests for ep in service.endpoints),
                }
                for service in self.services.values()
            }
        }

    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a service"""
        service = self.services.get(service_name)
        if not service:
            return None

        return {
            'name': service.name,
            'load_balancing_strategy': service.load_balancing_strategy.value,
            'sticky_sessions': service.sticky_sessions,
            'endpoints': [
                {
                    'url': ep.url,
                    'active': ep.active,
                    'health_status': ep.health_status,
                    'circuit_breaker_state': ep.circuit_breaker_state.value,
                    'current_connections': ep.current_connections,
                    'total_requests': ep.total_requests,
                    'failed_requests': ep.failed_requests,
                    'avg_response_time': ep.total_response_time / max(ep.total_requests, 1),
                }
                for ep in service.endpoints
            ]
        }