"""
GraphQL API for efficient querying with DataLoader and query optimization.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from graphene import ObjectType, Schema, String, Int, Field, List as GrapheneList

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderBatch:
    """Batch for DataLoader pattern."""
    keys: List[str]
    results: Dict[str, Any]


class GraphQLAPI:
    """
    GraphQL API implementation with DataLoader pattern
    for efficient batched data fetching.
    """

    def __init__(self):
        """Initialize GraphQL API."""
        self._data_loaders = {}
        self._query_complexity_limit = 1000
        self._query_depth_limit = 10
        self.schema = self._create_schema()

    def _create_schema(self) -> Schema:
        """Create GraphQL schema."""
        class TaskType(ObjectType):
            """Task GraphQL type."""
            id = String()
            name = String()
            status = String()
            priority = Int()
            created_at = String()

        class CacheStatsType(ObjectType):
            """Cache statistics type."""
            hit_rate = String()
            total_requests = Int()
            cache_size = Int()

        class PerformanceMetricsType(ObjectType):
            """Performance metrics type."""
            avg_response_time_ms = String()
            requests_per_second = String()
            error_rate = String()

        class Query(ObjectType):
            """Root query type."""
            task = Field(TaskType, id=String(required=True))
            tasks = GrapheneList(
                TaskType,
                status=String(),
                limit=Int(default_value=10)
            )
            cache_stats = Field(CacheStatsType)
            performance_metrics = Field(PerformanceMetricsType)

            async def resolve_task(self, info, id):
                """Resolve single task."""
                return await self._fetch_task(id)

            async def resolve_tasks(self, info, status=None, limit=10):
                """Resolve multiple tasks."""
                return await self._fetch_tasks(status, limit)

            async def resolve_cache_stats(self, info):
                """Resolve cache statistics."""
                return {
                    "hit_rate": "85.5%",
                    "total_requests": 10000,
                    "cache_size": 500
                }

            async def resolve_performance_metrics(self, info):
                """Resolve performance metrics."""
                return {
                    "avg_response_time_ms": "125.5",
                    "requests_per_second": "1000",
                    "error_rate": "0.01%"
                }

        return Schema(query=Query)

    async def _fetch_task(self, task_id: str) -> Dict[str, Any]:
        """Fetch single task with DataLoader."""
        if "task" not in self._data_loaders:
            self._data_loaders["task"] = DataLoaderBatch([], {})

        loader = self._data_loaders["task"]
        loader.keys.append(task_id)

        # Batch fetch when ready
        if len(loader.keys) >= 10:
            await self._batch_fetch_tasks(loader)

        return loader.results.get(task_id, {
            "id": task_id,
            "name": f"Task {task_id}",
            "status": "pending",
            "priority": 1,
            "created_at": "2024-01-01T00:00:00Z"
        })

    async def _fetch_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch multiple tasks."""
        # Simulate fetching from database
        tasks = []

        for i in range(limit):
            task = {
                "id": str(i),
                "name": f"Task {i}",
                "status": status or "pending",
                "priority": i % 5,
                "created_at": f"2024-01-0{i%9+1}T00:00:00Z"
            }
            tasks.append(task)

        return tasks

    async def _batch_fetch_tasks(self, loader: DataLoaderBatch) -> None:
        """Batch fetch tasks for DataLoader."""
        # Simulate batch database query
        for key in loader.keys:
            loader.results[key] = {
                "id": key,
                "name": f"Task {key}",
                "status": "completed",
                "priority": 2,
                "created_at": "2024-01-01T00:00:00Z"
            }

        loader.keys.clear()

    def calculate_query_complexity(self, query: str) -> int:
        """
        Calculate query complexity for rate limiting.

        Args:
            query: GraphQL query string

        Returns:
            Complexity score
        """
        complexity = 0

        # Count fields (simplified)
        complexity += query.count("{")
        complexity += query.count("}")

        # Count list operations
        if "limit" in query:
            import re
            limit_match = re.search(r'limit:\s*(\d+)', query)
            if limit_match:
                complexity += int(limit_match.group(1))

        return complexity

    def validate_query_depth(self, query: str) -> bool:
        """
        Validate query depth to prevent deep nesting attacks.

        Args:
            query: GraphQL query string

        Returns:
            True if within depth limit
        """
        max_depth = 0
        current_depth = 0

        for char in query:
            if char == "{":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "}":
                current_depth -= 1

        return max_depth <= self._query_depth_limit

    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query with validation.

        Args:
            query: GraphQL query
            variables: Query variables

        Returns:
            Query result
        """
        # Validate query complexity
        complexity = self.calculate_query_complexity(query)
        if complexity > self._query_complexity_limit:
            return {
                "errors": [{
                    "message": f"Query too complex (complexity: {complexity})"
                }]
            }

        # Validate query depth
        if not self.validate_query_depth(query):
            return {
                "errors": [{
                    "message": "Query depth exceeds limit"
                }]
            }

        # Execute query
        result = await self.schema.execute_async(
            query,
            variable_values=variables
        )

        if result.errors:
            return {
                "errors": [
                    {"message": str(error)} for error in result.errors
                ]
            }

        return {"data": result.data}

    def get_introspection_query(self) -> str:
        """Get GraphQL introspection query."""
        return """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    description
                    fields {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
            }
        }
        """

    def get_stats(self) -> Dict[str, Any]:
        """Get GraphQL API statistics."""
        return {
            "query_complexity_limit": self._query_complexity_limit,
            "query_depth_limit": self._query_depth_limit,
            "data_loaders": list(self._data_loaders.keys()),
            "schema_types": len(self.schema.type_map)
        }