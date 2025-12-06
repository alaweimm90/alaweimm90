"""
Query optimization with analysis, caching, and automatic index recommendations.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlparse import format as sql_format
from sqlparse import parse as sql_parse

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Represents a query execution plan."""
    query: str
    plan: Dict[str, Any]
    cost: float
    execution_time_ms: float
    rows_affected: int
    indexes_used: List[str]
    recommendations: List[str]


@dataclass
class QueryStats:
    """Statistics for a specific query."""
    query_hash: str
    query_template: str
    execution_count: int = 0
    total_time_ms: float = 0
    avg_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    avg_rows: float = 0
    cache_hits: int = 0
    last_executed: Optional[datetime] = None


class QueryOptimizer:
    """
    Advanced query optimizer with plan caching, automatic optimization,
    and index recommendations.
    """

    def __init__(self, connection_pool):
        """Initialize query optimizer."""
        self.pool = connection_pool
        self._query_cache: Dict[str, Any] = {}
        self._query_stats: Dict[str, QueryStats] = {}
        self._slow_query_threshold_ms = 1000
        self._cache_ttl_seconds = 3600

        # Common query patterns for optimization
        self._optimization_rules = [
            self._optimize_select_star,
            self._optimize_missing_indexes,
            self._optimize_join_order,
            self._optimize_subqueries,
            self._optimize_pagination
        ]

    def _normalize_query(self, query: str) -> Tuple[str, str]:
        """
        Normalize query for caching and analysis.

        Returns:
            Tuple of (query_hash, normalized_query)
        """
        # Remove comments and extra whitespace
        normalized = sql_format(query, reindent=True, keyword_case='upper')

        # Replace parameter values with placeholders
        normalized = re.sub(r"'[^']*'", '?', normalized)
        normalized = re.sub(r'\b\d+\b', '?', normalized)

        # Generate hash
        query_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]

        return query_hash, normalized

    async def optimize_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        analyze: bool = True
    ) -> Tuple[str, QueryPlan]:
        """
        Optimize a query and return execution plan.

        Args:
            query: SQL query to optimize
            params: Query parameters
            analyze: Whether to analyze execution plan

        Returns:
            Tuple of (optimized_query, execution_plan)
        """
        query_hash, normalized = self._normalize_query(query)

        # Check cache
        if query_hash in self._query_cache:
            cache_entry = self._query_cache[query_hash]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl_seconds:
                self._update_stats(query_hash, normalized, 0, 0, cache_hit=True)
                return cache_entry['optimized_query'], cache_entry['plan']

        # Apply optimization rules
        optimized_query = query
        for rule in self._optimization_rules:
            optimized_query = rule(optimized_query)

        # Analyze query plan if requested
        plan = None
        if analyze:
            plan = await self._analyze_query_plan(optimized_query, params)

        # Cache result
        self._query_cache[query_hash] = {
            'optimized_query': optimized_query,
            'plan': plan,
            'timestamp': time.time()
        }

        return optimized_query, plan

    async def _analyze_query_plan(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Analyze query execution plan."""
        start_time = time.time()

        try:
            # Get execution plan (PostgreSQL example)
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

            async with self.pool.get_connection() as conn:
                result = await conn.execute(explain_query, params or {})
                plan_data = result.fetchone()[0]

            execution_time = (time.time() - start_time) * 1000

            # Parse plan
            plan = QueryPlan(
                query=query,
                plan=plan_data[0] if plan_data else {},
                cost=plan_data[0]['Plan']['Total Cost'] if plan_data else 0,
                execution_time_ms=execution_time,
                rows_affected=plan_data[0]['Plan']['Actual Rows'] if plan_data else 0,
                indexes_used=self._extract_indexes(plan_data),
                recommendations=self._generate_recommendations(plan_data)
            )

            # Update statistics
            query_hash, normalized = self._normalize_query(query)
            self._update_stats(
                query_hash,
                normalized,
                execution_time,
                plan.rows_affected,
                cache_hit=False
            )

            # Log slow queries
            if execution_time > self._slow_query_threshold_ms:
                logger.warning(
                    f"Slow query detected ({execution_time:.2f}ms): "
                    f"{query[:100]}..."
                )

            return plan

        except Exception as e:
            logger.error(f"Failed to analyze query plan: {e}")
            return QueryPlan(
                query=query,
                plan={},
                cost=0,
                execution_time_ms=0,
                rows_affected=0,
                indexes_used=[],
                recommendations=[]
            )

    def _extract_indexes(self, plan_data: Dict[str, Any]) -> List[str]:
        """Extract used indexes from query plan."""
        indexes = []

        def traverse_plan(node):
            if 'Index Name' in node:
                indexes.append(node['Index Name'])

            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)

        if plan_data and len(plan_data) > 0:
            traverse_plan(plan_data[0]['Plan'])

        return indexes

    def _generate_recommendations(self, plan_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on query plan."""
        recommendations = []

        if not plan_data or len(plan_data) == 0:
            return recommendations

        plan = plan_data[0]['Plan']

        # Check for sequential scans on large tables
        if plan.get('Node Type') == 'Seq Scan' and plan.get('Actual Rows', 0) > 1000:
            table_name = plan.get('Relation Name', 'unknown')
            recommendations.append(
                f"Consider adding an index on {table_name} to avoid sequential scan"
            )

        # Check for missing indexes on join columns
        if 'Plans' in plan:
            for child in plan['Plans']:
                if child.get('Node Type') == 'Nested Loop' and child.get('Join Type'):
                    recommendations.append(
                        "Consider using Hash Join or Merge Join for better performance"
                    )

        # Check for high cost operations
        if plan.get('Total Cost', 0) > 10000:
            recommendations.append(
                "Query has high cost. Consider breaking into smaller operations"
            )

        return recommendations

    def _optimize_select_star(self, query: str) -> str:
        """Replace SELECT * with specific columns."""
        # This is a simplified implementation
        # Real implementation would parse table schema
        return query

    def _optimize_missing_indexes(self, query: str) -> str:
        """Add index hints for missing indexes."""
        # Simplified implementation
        return query

    def _optimize_join_order(self, query: str) -> str:
        """Optimize join order based on table statistics."""
        # Simplified implementation
        return query

    def _optimize_subqueries(self, query: str) -> str:
        """Convert subqueries to joins where possible."""
        # Simplified implementation
        return query

    def _optimize_pagination(self, query: str) -> str:
        """Optimize LIMIT/OFFSET pagination."""
        # Check for OFFSET pattern
        offset_pattern = r'OFFSET\s+(\d+)'
        match = re.search(offset_pattern, query, re.IGNORECASE)

        if match and int(match.group(1)) > 1000:
            # Suggest using keyset pagination
            logger.info("Consider using keyset pagination for large offsets")

        return query

    def _update_stats(
        self,
        query_hash: str,
        normalized_query: str,
        execution_time_ms: float,
        rows_affected: int,
        cache_hit: bool
    ) -> None:
        """Update query statistics."""
        if query_hash not in self._query_stats:
            self._query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_template=normalized_query
            )

        stats = self._query_stats[query_hash]
        stats.execution_count += 1

        if cache_hit:
            stats.cache_hits += 1
        else:
            stats.total_time_ms += execution_time_ms
            stats.avg_time_ms = stats.total_time_ms / (stats.execution_count - stats.cache_hits)
            stats.min_time_ms = min(stats.min_time_ms, execution_time_ms)
            stats.max_time_ms = max(stats.max_time_ms, execution_time_ms)
            stats.avg_rows = ((stats.avg_rows * (stats.execution_count - 1)) + rows_affected) / stats.execution_count

        stats.last_executed = datetime.utcnow()

    async def suggest_indexes(self, table_name: str) -> List[str]:
        """
        Suggest indexes based on query patterns.

        Args:
            table_name: Table to analyze

        Returns:
            List of index suggestions
        """
        suggestions = []

        # Analyze query patterns for the table
        table_queries = [
            stats for stats in self._query_stats.values()
            if table_name.upper() in stats.query_template.upper()
        ]

        # Find frequently used WHERE columns
        where_columns = set()
        join_columns = set()

        for stats in table_queries:
            # Extract WHERE columns (simplified)
            where_match = re.findall(
                rf'{table_name}\.\w+',
                stats.query_template,
                re.IGNORECASE
            )
            where_columns.update(where_match)

        # Generate index suggestions
        for col in where_columns:
            suggestions.append(f"CREATE INDEX idx_{table_name}_{col} ON {table_name}({col})")

        return suggestions

    async def get_slow_queries(
        self,
        threshold_ms: Optional[float] = None
    ) -> List[QueryStats]:
        """Get list of slow queries."""
        threshold = threshold_ms or self._slow_query_threshold_ms

        slow_queries = [
            stats for stats in self._query_stats.values()
            if stats.avg_time_ms > threshold
        ]

        # Sort by average execution time
        slow_queries.sort(key=lambda x: x.avg_time_ms, reverse=True)

        return slow_queries

    async def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        total_queries = sum(s.execution_count for s in self._query_stats.values())
        total_cache_hits = sum(s.cache_hits for s in self._query_stats.values())
        cache_hit_rate = total_cache_hits / total_queries if total_queries > 0 else 0

        return {
            "total_queries": total_queries,
            "unique_queries": len(self._query_stats),
            "cache_size": len(self._query_cache),
            "cache_hits": total_cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "slow_queries": len(await self.get_slow_queries()),
            "optimization_rules": len(self._optimization_rules)
        }