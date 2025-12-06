"""
ORCHEX Redis Blackboard Connector

Provides shared state management between ORCHEX and Libria
using Redis as a blackboard architecture.
"""

import redis
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ATLASBlackboard:
    """
    Redis blackboard for ORCHEX-Libria shared state

    Key Schema:
    - ORCHEX:agent:{agent_id}:type = agent type
    - ORCHEX:agent:{agent_id}:skill_level = skill level
    - ORCHEX:agent:{agent_id}:workload = current workload
    - ORCHEX:agent:{agent_id}:available = availability status
    - ORCHEX:agent:{agent_id}:history = execution history
    - ORCHEX:agent:{agent_id}:connections = connected agents (set)
    - execution:{execution_id}:* = execution records
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize blackboard connection

        Args:
            redis_url: Redis connection URL
        """
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.redis_url = redis_url
        logger.info(f"Initialized ORCHEX Blackboard at {redis_url}")

    def ping(self) -> bool:
        """
        Test Redis connection

        Returns:
            True if connection successful
        """
        try:
            return self.redis.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            return False

    def store_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """
        Store agent state in blackboard

        Args:
            agent_id: Agent identifier
            state: Dictionary with agent state
        """
        key = f"ORCHEX:agent:{agent_id}"
        self.redis.hset(key, mapping=state)
        logger.debug(f"Stored state for agent {agent_id}")

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """
        Retrieve agent state from blackboard

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with agent state
        """
        key = f"ORCHEX:agent:{agent_id}"
        state = self.redis.hgetall(key)
        return state

    def add_agent_execution(self, agent_id: str, execution: Dict):
        """
        Add execution record to agent history

        Args:
            agent_id: Agent identifier
            execution: Execution record dictionary
        """
        key = f"ORCHEX:agent:{agent_id}:history"
        self.redis.lpush(key, json.dumps(execution))
        logger.debug(f"Added execution record for agent {agent_id}")

    def get_agent_history(
        self, agent_id: str, limit: int = 100
    ) -> List[Dict]:
        """
        Get agent execution history

        Args:
            agent_id: Agent identifier
            limit: Maximum number of records to retrieve

        Returns:
            List of execution records
        """
        key = f"ORCHEX:agent:{agent_id}:history"
        records = self.redis.lrange(key, 0, limit - 1)
        return [json.loads(r) for r in records]

    def set_agent_connections(self, agent_id: str, connected_agent_ids: List[str]):
        """
        Set agent communication connections

        Args:
            agent_id: Agent identifier
            connected_agent_ids: List of connected agent IDs
        """
        key = f"ORCHEX:agent:{agent_id}:connections"
        self.redis.delete(key)  # Clear existing
        if connected_agent_ids:
            self.redis.sadd(key, *connected_agent_ids)
        logger.debug(f"Set {len(connected_agent_ids)} connections for agent {agent_id}")

    def get_agent_connections(self, agent_id: str) -> List[str]:
        """
        Get agent communication connections

        Args:
            agent_id: Agent identifier

        Returns:
            List of connected agent IDs
        """
        key = f"ORCHEX:agent:{agent_id}:connections"
        return list(self.redis.smembers(key))

    def store_execution_record(
        self,
        execution_id: str,
        agent_id: str,
        task_id: str,
        duration: float,
        success: bool,
        quality: float,
    ):
        """
        Store execution record in blackboard

        Args:
            execution_id: Unique execution identifier
            agent_id: Agent that executed task
            task_id: Task identifier
            duration: Execution time in seconds
            success: Whether execution succeeded
            quality: Quality score (0.0-1.0)
        """
        key_prefix = f"execution:{execution_id}"
        self.redis.set(f"{key_prefix}:agent", agent_id)
        self.redis.set(f"{key_prefix}:task", task_id)
        self.redis.set(f"{key_prefix}:duration", duration)
        self.redis.set(f"{key_prefix}:success", int(success))
        self.redis.set(f"{key_prefix}:quality", quality)
        logger.debug(f"Stored execution record {execution_id}")

    def get_execution_record(self, execution_id: str) -> Optional[Dict]:
        """
        Retrieve execution record

        Args:
            execution_id: Unique execution identifier

        Returns:
            Execution record dictionary or None
        """
        key_prefix = f"execution:{execution_id}"
        agent_id = self.redis.get(f"{key_prefix}:agent")

        if not agent_id:
            return None

        return {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "task_id": self.redis.get(f"{key_prefix}:task"),
            "duration": float(self.redis.get(f"{key_prefix}:duration") or 0),
            "success": bool(int(self.redis.get(f"{key_prefix}:success") or 0)),
            "quality": float(self.redis.get(f"{key_prefix}:quality") or 0),
        }

    def clear_agent_state(self, agent_id: str):
        """
        Clear all state for an agent

        Args:
            agent_id: Agent identifier
        """
        pattern = f"ORCHEX:agent:{agent_id}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
        logger.info(f"Cleared state for agent {agent_id}")

    def get_all_agents(self) -> List[str]:
        """
        Get list of all registered agents

        Returns:
            List of agent IDs
        """
        pattern = "ORCHEX:agent:*:type"
        keys = self.redis.keys(pattern)
        # Extract agent IDs from keys like "ORCHEX:agent:agent_0:type"
        agent_ids = [k.split(":")[2] for k in keys]
        return agent_ids

    def __repr__(self) -> str:
        return f"ATLASBlackboard(url={self.redis_url})"
