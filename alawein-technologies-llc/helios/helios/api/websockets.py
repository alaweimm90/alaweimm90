"""
WebSocket Handler - Real-time updates

Provides:
- Real-time leaderboard updates
- Live validation feedback
- Algorithm performance updates
- System event streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set, Dict, Any
import json
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# ============================================================================
# CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept and register new connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove disconnected client."""
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe connection to channel."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(channel)

    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe from channel."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(channel)

    async def broadcast(self, message: Dict[str, Any], channel: str = None):
        """Broadcast message to subscribed clients."""
        for connection in self.active_connections:
            if channel is None or channel in self.subscriptions.get(connection, set()):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to connection: {e}")


# Global connection manager
manager = ConnectionManager()


# ============================================================================
# MESSAGE TYPES
# ============================================================================

def create_message(msg_type: str, data: Any = None, channel: str = None) -> Dict[str, Any]:
    """Create properly formatted WebSocket message."""
    return {
        "type": msg_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "channel": channel,
    }


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time updates.

    Supported operations:
    - subscribe: Subscribe to channel
    - unsubscribe: Unsubscribe from channel
    - ping: Keep-alive check
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            msg_type = data.get("type")
            channel = data.get("channel")

            if msg_type == "subscribe":
                await manager.subscribe(websocket, channel)
                await websocket.send_json(create_message("subscribed", {"channel": channel}))
                logger.info(f"Client subscribed to {channel}")

            elif msg_type == "unsubscribe":
                await manager.unsubscribe(websocket, channel)
                await websocket.send_json(create_message("unsubscribed", {"channel": channel}))

            elif msg_type == "ping":
                await websocket.send_json(create_message("pong"))

            elif msg_type == "list_channels":
                channels = [
                    "leaderboard.updates",
                    "validation.complete",
                    "algorithm.performance",
                    "system.events",
                ]
                await websocket.send_json(create_message("channels", {"available": channels}))

            else:
                await websocket.send_json(create_message("error", {"message": f"Unknown message type: {msg_type}"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# BROADCAST FUNCTIONS (For use by other services)
# ============================================================================

async def broadcast_leaderboard_update(algorithm: str, new_rank: int, new_quality: float):
    """Broadcast leaderboard update to subscribed clients."""
    message = create_message(
        "leaderboard_update",
        {
            "algorithm": algorithm,
            "new_rank": new_rank,
            "new_quality": new_quality,
        },
        "leaderboard.updates"
    )
    await manager.broadcast(message, "leaderboard.updates")


async def broadcast_validation_complete(hypothesis_id: str, score: float, status: str):
    """Broadcast validation completion."""
    message = create_message(
        "validation_complete",
        {
            "hypothesis_id": hypothesis_id,
            "score": score,
            "status": status,
        },
        "validation.complete"
    )
    await manager.broadcast(message, "validation.complete")


async def broadcast_algorithm_performance(algorithm_id: str, speedup: float, quality: float):
    """Broadcast algorithm performance update."""
    message = create_message(
        "performance_update",
        {
            "algorithm_id": algorithm_id,
            "speedup": speedup,
            "quality": quality,
        },
        "algorithm.performance"
    )
    await manager.broadcast(message, "algorithm.performance")


async def broadcast_system_event(event_type: str, event_data: Dict[str, Any]):
    """Broadcast system-wide event."""
    message = create_message(
        "system_event",
        {
            "event_type": event_type,
            "event_data": event_data,
        },
        "system.events"
    )
    await manager.broadcast(message, "system.events")


# ============================================================================
# PERIODIC BROADCAST TASK (For demo/testing)
# ============================================================================

async def periodic_leaderboard_updates():
    """
    Simulate periodic leaderboard updates.
    In production, this would be triggered by real algorithm executions.
    """
    import random

    algorithms = [
        "Transformer Arch", "VQE Hybrid", "Hybrid Metaheuristic",
        "Structure GNN", "Differentiable NAS"
    ]

    while True:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds

            algorithm = random.choice(algorithms)
            new_quality = round(random.uniform(0.80, 0.95), 2)

            await broadcast_leaderboard_update(algorithm, random.randint(1, 10), new_quality)

        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
            await asyncio.sleep(5)
