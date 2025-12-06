#!/usr/bin/env python3
"""
Real-time Collaboration Features for TalAI Turing Challenge

Enables multi-user collaboration with WebSocket support,
live updates, and team workspace management.

Features:
- WebSocket support for live updates
- Multi-user validation sessions
- Commenting and annotation system
- Version control for hypotheses
- Team workspace management
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
import hashlib
import logging
from collections import defaultdict
import time
from weakref import WeakSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of collaboration events"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    MESSAGE_SENT = "message_sent"
    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    COMMENT_ADDED = "comment_added"
    ANNOTATION_ADDED = "annotation_added"
    WORKSPACE_UPDATED = "workspace_updated"
    VERSION_CREATED = "version_created"


class UserRole(Enum):
    """User roles in collaboration"""
    OWNER = "owner"
    ADMIN = "admin"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    OBSERVER = "observer"


class SessionStatus(Enum):
    """Validation session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class User:
    """Collaboration user"""
    user_id: str
    username: str
    email: str
    role: UserRole
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Comment:
    """Comment on hypothesis or validation"""
    comment_id: str
    author_id: str
    content: str
    target_id: str  # ID of hypothesis or validation
    target_type: str  # 'hypothesis' or 'validation'
    created_at: datetime = field(default_factory=datetime.now)
    edited_at: Optional[datetime] = None
    replies: List['Comment'] = field(default_factory=list)
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> user_ids


@dataclass
class Annotation:
    """Annotation on specific part of hypothesis"""
    annotation_id: str
    author_id: str
    hypothesis_id: str
    start_position: int
    end_position: int
    content: str
    category: str  # 'suggestion', 'question', 'issue', 'highlight'
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class HypothesisVersion:
    """Version of a hypothesis"""
    version_id: str
    hypothesis_id: str
    version_number: int
    content: str
    author_id: str
    created_at: datetime
    change_summary: str
    parent_version_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CollaborationSession:
    """Multi-user validation session"""
    session_id: str
    name: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE
    participants: Dict[str, User] = field(default_factory=dict)
    hypotheses: Dict[str, Any] = field(default_factory=dict)
    comments: List[Comment] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    versions: Dict[str, List[HypothesisVersion]] = field(default_factory=dict)
    activity_log: List[Dict[str, Any]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationEvent:
    """Real-time collaboration event"""
    event_id: str
    event_type: EventType
    session_id: str
    user_id: str
    payload: Any
    timestamp: datetime = field(default_factory=datetime.now)


class WebSocketConnection:
    """WebSocket connection handler"""

    def __init__(self, connection_id: str, user: User):
        self.connection_id = connection_id
        self.user = user
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to client"""
        await self.message_queue.put(json.dumps(message))

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from client"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return json.loads(message) if isinstance(message, str) else message
        except asyncio.TimeoutError:
            return None

    def is_alive(self) -> bool:
        """Check if connection is still alive"""
        return (datetime.now() - self.last_ping).total_seconds() < 60


class CollaborationHub:
    """Central hub for real-time collaboration"""

    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> connection_ids
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._running = False

    async def create_session(
        self,
        name: str,
        creator: User
    ) -> CollaborationSession:
        """Create a new collaboration session"""
        session = CollaborationSession(
            session_id=str(uuid.uuid4()),
            name=name,
            created_by=creator.user_id,
            participants={creator.user_id: creator}
        )

        self.sessions[session.session_id] = session

        # Log activity
        await self._log_activity(
            session.session_id,
            EventType.WORKSPACE_UPDATED,
            creator.user_id,
            {"action": "session_created", "session_name": name}
        )

        logger.info(f"Created collaboration session: {session.session_id}")
        return session

    async def join_session(
        self,
        session_id: str,
        user: User,
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Join a collaboration session"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}

        session = self.sessions[session_id]

        # Add user to session
        session.participants[user.user_id] = user

        # Create WebSocket connection if provided
        if connection_id:
            connection = WebSocketConnection(connection_id, user)
            self.connections[connection_id] = connection
            self.user_connections[user.user_id].add(connection_id)

        # Broadcast user joined event
        await self._broadcast_event(
            session_id,
            CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.USER_JOINED,
                session_id=session_id,
                user_id=user.user_id,
                payload={"username": user.username, "role": user.role.value}
            )
        )

        # Log activity
        await self._log_activity(
            session_id,
            EventType.USER_JOINED,
            user.user_id,
            {"username": user.username}
        )

        return {
            "status": "success",
            "session": self._serialize_session(session),
            "participants": len(session.participants)
        }

    async def leave_session(
        self,
        session_id: str,
        user_id: str
    ) -> None:
        """Leave a collaboration session"""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]

        # Remove user from session
        if user_id in session.participants:
            del session.participants[user_id]

        # Close WebSocket connections
        for conn_id in list(self.user_connections[user_id]):
            if conn_id in self.connections:
                del self.connections[conn_id]
        del self.user_connections[user_id]

        # Broadcast user left event
        await self._broadcast_event(
            session_id,
            CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.USER_LEFT,
                session_id=session_id,
                user_id=user_id,
                payload={}
            )
        )

        logger.info(f"User {user_id} left session {session_id}")

    async def add_hypothesis(
        self,
        session_id: str,
        user_id: str,
        hypothesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add hypothesis to session"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}

        session = self.sessions[session_id]

        # Check permissions
        user = session.participants.get(user_id)
        if not user or user.role == UserRole.OBSERVER:
            return {"status": "error", "message": "Insufficient permissions"}

        # Add hypothesis
        hypothesis_id = str(uuid.uuid4())
        hypothesis["id"] = hypothesis_id
        hypothesis["created_by"] = user_id
        hypothesis["created_at"] = datetime.now().isoformat()

        session.hypotheses[hypothesis_id] = hypothesis

        # Create initial version
        version = HypothesisVersion(
            version_id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            version_number=1,
            content=hypothesis.get("content", ""),
            author_id=user_id,
            created_at=datetime.now(),
            change_summary="Initial version"
        )

        if hypothesis_id not in session.versions:
            session.versions[hypothesis_id] = []
        session.versions[hypothesis_id].append(version)

        # Broadcast event
        await self._broadcast_event(
            session_id,
            CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.HYPOTHESIS_CREATED,
                session_id=session_id,
                user_id=user_id,
                payload=hypothesis
            )
        )

        return {"status": "success", "hypothesis_id": hypothesis_id}

    async def update_hypothesis(
        self,
        session_id: str,
        user_id: str,
        hypothesis_id: str,
        updates: Dict[str, Any],
        change_summary: str
    ) -> Dict[str, Any]:
        """Update hypothesis with version control"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}

        session = self.sessions[session_id]

        if hypothesis_id not in session.hypotheses:
            return {"status": "error", "message": "Hypothesis not found"}

        # Check permissions
        user = session.participants.get(user_id)
        if not user or user.role == UserRole.OBSERVER:
            return {"status": "error", "message": "Insufficient permissions"}

        # Update hypothesis
        hypothesis = session.hypotheses[hypothesis_id]
        hypothesis.update(updates)
        hypothesis["updated_by"] = user_id
        hypothesis["updated_at"] = datetime.now().isoformat()

        # Create new version
        versions = session.versions.get(hypothesis_id, [])
        new_version_number = len(versions) + 1

        version = HypothesisVersion(
            version_id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            version_number=new_version_number,
            content=hypothesis.get("content", ""),
            author_id=user_id,
            created_at=datetime.now(),
            change_summary=change_summary,
            parent_version_id=versions[-1].version_id if versions else None
        )

        session.versions[hypothesis_id].append(version)

        # Broadcast event
        await self._broadcast_event(
            session_id,
            CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.HYPOTHESIS_UPDATED,
                session_id=session_id,
                user_id=user_id,
                payload={
                    "hypothesis_id": hypothesis_id,
                    "updates": updates,
                    "version": new_version_number,
                    "change_summary": change_summary
                }
            )
        )

        return {
            "status": "success",
            "version": new_version_number,
            "version_id": version.version_id
        }

    async def add_comment(
        self,
        session_id: str,
        user_id: str,
        target_id: str,
        target_type: str,
        content: str
    ) -> Dict[str, Any]:
        """Add comment to hypothesis or validation"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}

        session = self.sessions[session_id]

        # Check permissions
        if user_id not in session.participants:
            return {"status": "error", "message": "User not in session"}

        # Create comment
        comment = Comment(
            comment_id=str(uuid.uuid4()),
            author_id=user_id,
            content=content,
            target_id=target_id,
            target_type=target_type
        )

        session.comments.append(comment)

        # Broadcast event
        await self._broadcast_event(
            session_id,
            CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.COMMENT_ADDED,
                session_id=session_id,
                user_id=user_id,
                payload=asdict(comment)
            )
        )

        return {"status": "success", "comment_id": comment.comment_id}

    async def add_annotation(
        self,
        session_id: str,
        user_id: str,
        hypothesis_id: str,
        start_pos: int,
        end_pos: int,
        content: str,
        category: str
    ) -> Dict[str, Any]:
        """Add annotation to hypothesis"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}

        session = self.sessions[session_id]

        # Check permissions
        if user_id not in session.participants:
            return {"status": "error", "message": "User not in session"}

        # Create annotation
        annotation = Annotation(
            annotation_id=str(uuid.uuid4()),
            author_id=user_id,
            hypothesis_id=hypothesis_id,
            start_position=start_pos,
            end_position=end_pos,
            content=content,
            category=category
        )

        session.annotations.append(annotation)

        # Broadcast event
        await self._broadcast_event(
            session_id,
            CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ANNOTATION_ADDED,
                session_id=session_id,
                user_id=user_id,
                payload=asdict(annotation)
            )
        )

        return {"status": "success", "annotation_id": annotation.annotation_id}

    async def get_hypothesis_history(
        self,
        session_id: str,
        hypothesis_id: str
    ) -> List[HypothesisVersion]:
        """Get version history of a hypothesis"""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return session.versions.get(hypothesis_id, [])

    async def rollback_hypothesis(
        self,
        session_id: str,
        user_id: str,
        hypothesis_id: str,
        version_id: str
    ) -> Dict[str, Any]:
        """Rollback hypothesis to a previous version"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}

        session = self.sessions[session_id]

        # Check permissions (only admin/owner can rollback)
        user = session.participants.get(user_id)
        if not user or user.role not in [UserRole.OWNER, UserRole.ADMIN]:
            return {"status": "error", "message": "Insufficient permissions"}

        # Find the version
        versions = session.versions.get(hypothesis_id, [])
        target_version = None
        for v in versions:
            if v.version_id == version_id:
                target_version = v
                break

        if not target_version:
            return {"status": "error", "message": "Version not found"}

        # Rollback hypothesis content
        hypothesis = session.hypotheses[hypothesis_id]
        hypothesis["content"] = target_version.content
        hypothesis["rolled_back_to"] = version_id
        hypothesis["rolled_back_by"] = user_id
        hypothesis["rolled_back_at"] = datetime.now().isoformat()

        # Create new version marking the rollback
        new_version = HypothesisVersion(
            version_id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            version_number=len(versions) + 1,
            content=target_version.content,
            author_id=user_id,
            created_at=datetime.now(),
            change_summary=f"Rolled back to version {target_version.version_number}"
        )
        session.versions[hypothesis_id].append(new_version)

        return {
            "status": "success",
            "rolled_back_to": target_version.version_number,
            "new_version": new_version.version_number
        }

    async def _broadcast_event(
        self,
        session_id: str,
        event: CollaborationEvent
    ) -> None:
        """Broadcast event to all session participants"""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]

        # Send to all connected users
        for user_id in session.participants:
            for conn_id in self.user_connections.get(user_id, []):
                if conn_id in self.connections:
                    connection = self.connections[conn_id]
                    await connection.send_message(asdict(event))

        # Execute registered handlers
        for handler in self.event_handlers.get(event.event_type, []):
            await handler(event)

    async def _log_activity(
        self,
        session_id: str,
        event_type: EventType,
        user_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log activity in session"""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        activity = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "details": details
        }
        session.activity_log.append(activity)

    def _serialize_session(self, session: CollaborationSession) -> Dict[str, Any]:
        """Serialize session for transmission"""
        return {
            "session_id": session.session_id,
            "name": session.name,
            "status": session.status.value,
            "participants": [
                {
                    "user_id": u.user_id,
                    "username": u.username,
                    "role": u.role.value
                }
                for u in session.participants.values()
            ],
            "hypotheses_count": len(session.hypotheses),
            "comments_count": len(session.comments),
            "annotations_count": len(session.annotations),
            "created_at": session.created_at.isoformat()
        }

    def register_event_handler(
        self,
        event_type: EventType,
        handler: Callable
    ) -> None:
        """Register handler for specific event type"""
        self.event_handlers[event_type].append(handler)

    async def cleanup_inactive_connections(self) -> None:
        """Clean up inactive WebSocket connections"""
        inactive = []
        for conn_id, connection in self.connections.items():
            if not connection.is_alive():
                inactive.append(conn_id)

        for conn_id in inactive:
            connection = self.connections[conn_id]
            user_id = connection.user.user_id

            # Remove from tracking
            del self.connections[conn_id]
            self.user_connections[user_id].discard(conn_id)

            logger.info(f"Cleaned up inactive connection: {conn_id}")


class TeamWorkspaceManager:
    """Manages team workspaces for collaboration"""

    def __init__(self):
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.user_workspaces: Dict[str, Set[str]] = defaultdict(set)

    async def create_workspace(
        self,
        name: str,
        owner_id: str,
        description: str = ""
    ) -> str:
        """Create a new team workspace"""
        workspace_id = str(uuid.uuid4())
        workspace = {
            "id": workspace_id,
            "name": name,
            "description": description,
            "owner_id": owner_id,
            "created_at": datetime.now().isoformat(),
            "members": {owner_id: "owner"},
            "sessions": [],
            "settings": {
                "public": False,
                "allow_guests": False,
                "auto_save": True
            }
        }

        self.workspaces[workspace_id] = workspace
        self.user_workspaces[owner_id].add(workspace_id)

        logger.info(f"Created workspace: {workspace_id}")
        return workspace_id

    async def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: str = "member"
    ) -> bool:
        """Add member to workspace"""
        if workspace_id not in self.workspaces:
            return False

        workspace = self.workspaces[workspace_id]
        workspace["members"][user_id] = role
        self.user_workspaces[user_id].add(workspace_id)

        return True

    def get_user_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all workspaces for a user"""
        workspace_ids = self.user_workspaces.get(user_id, set())
        return [
            self.workspaces[wid]
            for wid in workspace_ids
            if wid in self.workspaces
        ]