"""
Workflow Version - Version control and rollback for workflows

Manages workflow versioning, history tracking, and rollback capabilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import copy

from .workflow_dag import WorkflowDAG


@dataclass
class WorkflowVersion:
    """Version information for a workflow"""
    version_id: str
    version_number: str
    dag: WorkflowDAG
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    checksum: str = ""
    parent_version: Optional[str] = None
    changes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_current: bool = False

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self.calculate_checksum()

    def calculate_checksum(self) -> str:
        """Calculate checksum for this version"""
        return self.dag.calculate_checksum()


class VersionManager:
    """Manages workflow versions and history"""

    def __init__(self):
        self.versions: Dict[str, List[WorkflowVersion]] = {}
        self.current_versions: Dict[str, str] = {}

    def create_version(self, workflow_id: str, dag: WorkflowDAG,
                      version_number: Optional[str] = None,
                      description: str = "",
                      created_by: str = "") -> WorkflowVersion:
        """Create new version of workflow"""
        # Auto-generate version number if not provided
        if not version_number:
            existing_versions = self.versions.get(workflow_id, [])
            if existing_versions:
                last_version = existing_versions[-1].version_number
                major, minor, patch = map(int, last_version.split('.'))
                version_number = f"{major}.{minor}.{patch + 1}"
            else:
                version_number = "1.0.0"

        # Get parent version
        parent_version = None
        if workflow_id in self.current_versions:
            parent_version = self.current_versions[workflow_id]

        # Create version
        version = WorkflowVersion(
            version_id=f"{workflow_id}_v{version_number}",
            version_number=version_number,
            dag=dag.clone(),
            created_by=created_by,
            description=description,
            parent_version=parent_version
        )

        # Detect changes
        if parent_version:
            parent = self.get_version(workflow_id, parent_version)
            if parent:
                version.changes = self._detect_changes(parent.dag, dag)

        # Store version
        if workflow_id not in self.versions:
            self.versions[workflow_id] = []
        self.versions[workflow_id].append(version)

        # Set as current
        self.set_current_version(workflow_id, version.version_id)

        return version

    def get_version(self, workflow_id: str,
                   version_id: Optional[str] = None) -> Optional[WorkflowVersion]:
        """Get specific version or current version"""
        if workflow_id not in self.versions:
            return None

        if not version_id:
            version_id = self.current_versions.get(workflow_id)

        for version in self.versions[workflow_id]:
            if version.version_id == version_id:
                return version

        return None

    def list_versions(self, workflow_id: str) -> List[WorkflowVersion]:
        """List all versions of a workflow"""
        return self.versions.get(workflow_id, [])

    def set_current_version(self, workflow_id: str, version_id: str):
        """Set current version for workflow"""
        # Clear previous current flag
        if workflow_id in self.versions:
            for version in self.versions[workflow_id]:
                version.is_current = False

        # Set new current
        self.current_versions[workflow_id] = version_id

        # Update version flag
        for version in self.versions.get(workflow_id, []):
            if version.version_id == version_id:
                version.is_current = True
                break

    def rollback(self, workflow_id: str, target_version: str) -> WorkflowVersion:
        """Rollback to specific version"""
        target = self.get_version(workflow_id, target_version)
        if not target:
            raise ValueError(f"Version {target_version} not found")

        # Create rollback version
        rollback_version = self.create_version(
            workflow_id=workflow_id,
            dag=target.dag,
            description=f"Rollback to {target_version}",
            created_by="system"
        )

        rollback_version.changes = [f"Rolled back to version {target_version}"]
        return rollback_version

    def compare_versions(self, workflow_id: str,
                        version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.get_version(workflow_id, version1_id)
        v2 = self.get_version(workflow_id, version2_id)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        changes = self._detect_changes(v1.dag, v2.dag)

        return {
            "version1": version1_id,
            "version2": version2_id,
            "changes": changes,
            "nodes_added": self._get_added_nodes(v1.dag, v2.dag),
            "nodes_removed": self._get_removed_nodes(v1.dag, v2.dag),
            "nodes_modified": self._get_modified_nodes(v1.dag, v2.dag),
            "edges_added": self._get_added_edges(v1.dag, v2.dag),
            "edges_removed": self._get_removed_edges(v1.dag, v2.dag)
        }

    def get_version_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get version history for workflow"""
        versions = self.list_versions(workflow_id)
        history = []

        for version in versions:
            history.append({
                "version_id": version.version_id,
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "description": version.description,
                "is_current": version.is_current,
                "parent_version": version.parent_version,
                "changes": version.changes,
                "checksum": version.checksum
            })

        return history

    def _detect_changes(self, old_dag: WorkflowDAG,
                       new_dag: WorkflowDAG) -> List[str]:
        """Detect changes between two DAG versions"""
        changes = []

        # Check nodes
        added_nodes = self._get_added_nodes(old_dag, new_dag)
        if added_nodes:
            changes.append(f"Added nodes: {', '.join(added_nodes)}")

        removed_nodes = self._get_removed_nodes(old_dag, new_dag)
        if removed_nodes:
            changes.append(f"Removed nodes: {', '.join(removed_nodes)}")

        modified_nodes = self._get_modified_nodes(old_dag, new_dag)
        if modified_nodes:
            changes.append(f"Modified nodes: {', '.join(modified_nodes)}")

        # Check edges
        added_edges = self._get_added_edges(old_dag, new_dag)
        if added_edges:
            changes.append(f"Added {len(added_edges)} edges")

        removed_edges = self._get_removed_edges(old_dag, new_dag)
        if removed_edges:
            changes.append(f"Removed {len(removed_edges)} edges")

        return changes

    def _get_added_nodes(self, old_dag: WorkflowDAG,
                        new_dag: WorkflowDAG) -> List[str]:
        """Get nodes added in new version"""
        old_nodes = set(old_dag.nodes.keys())
        new_nodes = set(new_dag.nodes.keys())
        return list(new_nodes - old_nodes)

    def _get_removed_nodes(self, old_dag: WorkflowDAG,
                          new_dag: WorkflowDAG) -> List[str]:
        """Get nodes removed in new version"""
        old_nodes = set(old_dag.nodes.keys())
        new_nodes = set(new_dag.nodes.keys())
        return list(old_nodes - new_nodes)

    def _get_modified_nodes(self, old_dag: WorkflowDAG,
                           new_dag: WorkflowDAG) -> List[str]:
        """Get nodes modified in new version"""
        modified = []
        common_nodes = set(old_dag.nodes.keys()) & set(new_dag.nodes.keys())

        for node_id in common_nodes:
            old_node = old_dag.nodes[node_id]
            new_node = new_dag.nodes[node_id]

            # Compare node properties
            if (old_node.name != new_node.name or
                old_node.type != new_node.type or
                old_node.inputs != new_node.inputs or
                old_node.metadata.priority != new_node.metadata.priority):
                modified.append(node_id)

        return modified

    def _get_added_edges(self, old_dag: WorkflowDAG,
                        new_dag: WorkflowDAG) -> List[tuple]:
        """Get edges added in new version"""
        old_edges = set(old_dag.graph.edges())
        new_edges = set(new_dag.graph.edges())
        return list(new_edges - old_edges)

    def _get_removed_edges(self, old_dag: WorkflowDAG,
                          new_dag: WorkflowDAG) -> List[tuple]:
        """Get edges removed in new version"""
        old_edges = set(old_dag.graph.edges())
        new_edges = set(new_dag.graph.edges())
        return list(old_edges - new_edges)