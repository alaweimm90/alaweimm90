"""
Collaboration tools for research consortia.

Implements project management, contribution tracking, authorship determination,
milestone tracking, and resource allocation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """Project status."""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class TaskStatus(Enum):
    """Task status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    UNDER_REVIEW = "under_review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ContributionType(Enum):
    """Types of research contributions."""
    CONCEPTUALIZATION = "conceptualization"
    METHODOLOGY = "methodology"
    SOFTWARE = "software"
    VALIDATION = "validation"
    FORMAL_ANALYSIS = "formal_analysis"
    INVESTIGATION = "investigation"
    RESOURCES = "resources"
    DATA_CURATION = "data_curation"
    WRITING_ORIGINAL = "writing_original"
    WRITING_REVIEW = "writing_review"
    VISUALIZATION = "visualization"
    SUPERVISION = "supervision"
    PROJECT_ADMIN = "project_admin"
    FUNDING = "funding"


class ResourceType(Enum):
    """Types of research resources."""
    COMPUTE = "compute"
    STORAGE = "storage"
    EQUIPMENT = "equipment"
    PERSONNEL = "personnel"
    FUNDING = "funding"
    DATA = "data"
    SOFTWARE_LICENSE = "software_license"


@dataclass
class Project:
    """Research project."""
    id: str
    name: str
    description: str
    lead_institution: str
    participating_institutions: List[str]
    principal_investigator: str
    start_date: datetime
    end_date: datetime
    status: ProjectStatus = ProjectStatus.PLANNING
    objectives: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    budget: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Project task."""
    id: str
    project_id: str
    name: str
    description: str
    assignees: List[str]
    status: TaskStatus = TaskStatus.NOT_STARTED
    priority: int = 5  # 1-10, higher is more important
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)


@dataclass
class Contribution:
    """Research contribution."""
    id: str
    contributor_id: str
    project_id: str
    contribution_type: ContributionType
    description: str
    timestamp: datetime
    hours_spent: float = 0.0
    impact_score: float = 0.0  # 0-1, calculated based on various factors
    evidence: List[str] = field(default_factory=list)
    peer_validations: List[str] = field(default_factory=list)


@dataclass
class Milestone:
    """Project milestone."""
    id: str
    project_id: str
    name: str
    description: str
    target_date: datetime
    completed_date: Optional[datetime] = None
    completion_criteria: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    progress_percentage: float = 0.0


@dataclass
class ResourceAllocation:
    """Resource allocation."""
    id: str
    project_id: str
    resource_type: ResourceType
    resource_id: str
    allocated_amount: float
    unit: str  # 'hours', 'GB', 'USD', etc.
    start_date: datetime
    end_date: datetime
    requestor_id: str
    approver_id: Optional[str] = None
    status: str = "pending"  # 'pending', 'approved', 'denied', 'expired'
    usage_tracking: Dict[str, float] = field(default_factory=dict)


class ProjectManager:
    """Manage research projects."""

    def __init__(self):
        """Initialize project manager."""
        self.projects: Dict[str, Project] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_assignments: Dict[str, List[str]] = {}  # user_id -> task_ids
        self.project_members: Dict[str, Set[str]] = {}  # project_id -> member_ids

    def create_project(self, project: Project) -> str:
        """Create a new project.

        Args:
            project: Project to create

        Returns:
            Project ID
        """
        if project.id in self.projects:
            raise ValueError(f"Project {project.id} already exists")

        self.projects[project.id] = project
        self.project_members[project.id] = {project.principal_investigator}

        logger.info(f"Created project {project.id}: {project.name}")
        return project.id

    def add_project_member(self, project_id: str, member_id: str) -> bool:
        """Add member to project.

        Args:
            project_id: Project ID
            member_id: Member ID

        Returns:
            Success status
        """
        if project_id not in self.projects:
            logger.error(f"Project {project_id} not found")
            return False

        if project_id not in self.project_members:
            self.project_members[project_id] = set()

        self.project_members[project_id].add(member_id)
        return True

    def create_task(self, task: Task) -> str:
        """Create a task.

        Args:
            task: Task to create

        Returns:
            Task ID
        """
        if task.project_id not in self.projects:
            raise ValueError(f"Project {task.project_id} not found")

        self.tasks[task.id] = task

        # Update assignments
        for assignee in task.assignees:
            if assignee not in self.task_assignments:
                self.task_assignments[assignee] = []
            self.task_assignments[assignee].append(task.id)

        return task.id

    def update_task_status(self, task_id: str, status: TaskStatus,
                          actual_hours: Optional[float] = None) -> bool:
        """Update task status.

        Args:
            task_id: Task ID
            status: New status
            actual_hours: Actual hours spent

        Returns:
            Success status
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False

        task = self.tasks[task_id]
        task.status = status

        if actual_hours is not None:
            task.actual_hours = actual_hours

        if status == TaskStatus.COMPLETED:
            task.completed_date = datetime.now()
        elif status == TaskStatus.IN_PROGRESS and task.start_date is None:
            task.start_date = datetime.now()

        return True

    def get_project_progress(self, project_id: str) -> Dict[str, Any]:
        """Get project progress summary.

        Args:
            project_id: Project ID

        Returns:
            Progress summary
        """
        if project_id not in self.projects:
            return {}

        project_tasks = [t for t in self.tasks.values() if t.project_id == project_id]

        if not project_tasks:
            return {'progress': 0, 'tasks_total': 0}

        completed_tasks = sum(1 for t in project_tasks if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in project_tasks if t.status == TaskStatus.IN_PROGRESS)
        blocked = sum(1 for t in project_tasks if t.status == TaskStatus.BLOCKED)

        total_estimated = sum(t.estimated_hours for t in project_tasks)
        total_actual = sum(t.actual_hours for t in project_tasks)

        return {
            'progress': (completed_tasks / len(project_tasks)) * 100 if project_tasks else 0,
            'tasks_total': len(project_tasks),
            'tasks_completed': completed_tasks,
            'tasks_in_progress': in_progress,
            'tasks_blocked': blocked,
            'hours_estimated': total_estimated,
            'hours_actual': total_actual,
            'efficiency': (total_estimated / total_actual) if total_actual > 0 else 1.0
        }

    def get_critical_path(self, project_id: str) -> List[str]:
        """Get critical path of tasks.

        Args:
            project_id: Project ID

        Returns:
            List of task IDs in critical path
        """
        project_tasks = [t for t in self.tasks.values() if t.project_id == project_id]

        if not project_tasks:
            return []

        # Build dependency graph
        graph = {}
        for task in project_tasks:
            graph[task.id] = task.dependencies

        # Topological sort to find critical path
        critical_path = self._topological_sort(graph)
        return critical_path

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on dependency graph."""
        visited = set()
        stack = []

        def dfs(node):
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return stack[::-1]

    def generate_gantt_chart_data(self, project_id: str) -> List[Dict[str, Any]]:
        """Generate data for Gantt chart visualization.

        Args:
            project_id: Project ID

        Returns:
            Gantt chart data
        """
        project_tasks = [t for t in self.tasks.values() if t.project_id == project_id]

        gantt_data = []
        for task in project_tasks:
            gantt_data.append({
                'id': task.id,
                'name': task.name,
                'start': task.start_date.isoformat() if task.start_date else None,
                'end': task.due_date.isoformat() if task.due_date else None,
                'progress': self._calculate_task_progress(task),
                'dependencies': task.dependencies,
                'assignees': task.assignees,
                'status': task.status.value
            })

        return gantt_data

    def _calculate_task_progress(self, task: Task) -> float:
        """Calculate task progress percentage."""
        if task.status == TaskStatus.COMPLETED:
            return 100.0
        elif task.status == TaskStatus.IN_PROGRESS:
            if task.estimated_hours > 0:
                return min(100.0, (task.actual_hours / task.estimated_hours) * 100)
            return 50.0
        else:
            return 0.0


class ContributionTracker:
    """Track research contributions."""

    def __init__(self):
        """Initialize contribution tracker."""
        self.contributions: Dict[str, Contribution] = {}
        self.contributor_scores: Dict[str, float] = {}
        self.project_contributions: Dict[str, List[str]] = {}

    def record_contribution(self, contribution: Contribution) -> str:
        """Record a research contribution.

        Args:
            contribution: Contribution to record

        Returns:
            Contribution ID
        """
        self.contributions[contribution.id] = contribution

        # Update project contributions
        if contribution.project_id not in self.project_contributions:
            self.project_contributions[contribution.project_id] = []
        self.project_contributions[contribution.project_id].append(contribution.id)

        # Update contributor score
        self._update_contributor_score(contribution.contributor_id)

        return contribution.id

    def validate_contribution(self, contribution_id: str, validator_id: str) -> bool:
        """Validate a contribution by peer.

        Args:
            contribution_id: Contribution ID
            validator_id: Validator ID

        Returns:
            Success status
        """
        if contribution_id not in self.contributions:
            logger.error(f"Contribution {contribution_id} not found")
            return False

        contribution = self.contributions[contribution_id]

        if validator_id == contribution.contributor_id:
            logger.warning("Cannot self-validate contribution")
            return False

        contribution.peer_validations.append(validator_id)

        # Update impact score based on validations
        validation_factor = min(1.0, len(contribution.peer_validations) * 0.2)
        contribution.impact_score = min(1.0, contribution.impact_score + validation_factor * 0.1)

        return True

    def calculate_contribution_scores(self, project_id: str) -> Dict[str, float]:
        """Calculate contribution scores for a project.

        Args:
            project_id: Project ID

        Returns:
            Contributor scores
        """
        if project_id not in self.project_contributions:
            return {}

        scores = {}
        contribution_ids = self.project_contributions[project_id]

        for contrib_id in contribution_ids:
            contribution = self.contributions[contrib_id]
            contributor_id = contribution.contributor_id

            if contributor_id not in scores:
                scores[contributor_id] = 0.0

            # Weight by contribution type importance
            type_weight = self._get_contribution_type_weight(contribution.contribution_type)

            # Calculate score
            score = (
                type_weight *
                contribution.impact_score *
                (1 + len(contribution.peer_validations) * 0.1) *
                (1 + contribution.hours_spent * 0.001)
            )

            scores[contributor_id] += score

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for contributor_id in scores:
                scores[contributor_id] /= total_score

        return scores

    def _get_contribution_type_weight(self, contribution_type: ContributionType) -> float:
        """Get weight for contribution type."""
        weights = {
            ContributionType.CONCEPTUALIZATION: 1.0,
            ContributionType.METHODOLOGY: 0.9,
            ContributionType.SOFTWARE: 0.8,
            ContributionType.VALIDATION: 0.7,
            ContributionType.FORMAL_ANALYSIS: 0.8,
            ContributionType.INVESTIGATION: 0.85,
            ContributionType.RESOURCES: 0.6,
            ContributionType.DATA_CURATION: 0.7,
            ContributionType.WRITING_ORIGINAL: 0.9,
            ContributionType.WRITING_REVIEW: 0.7,
            ContributionType.VISUALIZATION: 0.6,
            ContributionType.SUPERVISION: 0.8,
            ContributionType.PROJECT_ADMIN: 0.6,
            ContributionType.FUNDING: 0.7
        }
        return weights.get(contribution_type, 0.5)

    def _update_contributor_score(self, contributor_id: str):
        """Update overall contributor score."""
        contributions = [c for c in self.contributions.values()
                        if c.contributor_id == contributor_id]

        if not contributions:
            self.contributor_scores[contributor_id] = 0.0
            return

        # Calculate weighted average of impact scores
        total_weight = 0
        weighted_sum = 0

        for contribution in contributions:
            weight = self._get_contribution_type_weight(contribution.contribution_type)
            weighted_sum += contribution.impact_score * weight
            total_weight += weight

        if total_weight > 0:
            self.contributor_scores[contributor_id] = weighted_sum / total_weight
        else:
            self.contributor_scores[contributor_id] = 0.0

    def generate_credit_statement(self, project_id: str) -> str:
        """Generate CRediT authorship statement.

        Args:
            project_id: Project ID

        Returns:
            CRediT statement
        """
        if project_id not in self.project_contributions:
            return "No contributions recorded for this project."

        # Group contributions by contributor and type
        contributor_roles = {}
        contribution_ids = self.project_contributions[project_id]

        for contrib_id in contribution_ids:
            contribution = self.contributions[contrib_id]
            contributor_id = contribution.contributor_id

            if contributor_id not in contributor_roles:
                contributor_roles[contributor_id] = set()

            contributor_roles[contributor_id].add(contribution.contribution_type)

        # Generate statement
        statement = "Author Contributions (CRediT):\n\n"

        for contributor_id, roles in contributor_roles.items():
            role_names = [role.value.replace('_', ' ').title() for role in roles]
            statement += f"{contributor_id}: {', '.join(role_names)}\n"

        return statement


class AuthorshipDeterminer:
    """Determine authorship order and eligibility."""

    def __init__(self, contribution_tracker: ContributionTracker):
        """Initialize authorship determiner.

        Args:
            contribution_tracker: Contribution tracker instance
        """
        self.contribution_tracker = contribution_tracker
        self.authorship_criteria = self._initialize_criteria()

    def _initialize_criteria(self) -> Dict[str, float]:
        """Initialize authorship criteria weights."""
        return {
            'conceptualization': 0.15,
            'methodology': 0.15,
            'investigation': 0.15,
            'writing_original': 0.20,
            'writing_review': 0.10,
            'formal_analysis': 0.10,
            'supervision': 0.10,
            'funding': 0.05
        }

    def determine_authorship(self, project_id: str,
                            custom_weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """Determine authorship order based on contributions.

        Args:
            project_id: Project ID
            custom_weights: Optional custom contribution weights

        Returns:
            List of (contributor_id, score) tuples in authorship order
        """
        scores = self.contribution_tracker.calculate_contribution_scores(project_id)

        if not scores:
            return []

        # Apply custom weights if provided
        if custom_weights:
            weighted_scores = {}
            for contributor_id in scores:
                contributions = [
                    c for c in self.contribution_tracker.contributions.values()
                    if c.contributor_id == contributor_id and c.project_id == project_id
                ]

                weighted_score = 0
                for contribution in contributions:
                    weight = custom_weights.get(contribution.contribution_type.value, 0.5)
                    weighted_score += contribution.impact_score * weight

                weighted_scores[contributor_id] = weighted_score

            scores = weighted_scores

        # Sort by score
        authorship_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return authorship_order

    def check_authorship_eligibility(self, contributor_id: str,
                                    project_id: str,
                                    min_contribution: float = 0.05) -> bool:
        """Check if contributor meets authorship criteria.

        Args:
            contributor_id: Contributor ID
            project_id: Project ID
            min_contribution: Minimum contribution threshold

        Returns:
            Eligibility status
        """
        scores = self.contribution_tracker.calculate_contribution_scores(project_id)
        contributor_score = scores.get(contributor_id, 0)

        return contributor_score >= min_contribution

    def resolve_authorship_dispute(self, project_id: str,
                                  disputed_positions: List[str]) -> Dict[str, Any]:
        """Resolve authorship disputes using objective metrics.

        Args:
            project_id: Project ID
            disputed_positions: List of contributor IDs in dispute

        Returns:
            Resolution with detailed metrics
        """
        detailed_metrics = {}

        for contributor_id in disputed_positions:
            contributions = [
                c for c in self.contribution_tracker.contributions.values()
                if c.contributor_id == contributor_id and c.project_id == project_id
            ]

            metrics = {
                'total_contributions': len(contributions),
                'total_hours': sum(c.hours_spent for c in contributions),
                'peer_validations': sum(len(c.peer_validations) for c in contributions),
                'contribution_types': list(set(c.contribution_type.value for c in contributions)),
                'average_impact': np.mean([c.impact_score for c in contributions]) if contributions else 0
            }

            detailed_metrics[contributor_id] = metrics

        # Calculate composite scores
        composite_scores = {}
        for contributor_id, metrics in detailed_metrics.items():
            score = (
                metrics['average_impact'] * 0.4 +
                min(1.0, metrics['total_hours'] / 100) * 0.2 +
                min(1.0, metrics['peer_validations'] / 10) * 0.2 +
                min(1.0, metrics['total_contributions'] / 20) * 0.1 +
                min(1.0, len(metrics['contribution_types']) / 5) * 0.1
            )
            composite_scores[contributor_id] = score

        # Sort by composite score
        resolved_order = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'resolved_order': resolved_order,
            'detailed_metrics': detailed_metrics,
            'composite_scores': composite_scores
        }


class MilestoneTracker:
    """Track project milestones."""

    def __init__(self):
        """Initialize milestone tracker."""
        self.milestones: Dict[str, Milestone] = {}
        self.project_milestones: Dict[str, List[str]] = {}

    def create_milestone(self, milestone: Milestone) -> str:
        """Create a milestone.

        Args:
            milestone: Milestone to create

        Returns:
            Milestone ID
        """
        self.milestones[milestone.id] = milestone

        if milestone.project_id not in self.project_milestones:
            self.project_milestones[milestone.project_id] = []
        self.project_milestones[milestone.project_id].append(milestone.id)

        return milestone.id

    def update_milestone_progress(self, milestone_id: str,
                                 progress: float,
                                 completed_criteria: Optional[List[str]] = None) -> bool:
        """Update milestone progress.

        Args:
            milestone_id: Milestone ID
            progress: Progress percentage (0-100)
            completed_criteria: List of completed criteria

        Returns:
            Success status
        """
        if milestone_id not in self.milestones:
            logger.error(f"Milestone {milestone_id} not found")
            return False

        milestone = self.milestones[milestone_id]
        milestone.progress_percentage = min(100.0, max(0.0, progress))

        if progress >= 100.0:
            milestone.status = "completed"
            milestone.completed_date = datetime.now()
        elif progress > 0:
            milestone.status = "in_progress"

        return True

    def check_milestone_dependencies(self, milestone_id: str) -> Tuple[bool, List[str]]:
        """Check if milestone dependencies are met.

        Args:
            milestone_id: Milestone ID

        Returns:
            Tuple of (dependencies_met, blocking_milestones)
        """
        if milestone_id not in self.milestones:
            return False, []

        milestone = self.milestones[milestone_id]
        blocking = []

        for dep_id in milestone.dependencies:
            if dep_id in self.milestones:
                dep = self.milestones[dep_id]
                if dep.status != "completed":
                    blocking.append(dep_id)

        return len(blocking) == 0, blocking

    def get_project_milestone_status(self, project_id: str) -> Dict[str, Any]:
        """Get milestone status for project.

        Args:
            project_id: Project ID

        Returns:
            Milestone status summary
        """
        if project_id not in self.project_milestones:
            return {}

        milestone_ids = self.project_milestones[project_id]
        milestones = [self.milestones[mid] for mid in milestone_ids]

        completed = sum(1 for m in milestones if m.status == "completed")
        overdue = sum(1 for m in milestones
                     if m.status != "completed" and m.target_date < datetime.now())

        return {
            'total_milestones': len(milestones),
            'completed': completed,
            'in_progress': sum(1 for m in milestones if m.status == "in_progress"),
            'pending': sum(1 for m in milestones if m.status == "pending"),
            'overdue': overdue,
            'completion_rate': (completed / len(milestones)) * 100 if milestones else 0,
            'next_milestone': self._get_next_milestone(milestones)
        }

    def _get_next_milestone(self, milestones: List[Milestone]) -> Optional[Dict[str, Any]]:
        """Get next upcoming milestone."""
        pending = [m for m in milestones if m.status != "completed"]
        if not pending:
            return None

        pending.sort(key=lambda m: m.target_date)
        next_milestone = pending[0]

        return {
            'id': next_milestone.id,
            'name': next_milestone.name,
            'target_date': next_milestone.target_date.isoformat(),
            'days_remaining': (next_milestone.target_date - datetime.now()).days
        }


class ResourceAllocator:
    """Allocate and track research resources."""

    def __init__(self):
        """Initialize resource allocator."""
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_inventory: Dict[str, Dict[str, Any]] = {}
        self.project_allocations: Dict[str, List[str]] = {}

    def register_resource(self, resource_id: str, resource_type: ResourceType,
                         total_capacity: float, unit: str,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a resource.

        Args:
            resource_id: Resource ID
            resource_type: Type of resource
            total_capacity: Total capacity
            unit: Unit of measurement
            metadata: Optional metadata

        Returns:
            Success status
        """
        self.resource_inventory[resource_id] = {
            'type': resource_type,
            'total_capacity': total_capacity,
            'available_capacity': total_capacity,
            'unit': unit,
            'metadata': metadata or {}
        }
        return True

    def request_allocation(self, allocation: ResourceAllocation) -> str:
        """Request resource allocation.

        Args:
            allocation: Resource allocation request

        Returns:
            Allocation ID
        """
        # Check availability
        if allocation.resource_id in self.resource_inventory:
            resource = self.resource_inventory[allocation.resource_id]

            if resource['available_capacity'] < allocation.allocated_amount:
                raise ValueError(f"Insufficient capacity for resource {allocation.resource_id}")

        self.allocations[allocation.id] = allocation

        if allocation.project_id not in self.project_allocations:
            self.project_allocations[allocation.project_id] = []
        self.project_allocations[allocation.project_id].append(allocation.id)

        return allocation.id

    def approve_allocation(self, allocation_id: str, approver_id: str) -> bool:
        """Approve resource allocation.

        Args:
            allocation_id: Allocation ID
            approver_id: Approver ID

        Returns:
            Success status
        """
        if allocation_id not in self.allocations:
            logger.error(f"Allocation {allocation_id} not found")
            return False

        allocation = self.allocations[allocation_id]
        allocation.status = "approved"
        allocation.approver_id = approver_id

        # Update resource availability
        if allocation.resource_id in self.resource_inventory:
            resource = self.resource_inventory[allocation.resource_id]
            resource['available_capacity'] -= allocation.allocated_amount

        return True

    def track_usage(self, allocation_id: str, usage_amount: float,
                   timestamp: Optional[datetime] = None) -> bool:
        """Track resource usage.

        Args:
            allocation_id: Allocation ID
            usage_amount: Amount used
            timestamp: Usage timestamp

        Returns:
            Success status
        """
        if allocation_id not in self.allocations:
            logger.error(f"Allocation {allocation_id} not found")
            return False

        allocation = self.allocations[allocation_id]
        timestamp = timestamp or datetime.now()
        timestamp_str = timestamp.isoformat()

        allocation.usage_tracking[timestamp_str] = usage_amount

        return True

    def get_resource_utilization(self, resource_id: str) -> Dict[str, Any]:
        """Get resource utilization metrics.

        Args:
            resource_id: Resource ID

        Returns:
            Utilization metrics
        """
        if resource_id not in self.resource_inventory:
            return {}

        resource = self.resource_inventory[resource_id]
        allocations = [a for a in self.allocations.values()
                      if a.resource_id == resource_id and a.status == "approved"]

        total_allocated = sum(a.allocated_amount for a in allocations)
        total_used = sum(sum(a.usage_tracking.values()) for a in allocations)

        return {
            'resource_id': resource_id,
            'type': resource['type'].value,
            'total_capacity': resource['total_capacity'],
            'allocated': total_allocated,
            'used': total_used,
            'available': resource['available_capacity'],
            'utilization_rate': (total_used / resource['total_capacity']) * 100 if resource['total_capacity'] > 0 else 0,
            'allocation_efficiency': (total_used / total_allocated) * 100 if total_allocated > 0 else 0
        }

    def optimize_allocations(self, project_id: str) -> Dict[str, Any]:
        """Optimize resource allocations for a project.

        Args:
            project_id: Project ID

        Returns:
            Optimization recommendations
        """
        if project_id not in self.project_allocations:
            return {}

        allocation_ids = self.project_allocations[project_id]
        allocations = [self.allocations[aid] for aid in allocation_ids]

        recommendations = []

        for allocation in allocations:
            # Check usage efficiency
            total_allocated = allocation.allocated_amount
            total_used = sum(allocation.usage_tracking.values())

            efficiency = (total_used / total_allocated) if total_allocated > 0 else 0

            if efficiency < 0.5:
                recommendations.append({
                    'allocation_id': allocation.id,
                    'resource_id': allocation.resource_id,
                    'recommendation': 'underutilized',
                    'current_allocation': total_allocated,
                    'actual_usage': total_used,
                    'suggested_allocation': total_used * 1.2  # 20% buffer
                })
            elif efficiency > 0.9:
                recommendations.append({
                    'allocation_id': allocation.id,
                    'resource_id': allocation.resource_id,
                    'recommendation': 'near_capacity',
                    'current_allocation': total_allocated,
                    'actual_usage': total_used,
                    'suggested_allocation': total_allocated * 1.3  # 30% increase
                })

        return {
            'project_id': project_id,
            'total_allocations': len(allocations),
            'recommendations': recommendations,
            'potential_savings': self._calculate_potential_savings(recommendations)
        }

    def _calculate_potential_savings(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate potential resource savings."""
        savings = 0
        for rec in recommendations:
            if rec['recommendation'] == 'underutilized':
                savings += rec['current_allocation'] - rec['suggested_allocation']
        return savings