"""
Research Consortia Management System

Multi-institution research project management with role-based access control,
data governance, contribution tracking, and collaboration tools.
"""

from .governance import (
    ConsortiumGovernance,
    RoleManager,
    AccessController,
    DataSharingAgreement,
    EmbargoManager,
)
from .collaboration import (
    ProjectManager,
    ContributionTracker,
    AuthorshipDeterminer,
    MilestoneTracker,
    ResourceAllocator,
)
from .intellectual_property import (
    IPManager,
    PatentWorkflow,
    LicenseManager,
    PublicationRights,
)
from .grants import (
    GrantManager,
    BudgetTracker,
    CostAllocator,
    ReportingEngine,
)
from .workflow import (
    PublicationWorkflow,
    PreRegistration,
    DataArchiver,
    IRBIntegration,
)
from .communication import (
    MeetingScheduler,
    CollaborationHub,
    NotificationSystem,
    DocumentSharing,
)

__version__ = "1.0.0"
__all__ = [
    "ConsortiumGovernance",
    "RoleManager",
    "AccessController",
    "DataSharingAgreement",
    "EmbargoManager",
    "ProjectManager",
    "ContributionTracker",
    "AuthorshipDeterminer",
    "MilestoneTracker",
    "ResourceAllocator",
    "IPManager",
    "PatentWorkflow",
    "LicenseManager",
    "PublicationRights",
    "GrantManager",
    "BudgetTracker",
    "CostAllocator",
    "ReportingEngine",
    "PublicationWorkflow",
    "PreRegistration",
    "DataArchiver",
    "IRBIntegration",
    "MeetingScheduler",
    "CollaborationHub",
    "NotificationSystem",
    "DocumentSharing",
]