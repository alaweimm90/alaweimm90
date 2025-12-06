"""
Governance and access control for research consortia.

Implements role-based access control, data sharing agreements, and embargo management.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class Role(Enum):
    """Research consortium roles."""
    PRINCIPAL_INVESTIGATOR = "pi"
    CO_INVESTIGATOR = "co_pi"
    POSTDOC = "postdoc"
    GRADUATE_STUDENT = "grad_student"
    UNDERGRADUATE = "undergrad"
    RESEARCH_ASSISTANT = "research_assistant"
    COLLABORATOR = "collaborator"
    EXTERNAL_ADVISOR = "advisor"
    DATA_MANAGER = "data_manager"
    ADMINISTRATOR = "admin"
    OBSERVER = "observer"


class Permission(Enum):
    """System permissions."""
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    DATA_SHARE = "data:share"

    # Project permissions
    PROJECT_VIEW = "project:view"
    PROJECT_EDIT = "project:edit"
    PROJECT_CREATE = "project:create"
    PROJECT_DELETE = "project:delete"
    PROJECT_ARCHIVE = "project:archive"

    # Member permissions
    MEMBER_INVITE = "member:invite"
    MEMBER_REMOVE = "member:remove"
    MEMBER_EDIT_ROLE = "member:edit_role"

    # Publication permissions
    PUBLICATION_DRAFT = "publication:draft"
    PUBLICATION_REVIEW = "publication:review"
    PUBLICATION_SUBMIT = "publication:submit"
    PUBLICATION_APPROVE = "publication:approve"

    # Grant permissions
    GRANT_VIEW = "grant:view"
    GRANT_EDIT = "grant:edit"
    GRANT_SUBMIT = "grant:submit"
    BUDGET_VIEW = "budget:view"
    BUDGET_EDIT = "budget:edit"

    # Administrative permissions
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_COMPLIANCE = "admin:compliance"


@dataclass
class Institution:
    """Represents a participating institution."""
    id: str
    name: str
    country: str
    irb_number: Optional[str] = None
    data_governance_policy: Optional[str] = None
    contact_email: str = ""
    sso_endpoint: Optional[str] = None
    compliance_certifications: List[str] = field(default_factory=list)


@dataclass
class Member:
    """Represents a consortium member."""
    id: str
    name: str
    email: str
    institution_id: str
    role: Role
    orcid: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    join_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    contributions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAsset:
    """Represents a data asset in the consortium."""
    id: str
    name: str
    description: str
    owner_institution: str
    data_type: str  # 'clinical', 'genomic', 'imaging', 'behavioral', etc.
    size_bytes: int
    created_date: datetime
    modified_date: datetime
    embargo_until: Optional[datetime] = None
    access_restrictions: List[str] = field(default_factory=list)
    usage_terms: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Audit log entry."""
    id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    success: bool = True


class ConsortiumGovernance:
    """Main governance system for research consortia."""

    def __init__(self, consortium_id: str, name: str):
        """Initialize consortium governance.

        Args:
            consortium_id: Unique consortium identifier
            name: Consortium name
        """
        self.consortium_id = consortium_id
        self.name = name
        self.institutions: Dict[str, Institution] = {}
        self.members: Dict[str, Member] = {}
        self.data_assets: Dict[str, DataAsset] = {}
        self.audit_logs: List[AuditLog] = []
        self.agreements: Dict[str, 'DataSharingAgreement'] = {}
        self.embargoes: Dict[str, 'EmbargoPolicy'] = {}

        # Initialize subsystems
        self.role_manager = RoleManager()
        self.access_controller = AccessController(self)
        self.embargo_manager = EmbargoManager(self)

    def add_institution(self, institution: Institution) -> bool:
        """Add an institution to the consortium.

        Args:
            institution: Institution to add

        Returns:
            Success status
        """
        if institution.id in self.institutions:
            logger.warning(f"Institution {institution.id} already exists")
            return False

        self.institutions[institution.id] = institution

        self._audit_log(
            user_id="system",
            action="add_institution",
            resource_type="institution",
            resource_id=institution.id,
            details={"name": institution.name, "country": institution.country}
        )

        return True

    def add_member(self, member: Member) -> bool:
        """Add a member to the consortium.

        Args:
            member: Member to add

        Returns:
            Success status
        """
        if member.id in self.members:
            logger.warning(f"Member {member.id} already exists")
            return False

        if member.institution_id not in self.institutions:
            logger.error(f"Institution {member.institution_id} not found")
            return False

        # Assign role-based permissions
        member.permissions = self.role_manager.get_permissions_for_role(member.role)
        self.members[member.id] = member

        self._audit_log(
            user_id="system",
            action="add_member",
            resource_type="member",
            resource_id=member.id,
            details={"name": member.name, "role": member.role.value}
        )

        return True

    def register_data_asset(self, asset: DataAsset, owner_id: str) -> bool:
        """Register a data asset with the consortium.

        Args:
            asset: Data asset to register
            owner_id: ID of the member registering the asset

        Returns:
            Success status
        """
        if asset.id in self.data_assets:
            logger.warning(f"Data asset {asset.id} already exists")
            return False

        if asset.owner_institution not in self.institutions:
            logger.error(f"Institution {asset.owner_institution} not found")
            return False

        self.data_assets[asset.id] = asset

        self._audit_log(
            user_id=owner_id,
            action="register_data_asset",
            resource_type="data_asset",
            resource_id=asset.id,
            details={
                "name": asset.name,
                "type": asset.data_type,
                "size": asset.size_bytes
            }
        )

        return True

    def create_data_sharing_agreement(self, agreement_id: str,
                                     parties: List[str],
                                     terms: Dict[str, Any]) -> 'DataSharingAgreement':
        """Create a data sharing agreement.

        Args:
            agreement_id: Unique agreement ID
            parties: List of institution IDs
            terms: Agreement terms

        Returns:
            Created agreement
        """
        agreement = DataSharingAgreement(
            agreement_id=agreement_id,
            parties=parties,
            terms=terms,
            created_date=datetime.now()
        )

        self.agreements[agreement_id] = agreement

        self._audit_log(
            user_id="system",
            action="create_agreement",
            resource_type="agreement",
            resource_id=agreement_id,
            details={"parties": parties, "terms_hash": agreement.terms_hash}
        )

        return agreement

    def check_access(self, user_id: str, resource_id: str,
                    permission: Permission) -> bool:
        """Check if user has access to resource.

        Args:
            user_id: User ID
            resource_id: Resource ID
            permission: Required permission

        Returns:
            Access granted status
        """
        return self.access_controller.check_access(user_id, resource_id, permission)

    def _audit_log(self, user_id: str, action: str, resource_type: str,
                  resource_id: str, details: Dict[str, Any],
                  success: bool = True):
        """Create audit log entry."""
        log_entry = AuditLog(
            id=str(uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=success
        )
        self.audit_logs.append(log_entry)

    def get_audit_trail(self, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       user_id: Optional[str] = None) -> List[AuditLog]:
        """Get audit trail with filters.

        Args:
            start_date: Start date filter
            end_date: End date filter
            user_id: User ID filter

        Returns:
            Filtered audit logs
        """
        logs = self.audit_logs

        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]

        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]

        if user_id:
            logs = [l for l in logs if l.user_id == user_id]

        return logs

    def export_governance_report(self) -> Dict[str, Any]:
        """Export comprehensive governance report.

        Returns:
            Governance report
        """
        return {
            'consortium_id': self.consortium_id,
            'name': self.name,
            'institutions': len(self.institutions),
            'members': len(self.members),
            'data_assets': len(self.data_assets),
            'active_agreements': len(self.agreements),
            'member_distribution': self._get_member_distribution(),
            'data_summary': self._get_data_summary(),
            'compliance_status': self._check_compliance_status()
        }

    def _get_member_distribution(self) -> Dict[str, int]:
        """Get member distribution by role."""
        distribution = {}
        for member in self.members.values():
            role_name = member.role.value
            distribution[role_name] = distribution.get(role_name, 0) + 1
        return distribution

    def _get_data_summary(self) -> Dict[str, Any]:
        """Get data assets summary."""
        total_size = sum(asset.size_bytes for asset in self.data_assets.values())
        types = {}
        for asset in self.data_assets.values():
            types[asset.data_type] = types.get(asset.data_type, 0) + 1

        return {
            'total_assets': len(self.data_assets),
            'total_size_gb': total_size / (1024**3),
            'asset_types': types,
            'embargoed_assets': sum(1 for a in self.data_assets.values() if a.embargo_until)
        }

    def _check_compliance_status(self) -> Dict[str, Any]:
        """Check overall compliance status."""
        irb_compliance = all(
            inst.irb_number for inst in self.institutions.values()
        )

        data_governance = all(
            inst.data_governance_policy for inst in self.institutions.values()
        )

        return {
            'irb_compliance': irb_compliance,
            'data_governance': data_governance,
            'gdpr_ready': self._check_gdpr_compliance(),
            'hipaa_ready': self._check_hipaa_compliance()
        }

    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance."""
        # Simplified check
        eu_countries = {'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'PL', 'SE', 'AT', 'DK'}
        has_eu = any(inst.country in eu_countries for inst in self.institutions.values())

        if has_eu:
            # Check for required GDPR measures
            has_audit = len(self.audit_logs) > 0
            has_agreements = len(self.agreements) > 0
            return has_audit and has_agreements

        return True

    def _check_hipaa_compliance(self) -> bool:
        """Check HIPAA compliance."""
        # Check if handling healthcare data
        has_clinical = any(
            asset.data_type == 'clinical'
            for asset in self.data_assets.values()
        )

        if has_clinical:
            # Check for required HIPAA measures
            has_encryption = True  # Assumed
            has_access_control = len(self.members) > 0
            has_audit = len(self.audit_logs) > 0
            return has_encryption and has_access_control and has_audit

        return True


class RoleManager:
    """Manage roles and permissions."""

    def __init__(self):
        """Initialize role manager."""
        self.role_permissions = self._initialize_role_permissions()
        self.custom_roles: Dict[str, Set[Permission]] = {}

    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role permissions."""
        return {
            Role.PRINCIPAL_INVESTIGATOR: {
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_DELETE,
                Permission.DATA_EXPORT, Permission.DATA_SHARE,
                Permission.PROJECT_VIEW, Permission.PROJECT_EDIT, Permission.PROJECT_CREATE,
                Permission.PROJECT_DELETE, Permission.PROJECT_ARCHIVE,
                Permission.MEMBER_INVITE, Permission.MEMBER_REMOVE, Permission.MEMBER_EDIT_ROLE,
                Permission.PUBLICATION_DRAFT, Permission.PUBLICATION_REVIEW,
                Permission.PUBLICATION_SUBMIT, Permission.PUBLICATION_APPROVE,
                Permission.GRANT_VIEW, Permission.GRANT_EDIT, Permission.GRANT_SUBMIT,
                Permission.BUDGET_VIEW, Permission.BUDGET_EDIT,
                Permission.ADMIN_SETTINGS, Permission.ADMIN_AUDIT, Permission.ADMIN_COMPLIANCE
            },
            Role.CO_INVESTIGATOR: {
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT,
                Permission.DATA_SHARE,
                Permission.PROJECT_VIEW, Permission.PROJECT_EDIT, Permission.PROJECT_CREATE,
                Permission.MEMBER_INVITE,
                Permission.PUBLICATION_DRAFT, Permission.PUBLICATION_REVIEW,
                Permission.PUBLICATION_SUBMIT,
                Permission.GRANT_VIEW, Permission.GRANT_EDIT,
                Permission.BUDGET_VIEW
            },
            Role.POSTDOC: {
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT,
                Permission.PROJECT_VIEW, Permission.PROJECT_EDIT,
                Permission.PUBLICATION_DRAFT, Permission.PUBLICATION_REVIEW,
                Permission.GRANT_VIEW
            },
            Role.GRADUATE_STUDENT: {
                Permission.DATA_READ, Permission.DATA_WRITE,
                Permission.PROJECT_VIEW, Permission.PROJECT_EDIT,
                Permission.PUBLICATION_DRAFT
            },
            Role.UNDERGRADUATE: {
                Permission.DATA_READ,
                Permission.PROJECT_VIEW
            },
            Role.RESEARCH_ASSISTANT: {
                Permission.DATA_READ, Permission.DATA_WRITE,
                Permission.PROJECT_VIEW
            },
            Role.COLLABORATOR: {
                Permission.DATA_READ, Permission.DATA_EXPORT,
                Permission.PROJECT_VIEW,
                Permission.PUBLICATION_DRAFT, Permission.PUBLICATION_REVIEW
            },
            Role.EXTERNAL_ADVISOR: {
                Permission.PROJECT_VIEW,
                Permission.PUBLICATION_REVIEW,
                Permission.GRANT_VIEW
            },
            Role.DATA_MANAGER: {
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_DELETE,
                Permission.DATA_EXPORT, Permission.DATA_SHARE,
                Permission.PROJECT_VIEW,
                Permission.ADMIN_AUDIT, Permission.ADMIN_COMPLIANCE
            },
            Role.ADMINISTRATOR: {
                Permission.ADMIN_SETTINGS, Permission.ADMIN_AUDIT, Permission.ADMIN_COMPLIANCE,
                Permission.MEMBER_INVITE, Permission.MEMBER_REMOVE, Permission.MEMBER_EDIT_ROLE
            },
            Role.OBSERVER: {
                Permission.PROJECT_VIEW
            }
        }

    def get_permissions_for_role(self, role: Role) -> Set[Permission]:
        """Get permissions for a role.

        Args:
            role: Role

        Returns:
            Set of permissions
        """
        return self.role_permissions.get(role, set())

    def create_custom_role(self, role_name: str,
                          permissions: Set[Permission]) -> bool:
        """Create a custom role.

        Args:
            role_name: Custom role name
            permissions: Set of permissions

        Returns:
            Success status
        """
        if role_name in self.custom_roles:
            logger.warning(f"Custom role {role_name} already exists")
            return False

        self.custom_roles[role_name] = permissions
        return True

    def check_permission(self, role: Role, permission: Permission) -> bool:
        """Check if role has permission.

        Args:
            role: Role to check
            permission: Required permission

        Returns:
            Has permission status
        """
        return permission in self.get_permissions_for_role(role)


class AccessController:
    """Control access to consortium resources."""

    def __init__(self, governance: ConsortiumGovernance):
        """Initialize access controller.

        Args:
            governance: Consortium governance instance
        """
        self.governance = governance
        self.access_cache: Dict[str, Dict[str, bool]] = {}
        self.resource_owners: Dict[str, str] = {}

    def check_access(self, user_id: str, resource_id: str,
                    permission: Permission) -> bool:
        """Check if user has access to resource.

        Args:
            user_id: User ID
            resource_id: Resource ID
            permission: Required permission

        Returns:
            Access granted status
        """
        # Check cache
        cache_key = f"{user_id}:{resource_id}:{permission.value}"
        if cache_key in self.access_cache:
            return self.access_cache[cache_key]

        # Get user
        if user_id not in self.governance.members:
            result = False
        else:
            member = self.governance.members[user_id]

            # Check if user has permission
            if permission in member.permissions:
                result = True
            else:
                # Check resource-specific rules
                result = self._check_resource_specific_access(
                    member, resource_id, permission
                )

        # Cache result
        self.access_cache[cache_key] = result

        # Audit access check
        self.governance._audit_log(
            user_id=user_id,
            action="access_check",
            resource_type="resource",
            resource_id=resource_id,
            details={"permission": permission.value, "granted": result}
        )

        return result

    def _check_resource_specific_access(self, member: Member,
                                       resource_id: str,
                                       permission: Permission) -> bool:
        """Check resource-specific access rules.

        Args:
            member: Member requesting access
            resource_id: Resource ID
            permission: Required permission

        Returns:
            Access granted status
        """
        # Check if resource is a data asset
        if resource_id in self.governance.data_assets:
            asset = self.governance.data_assets[resource_id]

            # Check institution match
            if asset.owner_institution == member.institution_id:
                return True

            # Check embargo
            if asset.embargo_until and datetime.now() < asset.embargo_until:
                return False

            # Check data sharing agreements
            for agreement in self.governance.agreements.values():
                if member.institution_id in agreement.parties:
                    if agreement.is_active() and agreement.covers_asset(resource_id):
                        return True

        return False

    def grant_access(self, user_id: str, resource_id: str,
                    permissions: Set[Permission], duration: Optional[timedelta] = None):
        """Grant temporary access to a resource.

        Args:
            user_id: User ID
            resource_id: Resource ID
            permissions: Permissions to grant
            duration: Optional duration for temporary access
        """
        # Implementation would handle temporary access grants
        pass

    def revoke_access(self, user_id: str, resource_id: str,
                     permissions: Optional[Set[Permission]] = None):
        """Revoke access to a resource.

        Args:
            user_id: User ID
            resource_id: Resource ID
            permissions: Specific permissions to revoke (all if None)
        """
        # Clear cache
        keys_to_remove = []
        for key in self.access_cache:
            if key.startswith(f"{user_id}:{resource_id}:"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.access_cache[key]


class DataSharingAgreement:
    """Data sharing agreement between institutions."""

    def __init__(self, agreement_id: str, parties: List[str],
                 terms: Dict[str, Any], created_date: datetime):
        """Initialize data sharing agreement.

        Args:
            agreement_id: Unique agreement ID
            parties: List of institution IDs
            terms: Agreement terms
            created_date: Creation date
        """
        self.agreement_id = agreement_id
        self.parties = parties
        self.terms = terms
        self.created_date = created_date
        self.effective_date = terms.get('effective_date', created_date)
        self.expiration_date = terms.get('expiration_date')
        self.covered_data_types = terms.get('data_types', [])
        self.usage_restrictions = terms.get('restrictions', [])
        self.terms_hash = self._calculate_terms_hash()
        self.signatures: Dict[str, Dict[str, Any]] = {}

    def _calculate_terms_hash(self) -> str:
        """Calculate hash of agreement terms."""
        terms_str = json.dumps(self.terms, sort_keys=True)
        return hashlib.sha256(terms_str.encode()).hexdigest()

    def sign(self, institution_id: str, signatory_id: str,
            signature_data: Dict[str, Any]) -> bool:
        """Sign the agreement.

        Args:
            institution_id: Institution ID
            signatory_id: Signatory member ID
            signature_data: Signature data

        Returns:
            Success status
        """
        if institution_id not in self.parties:
            logger.error(f"Institution {institution_id} not party to agreement")
            return False

        self.signatures[institution_id] = {
            'signatory_id': signatory_id,
            'signature_date': datetime.now(),
            'signature_data': signature_data
        }

        return True

    def is_active(self) -> bool:
        """Check if agreement is active."""
        now = datetime.now()

        if now < self.effective_date:
            return False

        if self.expiration_date and now > self.expiration_date:
            return False

        # Check if all parties have signed
        return len(self.signatures) == len(self.parties)

    def covers_asset(self, asset_id: str) -> bool:
        """Check if agreement covers a data asset.

        Args:
            asset_id: Data asset ID

        Returns:
            Coverage status
        """
        # Simplified check - would need asset metadata
        return True

    def renew(self, extension_period: timedelta) -> bool:
        """Renew the agreement.

        Args:
            extension_period: Extension period

        Returns:
            Success status
        """
        if not self.expiration_date:
            logger.warning("Agreement has no expiration date")
            return False

        self.expiration_date += extension_period
        return True


class EmbargoManager:
    """Manage data embargoes."""

    def __init__(self, governance: ConsortiumGovernance):
        """Initialize embargo manager.

        Args:
            governance: Consortium governance instance
        """
        self.governance = governance
        self.embargo_policies: Dict[str, 'EmbargoPolicy'] = {}

    def create_embargo(self, asset_id: str, embargo_until: datetime,
                      reason: str, exceptions: Optional[List[str]] = None) -> 'EmbargoPolicy':
        """Create an embargo on a data asset.

        Args:
            asset_id: Data asset ID
            embargo_until: Embargo end date
            reason: Reason for embargo
            exceptions: List of excepted institution IDs

        Returns:
            Created embargo policy
        """
        if asset_id not in self.governance.data_assets:
            raise ValueError(f"Data asset {asset_id} not found")

        policy = EmbargoPolicy(
            asset_id=asset_id,
            embargo_until=embargo_until,
            reason=reason,
            exceptions=exceptions or []
        )

        self.embargo_policies[asset_id] = policy

        # Update asset
        self.governance.data_assets[asset_id].embargo_until = embargo_until

        self.governance._audit_log(
            user_id="system",
            action="create_embargo",
            resource_type="data_asset",
            resource_id=asset_id,
            details={"until": embargo_until.isoformat(), "reason": reason}
        )

        return policy

    def check_embargo(self, asset_id: str, institution_id: str) -> Tuple[bool, Optional[str]]:
        """Check if asset is embargoed for institution.

        Args:
            asset_id: Data asset ID
            institution_id: Institution ID

        Returns:
            Tuple of (is_embargoed, reason)
        """
        if asset_id not in self.embargo_policies:
            return False, None

        policy = self.embargo_policies[asset_id]

        if datetime.now() >= policy.embargo_until:
            # Embargo expired
            return False, None

        if institution_id in policy.exceptions:
            # Institution is excepted
            return False, None

        return True, policy.reason

    def lift_embargo(self, asset_id: str) -> bool:
        """Lift embargo on asset.

        Args:
            asset_id: Data asset ID

        Returns:
            Success status
        """
        if asset_id not in self.embargo_policies:
            logger.warning(f"No embargo found for asset {asset_id}")
            return False

        del self.embargo_policies[asset_id]

        if asset_id in self.governance.data_assets:
            self.governance.data_assets[asset_id].embargo_until = None

        self.governance._audit_log(
            user_id="system",
            action="lift_embargo",
            resource_type="data_asset",
            resource_id=asset_id,
            details={"timestamp": datetime.now().isoformat()}
        )

        return True


@dataclass
class EmbargoPolicy:
    """Embargo policy for a data asset."""
    asset_id: str
    embargo_until: datetime
    reason: str
    exceptions: List[str]  # Institution IDs excepted from embargo
    created_date: datetime = field(default_factory=datetime.now)