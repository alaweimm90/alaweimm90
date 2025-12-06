#!/usr/bin/env python3
"""
PII Redaction Pipeline for ORCHEX
Detects and redacts personally identifiable information from inputs and outputs.

Priority [041] implementation with consent management.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PIIType(Enum):
    """Types of PII that can be detected and redacted."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "dob"
    MEDICAL_ID = "medical_id"
    FINANCIAL_ACCOUNT = "financial_account"


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""

    type: PIIType
    pattern: str
    replacement: str
    confidence: float
    requires_context: bool = False


class PIIRedactor:
    """Main PII redaction engine."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize redactor with configuration."""
        self.patterns = self._load_patterns()
        self.consent_registry = {}
        self.redaction_log = []

        if config_path:
            self._load_config(config_path)

    def _load_patterns(self) -> List[PIIPattern]:
        """Load PII detection patterns."""
        return [
            # Email addresses
            PIIPattern(
                PIIType.EMAIL,
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "[EMAIL_REDACTED]",
                0.95,
            ),
            # Phone numbers (US format)
            PIIPattern(
                PIIType.PHONE,
                r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                "[PHONE_REDACTED]",
                0.90,
            ),
            # Social Security Numbers
            PIIPattern(PIIType.SSN, r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b", "[SSN_REDACTED]", 0.95),
            # Credit Card Numbers
            PIIPattern(PIIType.CREDIT_CARD, r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CC_REDACTED]", 0.90),
            # IP Addresses
            PIIPattern(PIIType.IP_ADDRESS, r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]", 0.85),
            # Medical Record Numbers
            PIIPattern(
                PIIType.MEDICAL_ID,
                r"\b(MRN|Medical\sRecord\sNumber)[:\s]*[A-Z0-9]{6,12}\b",
                "[MEDICAL_ID_REDACTED]",
                0.90,
                requires_context=True,
            ),
            # Financial Account Numbers
            PIIPattern(
                PIIType.FINANCIAL_ACCOUNT,
                r"\b(Account|Acct)[:\s#]*\d{8,12}\b",
                "[ACCOUNT_REDACTED]",
                0.85,
                requires_context=True,
            ),
        ]

    def scan(self, text: str) -> List[Tuple[PIIType, str, int, int]]:
        """Scan text for PII and return findings."""
        findings = []

        for pattern in self.patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append((pattern.type, match.group(), match.start(), match.end()))

        return findings

    def redact(self, text: str, consent: Optional[Dict[str, bool]] = None) -> Tuple[str, Dict]:
        """Redact PII from text based on consent."""
        if consent is None:
            consent = {}

        redacted = text
        redaction_map = {}

        for pattern in self.patterns:
            # Check consent for this PII type
            if consent.get(pattern.type.value, False):
                continue  # User consented to keep this PII type

            # Apply redaction
            matches = list(re.finditer(pattern.pattern, redacted, re.IGNORECASE))

            # Process matches in reverse to maintain position accuracy
            for match in reversed(matches):
                original = match.group()
                # Create reversible token
                token = self._create_token(original, pattern.type)
                replacement = f"{pattern.replacement}_{token}"

                redacted = redacted[: match.start()] + replacement + redacted[match.end() :]
                redaction_map[token] = {
                    "type": pattern.type.value,
                    "original_hash": hashlib.sha256(original.encode()).hexdigest(),
                    "position": match.start(),
                    "length": len(original),
                }

        # Log redaction activity
        self.redaction_log.append(
            {
                "timestamp": self._get_timestamp(),
                "redactions": len(redaction_map),
                "consent_provided": consent,
            }
        )

        return redacted, redaction_map

    def _create_token(self, value: str, pii_type: PIIType) -> str:
        """Create a reversible token for redacted content."""
        # Use first 8 chars of SHA256 for uniqueness
        return hashlib.sha256(f"{value}:{pii_type.value}".encode()).hexdigest()[:8]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def validate_consent(self, consent_form: Dict[str, Any]) -> bool:
        """Validate that consent form meets requirements."""
        required_fields = ["user_id", "timestamp", "pii_types", "purpose", "retention_period"]

        for field in required_fields:
            if field not in consent_form:
                return False

        # Validate PII types
        valid_types = {t.value for t in PIIType}
        provided_types = set(consent_form.get("pii_types", {}).keys())

        if not provided_types.issubset(valid_types):
            return False

        return True

    def store_consent(self, consent_form: Dict[str, Any]) -> str:
        """Store consent and return consent ID."""
        if not self.validate_consent(consent_form):
            raise ValueError("Invalid consent form")

        consent_id = hashlib.sha256(json.dumps(consent_form, sort_keys=True).encode()).hexdigest()[
            :16
        ]

        self.consent_registry[consent_id] = consent_form
        return consent_id

    def get_consent(self, consent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve consent by ID."""
        return self.consent_registry.get(consent_id)

    def create_consent_form(
        self, user_id: str, pii_types: Dict[str, bool], purpose: str, retention_days: int = 90
    ) -> Dict[str, Any]:
        """Create a consent form for PII processing."""
        return {
            "user_id": user_id,
            "timestamp": self._get_timestamp(),
            "pii_types": pii_types,
            "purpose": purpose,
            "retention_period": retention_days,
            "version": "1.0.0",
            "jurisdiction": "US",
            "withdrawal_allowed": True,
        }


class PIIPipeline:
    """End-to-end PII handling pipeline."""

    def __init__(self):
        self.redactor = PIIRedactor()
        self.scan_results = {}

    def process_input(
        self, input_data: Dict[str, Any], consent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process input data for PII."""
        consent = {}
        if consent_id:
            consent_form = self.redactor.get_consent(consent_id)
            if consent_form:
                consent = consent_form["pii_types"]

        processed = {}
        redaction_metadata = {}

        for key, value in input_data.items():
            if isinstance(value, str):
                # Scan for PII
                findings = self.redactor.scan(value)
                if findings:
                    # Redact based on consent
                    redacted, metadata = self.redactor.redact(value, consent)
                    processed[key] = redacted
                    if metadata:
                        redaction_metadata[key] = metadata
                else:
                    processed[key] = value
            elif isinstance(value, dict):
                # Recurse for nested structures
                processed[key] = self.process_input(value, consent_id)
            else:
                processed[key] = value

        # Attach metadata if any redactions occurred
        if redaction_metadata:
            processed["_pii_metadata"] = {
                "redacted": True,
                "consent_id": consent_id,
                "redaction_map": redaction_metadata,
                "timestamp": self.redactor._get_timestamp(),
            }

        return processed

    def generate_privacy_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a privacy report for given data."""
        report = {
            "timestamp": self.redactor._get_timestamp(),
            "pii_found": {},
            "risk_score": 0.0,
            "recommendations": [],
        }

        # Scan all text fields
        for key, value in data.items():
            if isinstance(value, str):
                findings = self.redactor.scan(value)
                if findings:
                    report["pii_found"][key] = [
                        {"type": f.value, "count": 1} for f, _, _, _ in findings
                    ]

        # Calculate risk score
        total_pii = sum(len(v) for v in report["pii_found"].values())
        if total_pii > 0:
            high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_ID}
            high_risk_count = sum(
                1
                for items in report["pii_found"].values()
                for item in items
                if item["type"] in {t.value for t in high_risk_types}
            )
            report["risk_score"] = min(1.0, (high_risk_count * 0.3 + total_pii * 0.1))

        # Generate recommendations
        if report["risk_score"] > 0.7:
            report["recommendations"].append("HIGH RISK: Implement immediate redaction")
        elif report["risk_score"] > 0.3:
            report["recommendations"].append("MEDIUM RISK: Review and obtain consent")
        else:
            report["recommendations"].append("LOW RISK: Standard processing acceptable")

        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PIIPipeline()

    # Example data with PII
    test_data = {
        "hypothesis": "Study participant John Doe (email: john.doe@example.com) showed improvement",
        "evidence": "Patient MRN: A123456 reported symptoms on 555-1234",
        "metadata": {"researcher": "Dr. Smith at 192.168.1.1", "account": "Account #12345678"},
    }

    # Create consent form
    consent_form = pipeline.redactor.create_consent_form(
        user_id="researcher_001",
        pii_types={
            "email": False,  # Don't allow email
            "phone": False,  # Don't allow phone
            "name": True,  # Allow names
            "ip_address": False,
        },
        purpose="Research hypothesis validation",
        retention_days=30,
    )

    # Store consent
    consent_id = pipeline.redactor.store_consent(consent_form)

    # Process with redaction
    processed_data = pipeline.process_input(test_data, consent_id)

    # Generate privacy report
    privacy_report = pipeline.generate_privacy_report(test_data)

    # Output results
    print("Original Data:", json.dumps(test_data, indent=2))
    print("\nProcessed Data:", json.dumps(processed_data, indent=2))
    print("\nPrivacy Report:", json.dumps(privacy_report, indent=2))
