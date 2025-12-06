"""
Billing & Subscription Models
Enterprise billing system with usage-based pricing and Stripe integration
"""

import uuid
from enum import Enum
from datetime import datetime, UTC
from typing import Optional, Dict, Any
from decimal import Decimal

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, Text, JSON,
    ForeignKey, Numeric, CheckConstraint, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from database import Base, TimestampMixin

# Enums
class SubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class BillingInterval(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ONE_TIME = "one_time"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"

class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"

class UsageType(str, Enum):
    API_CALLS = "api_calls"
    STORAGE_GB = "storage_gb"
    COMPUTE_HOURS = "compute_hours"
    BANDWIDTH_GB = "bandwidth_gb"
    USERS = "users"
    PROJECTS = "projects"
    CUSTOM = "custom"

# Models
class SubscriptionPlan(Base, TimestampMixin):
    """Subscription plan definitions"""
    __tablename__ = 'subscription_plans'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic info
    name = Column(String(100), nullable=False, unique=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    tier = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # Pricing
    base_price = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default='USD')
    billing_interval = Column(String(20), nullable=False)
    trial_days = Column(Integer, default=0)

    # Stripe integration
    stripe_price_id = Column(String(255), nullable=True, unique=True)
    stripe_product_id = Column(String(255), nullable=True)

    # Features & Limits
    features = Column(JSONB, default=dict)
    limits = Column(JSONB, default={
        "api_calls_per_month": 1000,
        "storage_gb": 10,
        "users": 5,
        "projects": 10,
        "compute_hours": 100,
        "bandwidth_gb": 100,
        "support_level": "email",
        "sla": 99.0
    })

    # Usage-based pricing components
    usage_pricing = Column(JSONB, default={
        "api_calls": {"included": 1000, "per_1000": 10.00},
        "storage_gb": {"included": 10, "per_gb": 0.10},
        "compute_hours": {"included": 100, "per_hour": 0.50},
        "bandwidth_gb": {"included": 100, "per_gb": 0.05}
    })

    # Discounts
    yearly_discount_percent = Column(Numeric(5, 2), default=20.0)
    volume_discounts = Column(JSONB, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)
    is_recommended = Column(Boolean, default=False)

    # Metadata
    metadata = Column(JSONB, default=dict)
    tags = Column(ARRAY(String), default=list)

    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
    addons = relationship("PlanAddon", back_populates="plan")

    __table_args__ = (
        CheckConstraint('base_price >= 0', name='check_positive_price'),
        CheckConstraint('trial_days >= 0', name='check_positive_trial'),
        Index('ix_subscription_plans_tier', 'tier'),
    )

class Subscription(Base, TimestampMixin):
    """Active subscriptions for organizations"""
    __tablename__ = 'subscriptions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    plan_id = Column(UUID(as_uuid=True), ForeignKey('subscription_plans.id'), nullable=False)

    # Stripe integration
    stripe_subscription_id = Column(String(255), nullable=True, unique=True)
    stripe_customer_id = Column(String(255), nullable=True)

    # Billing details
    billing_email = Column(String(255), nullable=False)
    billing_name = Column(String(255), nullable=True)
    billing_address = Column(JSONB, nullable=True)
    tax_id = Column(String(100), nullable=True)

    # Subscription dates
    start_date = Column(DateTime(timezone=True), nullable=False, default=datetime.now(UTC))
    current_period_start = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    trial_end = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    cancellation_reason = Column(Text, nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Status
    status = Column(String(50), nullable=False, default='active')
    is_trial = Column(Boolean, default=False)
    auto_renew = Column(Boolean, default=True)

    # Pricing overrides
    custom_price = Column(Numeric(10, 2), nullable=True)
    discount_percent = Column(Numeric(5, 2), nullable=True)
    discount_amount = Column(Numeric(10, 2), nullable=True)
    credits_balance = Column(Numeric(10, 2), default=0)

    # Usage tracking
    current_usage = Column(JSONB, default=dict)
    usage_updated_at = Column(DateTime(timezone=True), nullable=True)

    # Payment method
    payment_method_id = Column(UUID(as_uuid=True), ForeignKey('payment_methods.id'), nullable=True)
    failed_payment_attempts = Column(Integer, default=0)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    metadata = Column(JSONB, default=dict)
    notes = Column(Text, nullable=True)

    # Relationships
    organization = relationship("Organization", backref="subscriptions")
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")
    payment_method = relationship("PaymentMethod", backref="subscriptions")
    invoices = relationship("Invoice", back_populates="subscription")
    usage_records = relationship("UsageRecord", back_populates="subscription")
    addons = relationship("SubscriptionAddon", back_populates="subscription")

    __table_args__ = (
        Index('ix_subscriptions_organization', 'organization_id'),
        Index('ix_subscriptions_status', 'status'),
        Index('ix_subscriptions_current_period_end', 'current_period_end'),
    )

class PaymentMethod(Base, TimestampMixin):
    """Payment methods for billing"""
    __tablename__ = 'payment_methods'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)

    # Stripe integration
    stripe_payment_method_id = Column(String(255), nullable=True, unique=True)

    # Payment details (encrypted/masked)
    type = Column(String(50), nullable=False)  # card, bank_account, etc.
    card_brand = Column(String(50), nullable=True)  # visa, mastercard, etc.
    card_last4 = Column(String(4), nullable=True)
    card_exp_month = Column(Integer, nullable=True)
    card_exp_year = Column(Integer, nullable=True)
    bank_name = Column(String(100), nullable=True)
    bank_last4 = Column(String(4), nullable=True)

    # Billing details
    billing_email = Column(String(255), nullable=False)
    billing_name = Column(String(255), nullable=True)
    billing_address = Column(JSONB, nullable=True)

    # Status
    is_default = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    verification_status = Column(String(50), nullable=True)

    # Metadata
    metadata = Column(JSONB, default=dict)

    # Relationships
    organization = relationship("Organization", backref="payment_methods")

    __table_args__ = (
        Index('ix_payment_methods_organization', 'organization_id'),
        UniqueConstraint('organization_id', 'is_default', name='uq_one_default_per_org',
                         postgresql_where=Column('is_default') == True),
    )

class Invoice(Base, TimestampMixin):
    """Invoices for subscription billing"""
    __tablename__ = 'invoices'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('subscriptions.id'), nullable=True)

    # Invoice details
    invoice_number = Column(String(50), nullable=False, unique=True, index=True)
    stripe_invoice_id = Column(String(255), nullable=True, unique=True)

    # Amounts (in cents to avoid floating point issues)
    subtotal = Column(Integer, nullable=False)
    tax = Column(Integer, default=0)
    discount = Column(Integer, default=0)
    total = Column(Integer, nullable=False)
    amount_paid = Column(Integer, default=0)
    amount_due = Column(Integer, nullable=False)
    currency = Column(String(3), default='USD')

    # Dates
    invoice_date = Column(DateTime(timezone=True), nullable=False, default=datetime.now(UTC))
    due_date = Column(DateTime(timezone=True), nullable=False)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    period_start = Column(DateTime(timezone=True), nullable=True)
    period_end = Column(DateTime(timezone=True), nullable=True)

    # Status
    status = Column(String(20), nullable=False, default=InvoiceStatus.DRAFT)

    # Payment
    payment_method_id = Column(UUID(as_uuid=True), ForeignKey('payment_methods.id'), nullable=True)
    payment_intent_id = Column(String(255), nullable=True)
    charge_id = Column(String(255), nullable=True)

    # Line items stored as JSON
    line_items = Column(JSONB, nullable=False, default=list)

    # PDF
    pdf_url = Column(String(500), nullable=True)
    hosted_invoice_url = Column(String(500), nullable=True)

    # Metadata
    metadata = Column(JSONB, default=dict)
    notes = Column(Text, nullable=True)

    # Relationships
    organization = relationship("Organization", backref="invoices")
    subscription = relationship("Subscription", back_populates="invoices")
    payment_method = relationship("PaymentMethod", backref="invoices")
    payments = relationship("Payment", back_populates="invoice")

    __table_args__ = (
        Index('ix_invoices_organization', 'organization_id'),
        Index('ix_invoices_status', 'status'),
        Index('ix_invoices_due_date', 'due_date'),
    )

class Payment(Base, TimestampMixin):
    """Payment transactions"""
    __tablename__ = 'payments'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey('invoices.id'), nullable=True)

    # Payment details
    amount = Column(Integer, nullable=False)  # in cents
    currency = Column(String(3), default='USD')
    description = Column(Text, nullable=True)

    # Stripe integration
    stripe_payment_intent_id = Column(String(255), nullable=True, unique=True)
    stripe_charge_id = Column(String(255), nullable=True, unique=True)

    # Status
    status = Column(String(20), nullable=False, default=PaymentStatus.PENDING)
    failure_reason = Column(Text, nullable=True)
    failure_code = Column(String(50), nullable=True)

    # Processing details
    processed_at = Column(DateTime(timezone=True), nullable=True)
    processor_response = Column(JSONB, nullable=True)

    # Refund tracking
    refunded_amount = Column(Integer, default=0)
    refund_reason = Column(Text, nullable=True)
    refunded_at = Column(DateTime(timezone=True), nullable=True)

    # Payment method
    payment_method_id = Column(UUID(as_uuid=True), ForeignKey('payment_methods.id'), nullable=True)
    payment_method_details = Column(JSONB, nullable=True)

    # Metadata
    metadata = Column(JSONB, default=dict)

    # Relationships
    organization = relationship("Organization", backref="payments")
    invoice = relationship("Invoice", back_populates="payments")
    payment_method = relationship("PaymentMethod", backref="payments")

    __table_args__ = (
        Index('ix_payments_organization', 'organization_id'),
        Index('ix_payments_status', 'status'),
        CheckConstraint('amount >= 0', name='check_positive_amount'),
        CheckConstraint('refunded_amount >= 0', name='check_positive_refund'),
        CheckConstraint('refunded_amount <= amount', name='check_refund_not_exceed'),
    )

class UsageRecord(Base, TimestampMixin):
    """Track usage for billing purposes"""
    __tablename__ = 'usage_records'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('subscriptions.id'), nullable=True)

    # Usage details
    usage_type = Column(String(50), nullable=False)
    quantity = Column(Numeric(12, 4), nullable=False)
    unit = Column(String(20), nullable=False)  # calls, bytes, hours, etc.

    # Billing
    unit_price = Column(Numeric(10, 6), nullable=True)
    total_price = Column(Numeric(10, 2), nullable=True)
    is_billable = Column(Boolean, default=True)
    billed = Column(Boolean, default=False)
    billed_at = Column(DateTime(timezone=True), nullable=True)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey('invoices.id'), nullable=True)

    # Time tracking
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=datetime.now(UTC))

    # Source tracking
    source_type = Column(String(50), nullable=True)  # api, web, system, etc.
    source_id = Column(String(255), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)

    # Metadata
    metadata = Column(JSONB, default=dict)
    description = Column(Text, nullable=True)

    # Relationships
    organization = relationship("Organization", backref="usage_records")
    subscription = relationship("Subscription", back_populates="usage_records")
    invoice = relationship("Invoice", backref="usage_records")
    user = relationship("User", backref="usage_records")

    __table_args__ = (
        Index('ix_usage_records_organization', 'organization_id'),
        Index('ix_usage_records_period', 'period_start', 'period_end'),
        Index('ix_usage_records_billed', 'billed'),
        CheckConstraint('quantity >= 0', name='check_positive_quantity'),
        CheckConstraint('period_end >= period_start', name='check_valid_period'),
    )

class PlanAddon(Base, TimestampMixin):
    """Available add-ons for subscription plans"""
    __tablename__ = 'plan_addons'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id = Column(UUID(as_uuid=True), ForeignKey('subscription_plans.id'), nullable=True)

    # Addon details
    name = Column(String(100), nullable=False)
    slug = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)

    # Pricing
    price = Column(Numeric(10, 2), nullable=False)
    billing_interval = Column(String(20), nullable=False)
    setup_fee = Column(Numeric(10, 2), default=0)

    # Stripe integration
    stripe_price_id = Column(String(255), nullable=True, unique=True)

    # Features
    features = Column(JSONB, default=dict)
    limits_increase = Column(JSONB, default=dict)

    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True)

    # Metadata
    metadata = Column(JSONB, default=dict)

    # Relationships
    plan = relationship("SubscriptionPlan", back_populates="addons")
    subscription_addons = relationship("SubscriptionAddon", back_populates="addon")

    __table_args__ = (
        Index('ix_plan_addons_plan', 'plan_id'),
        CheckConstraint('price >= 0', name='check_addon_positive_price'),
    )

class SubscriptionAddon(Base, TimestampMixin):
    """Active add-ons for subscriptions"""
    __tablename__ = 'subscription_addons'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('subscriptions.id', ondelete='CASCADE'), nullable=False)
    addon_id = Column(UUID(as_uuid=True), ForeignKey('plan_addons.id'), nullable=False)

    # Pricing overrides
    custom_price = Column(Numeric(10, 2), nullable=True)
    quantity = Column(Integer, default=1)

    # Dates
    start_date = Column(DateTime(timezone=True), nullable=False, default=datetime.now(UTC))
    end_date = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)

    # Status
    is_active = Column(Boolean, default=True)

    # Relationships
    subscription = relationship("Subscription", back_populates="addons")
    addon = relationship("PlanAddon", back_populates="subscription_addons")

    __table_args__ = (
        UniqueConstraint('subscription_id', 'addon_id', name='uq_subscription_addon'),
        Index('ix_subscription_addons_subscription', 'subscription_id'),
        CheckConstraint('quantity > 0', name='check_positive_addon_quantity'),
    )

class BillingEvent(Base, TimestampMixin):
    """Log of all billing-related events"""
    __tablename__ = 'billing_events'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)

    # Event details
    event_type = Column(String(50), nullable=False)
    event_source = Column(String(50), nullable=False)  # stripe, system, admin, etc.
    event_id = Column(String(255), nullable=True)  # External event ID

    # Related entities
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('subscriptions.id'), nullable=True)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey('invoices.id'), nullable=True)
    payment_id = Column(UUID(as_uuid=True), ForeignKey('payments.id'), nullable=True)

    # Event data
    data = Column(JSONB, nullable=False)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)

    # Relationships
    organization = relationship("Organization", backref="billing_events")

    __table_args__ = (
        Index('ix_billing_events_organization', 'organization_id'),
        Index('ix_billing_events_type', 'event_type'),
        Index('ix_billing_events_processed', 'processed'),
    )