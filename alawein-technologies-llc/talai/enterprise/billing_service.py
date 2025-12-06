"""
Billing & Subscription Service
Enterprise billing with Stripe integration and usage-based pricing
"""

import os
import stripe
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
import asyncio
import json
from enum import Enum

from fastapi import HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, EmailStr, Field, validator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from database import get_db
from billing_models import (
    SubscriptionPlan, Subscription, PaymentMethod, Invoice, Payment,
    UsageRecord, PlanAddon, SubscriptionAddon, BillingEvent,
    SubscriptionTier, BillingInterval, PaymentStatus, InvoiceStatus, UsageType
)
from auth_models import Organization, User

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")

# Configuration
class BillingConfig:
    # Default plan limits
    DEFAULT_LIMITS = {
        "free": {
            "api_calls_per_month": 1000,
            "storage_gb": 5,
            "users": 3,
            "projects": 2,
            "compute_hours": 10
        },
        "starter": {
            "api_calls_per_month": 10000,
            "storage_gb": 50,
            "users": 10,
            "projects": 10,
            "compute_hours": 100
        },
        "pro": {
            "api_calls_per_month": 100000,
            "storage_gb": 500,
            "users": 50,
            "projects": 50,
            "compute_hours": 1000
        },
        "team": {
            "api_calls_per_month": 500000,
            "storage_gb": 2000,
            "users": 200,
            "projects": 200,
            "compute_hours": 5000
        },
        "enterprise": {
            "api_calls_per_month": -1,  # unlimited
            "storage_gb": -1,
            "users": -1,
            "projects": -1,
            "compute_hours": -1
        }
    }

    # Overage pricing (per unit after included amount)
    OVERAGE_PRICING = {
        "api_calls": 0.01,  # per 1000 calls
        "storage_gb": 0.10,  # per GB per month
        "compute_hours": 0.50,  # per hour
        "bandwidth_gb": 0.05,  # per GB
        "users": 10.00  # per additional user
    }

    # Tax rates by country/region
    TAX_RATES = {
        "US": {"CA": 0.0875, "NY": 0.08, "TX": 0.0625},  # State sales tax
        "EU": 0.20,  # VAT
        "UK": 0.20,  # VAT
        "AU": 0.10,  # GST
        "default": 0.0
    }

# Request/Response models
class CreateSubscriptionRequest(BaseModel):
    plan_id: str
    billing_email: EmailStr
    payment_method_id: Optional[str] = None
    billing_address: Optional[Dict] = None
    tax_id: Optional[str] = None
    coupon_code: Optional[str] = None

class UpdateSubscriptionRequest(BaseModel):
    plan_id: Optional[str] = None
    billing_email: Optional[EmailStr] = None
    billing_address: Optional[Dict] = None
    auto_renew: Optional[bool] = None
    payment_method_id: Optional[str] = None

class AddPaymentMethodRequest(BaseModel):
    stripe_payment_method_id: str
    set_as_default: bool = False
    billing_address: Optional[Dict] = None

class RecordUsageRequest(BaseModel):
    usage_type: str
    quantity: float
    unit: str
    description: Optional[str] = None
    metadata: Optional[Dict] = None

class SubscriptionResponse(BaseModel):
    id: str
    plan_name: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    is_trial: bool
    usage_summary: Dict
    next_invoice_amount: Optional[float] = None

# Main Billing Service
class BillingService:
    """Comprehensive billing and subscription management"""

    @staticmethod
    async def create_subscription(
        organization: Organization,
        request: CreateSubscriptionRequest,
        db: AsyncSession
    ) -> Subscription:
        """Create a new subscription for an organization"""

        # Get plan
        plan_query = select(SubscriptionPlan).where(
            SubscriptionPlan.id == request.plan_id
        )
        plan_result = await db.execute(plan_query)
        plan = plan_result.scalar_one_or_none()

        if not plan or not plan.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription plan not found or inactive"
            )

        # Check if organization already has active subscription
        existing_query = select(Subscription).where(
            and_(
                Subscription.organization_id == organization.id,
                Subscription.status.in_(['active', 'trialing'])
            )
        )
        existing_result = await db.execute(existing_query)
        existing_sub = existing_result.scalar_one_or_none()

        if existing_sub:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization already has an active subscription"
            )

        # Create Stripe customer if not exists
        if not organization.settings.get("stripe_customer_id"):
            try:
                stripe_customer = stripe.Customer.create(
                    email=request.billing_email,
                    name=organization.name,
                    metadata={"organization_id": str(organization.id)}
                )
                organization.settings["stripe_customer_id"] = stripe_customer.id
                await db.flush()
            except stripe.error.StripeError as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create Stripe customer: {str(e)}"
                )
        else:
            stripe_customer_id = organization.settings["stripe_customer_id"]

        # Calculate trial end
        trial_end = None
        is_trial = False
        if plan.trial_days > 0:
            trial_end = datetime.now(UTC) + timedelta(days=plan.trial_days)
            is_trial = True

        # Calculate billing period
        if plan.billing_interval == BillingInterval.MONTHLY:
            period_end = datetime.now(UTC) + timedelta(days=30)
        elif plan.billing_interval == BillingInterval.QUARTERLY:
            period_end = datetime.now(UTC) + timedelta(days=90)
        elif plan.billing_interval == BillingInterval.YEARLY:
            period_end = datetime.now(UTC) + timedelta(days=365)
        else:
            period_end = datetime.now(UTC) + timedelta(days=30)

        # Create Stripe subscription
        stripe_subscription = None
        if plan.stripe_price_id and request.payment_method_id:
            try:
                # Attach payment method to customer
                stripe.PaymentMethod.attach(
                    request.payment_method_id,
                    customer=stripe_customer_id
                )

                # Set as default payment method
                stripe.Customer.modify(
                    stripe_customer_id,
                    invoice_settings={
                        "default_payment_method": request.payment_method_id
                    }
                )

                # Create subscription
                stripe_subscription = stripe.Subscription.create(
                    customer=stripe_customer_id,
                    items=[{"price": plan.stripe_price_id}],
                    trial_end=int(trial_end.timestamp()) if trial_end else None,
                    metadata={
                        "organization_id": str(organization.id),
                        "plan_id": str(plan.id)
                    },
                    expand=["latest_invoice"]
                )
            except stripe.error.StripeError as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create Stripe subscription: {str(e)}"
                )

        # Create subscription record
        subscription = Subscription(
            organization_id=organization.id,
            plan_id=plan.id,
            stripe_subscription_id=stripe_subscription.id if stripe_subscription else None,
            stripe_customer_id=stripe_customer_id if stripe_subscription else None,
            billing_email=request.billing_email,
            billing_name=organization.name,
            billing_address=request.billing_address,
            tax_id=request.tax_id,
            start_date=datetime.now(UTC),
            current_period_start=datetime.now(UTC),
            current_period_end=period_end,
            trial_end=trial_end,
            status='trialing' if is_trial else 'active',
            is_trial=is_trial,
            auto_renew=True,
            current_usage={
                "api_calls": 0,
                "storage_gb": 0,
                "compute_hours": 0,
                "bandwidth_gb": 0,
                "users": 0
            }
        )

        # Apply coupon if provided
        if request.coupon_code:
            # Validate and apply coupon logic here
            pass

        db.add(subscription)
        await db.commit()

        # Update organization subscription tier
        organization.subscription_tier = plan.tier
        await db.commit()

        # Log event
        await BillingService._log_billing_event(
            db=db,
            organization_id=organization.id,
            event_type="subscription_created",
            event_source="system",
            subscription_id=subscription.id,
            data={
                "plan_name": plan.name,
                "trial_days": plan.trial_days,
                "billing_interval": plan.billing_interval
            }
        )

        return subscription

    @staticmethod
    async def update_subscription(
        subscription: Subscription,
        request: UpdateSubscriptionRequest,
        db: AsyncSession
    ) -> Subscription:
        """Update existing subscription"""

        # Update plan if requested
        if request.plan_id:
            # Get new plan
            plan_query = select(SubscriptionPlan).where(
                SubscriptionPlan.id == request.plan_id
            )
            plan_result = await db.execute(plan_query)
            new_plan = plan_result.scalar_one_or_none()

            if not new_plan or not new_plan.is_active:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="New subscription plan not found or inactive"
                )

            # Update Stripe subscription if exists
            if subscription.stripe_subscription_id and new_plan.stripe_price_id:
                try:
                    stripe_sub = stripe.Subscription.retrieve(subscription.stripe_subscription_id)
                    stripe.Subscription.modify(
                        subscription.stripe_subscription_id,
                        items=[{
                            "id": stripe_sub["items"]["data"][0].id,
                            "price": new_plan.stripe_price_id
                        }],
                        proration_behavior="create_prorations"
                    )
                except stripe.error.StripeError as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to update Stripe subscription: {str(e)}"
                    )

            subscription.plan_id = new_plan.id

        # Update billing details
        if request.billing_email:
            subscription.billing_email = request.billing_email
        if request.billing_address:
            subscription.billing_address = request.billing_address
        if request.auto_renew is not None:
            subscription.auto_renew = request.auto_renew
        if request.payment_method_id:
            subscription.payment_method_id = request.payment_method_id

        await db.commit()

        return subscription

    @staticmethod
    async def cancel_subscription(
        subscription: Subscription,
        reason: Optional[str],
        immediate: bool,
        db: AsyncSession
    ) -> Subscription:
        """Cancel a subscription"""

        # Cancel Stripe subscription
        if subscription.stripe_subscription_id:
            try:
                if immediate:
                    stripe.Subscription.delete(subscription.stripe_subscription_id)
                else:
                    stripe.Subscription.modify(
                        subscription.stripe_subscription_id,
                        cancel_at_period_end=True
                    )
            except stripe.error.StripeError as e:
                # Log error but continue with local cancellation
                print(f"Failed to cancel Stripe subscription: {str(e)}")

        # Update subscription
        subscription.cancelled_at = datetime.now(UTC)
        subscription.cancellation_reason = reason

        if immediate:
            subscription.status = 'cancelled'
            subscription.ended_at = datetime.now(UTC)
        else:
            subscription.auto_renew = False

        await db.commit()

        # Log event
        await BillingService._log_billing_event(
            db=db,
            organization_id=subscription.organization_id,
            event_type="subscription_cancelled",
            event_source="system",
            subscription_id=subscription.id,
            data={
                "reason": reason,
                "immediate": immediate
            }
        )

        return subscription

    @staticmethod
    async def record_usage(
        organization: Organization,
        request: RecordUsageRequest,
        user: Optional[User],
        db: AsyncSession
    ) -> UsageRecord:
        """Record usage for billing purposes"""

        # Get active subscription
        sub_query = select(Subscription).where(
            and_(
                Subscription.organization_id == organization.id,
                Subscription.status.in_(['active', 'trialing'])
            )
        )
        sub_result = await db.execute(sub_query)
        subscription = sub_result.scalar_one_or_none()

        # Calculate price based on usage type
        unit_price = BillingConfig.OVERAGE_PRICING.get(request.usage_type, 0)
        total_price = Decimal(str(request.quantity * unit_price))

        # Create usage record
        usage_record = UsageRecord(
            organization_id=organization.id,
            subscription_id=subscription.id if subscription else None,
            usage_type=request.usage_type,
            quantity=request.quantity,
            unit=request.unit,
            unit_price=unit_price,
            total_price=total_price,
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC),
            user_id=user.id if user else None,
            description=request.description,
            metadata=request.metadata or {}
        )

        db.add(usage_record)

        # Update subscription current usage
        if subscription:
            if request.usage_type in subscription.current_usage:
                subscription.current_usage[request.usage_type] += request.quantity
            else:
                subscription.current_usage[request.usage_type] = request.quantity
            subscription.usage_updated_at = datetime.now(UTC)

        await db.commit()

        return usage_record

    @staticmethod
    async def calculate_usage_charges(
        subscription: Subscription,
        db: AsyncSession
    ) -> Tuple[Decimal, List[Dict]]:
        """Calculate usage-based charges for the current billing period"""

        # Get unbilled usage records
        usage_query = select(UsageRecord).where(
            and_(
                UsageRecord.subscription_id == subscription.id,
                UsageRecord.billed == False,
                UsageRecord.is_billable == True,
                UsageRecord.period_start >= subscription.current_period_start,
                UsageRecord.period_end <= subscription.current_period_end
            )
        )
        usage_result = await db.execute(usage_query)
        usage_records = usage_result.scalars().all()

        # Get plan for limits
        plan_query = select(SubscriptionPlan).where(
            SubscriptionPlan.id == subscription.plan_id
        )
        plan_result = await db.execute(plan_query)
        plan = plan_result.scalar_one()

        total_charges = Decimal('0')
        line_items = []

        # Aggregate usage by type
        usage_summary = {}
        for record in usage_records:
            if record.usage_type not in usage_summary:
                usage_summary[record.usage_type] = {
                    "quantity": 0,
                    "unit": record.unit,
                    "records": []
                }
            usage_summary[record.usage_type]["quantity"] += float(record.quantity)
            usage_summary[record.usage_type]["records"].append(record)

        # Calculate charges for each usage type
        for usage_type, summary in usage_summary.items():
            included = plan.usage_pricing.get(usage_type, {}).get("included", 0)
            overage = max(0, summary["quantity"] - included)

            if overage > 0:
                rate = Decimal(str(plan.usage_pricing.get(usage_type, {}).get(
                    f"per_{summary['unit']}",
                    BillingConfig.OVERAGE_PRICING.get(usage_type, 0)
                )))
                charge = overage * rate
                total_charges += charge

                line_items.append({
                    "type": "usage",
                    "description": f"{usage_type} overage",
                    "quantity": overage,
                    "unit": summary["unit"],
                    "unit_price": float(rate),
                    "total": float(charge)
                })

                # Mark records as billed
                for record in summary["records"]:
                    record.billed = True
                    record.billed_at = datetime.now(UTC)

        await db.commit()

        return total_charges, line_items

    @staticmethod
    async def generate_invoice(
        subscription: Subscription,
        db: AsyncSession
    ) -> Invoice:
        """Generate invoice for subscription"""

        # Calculate usage charges
        usage_charges, usage_line_items = await BillingService.calculate_usage_charges(
            subscription, db
        )

        # Get plan
        plan_query = select(SubscriptionPlan).where(
            SubscriptionPlan.id == subscription.plan_id
        )
        plan_result = await db.execute(plan_query)
        plan = plan_result.scalar_one()

        # Build line items
        line_items = [{
            "type": "subscription",
            "description": f"{plan.name} subscription",
            "quantity": 1,
            "unit_price": float(subscription.custom_price or plan.base_price),
            "total": float(subscription.custom_price or plan.base_price)
        }]
        line_items.extend(usage_line_items)

        # Calculate totals
        subtotal = sum(item["total"] for item in line_items)

        # Apply discount
        discount = 0
        if subscription.discount_percent:
            discount = subtotal * float(subscription.discount_percent) / 100
        elif subscription.discount_amount:
            discount = float(subscription.discount_amount)

        # Calculate tax
        tax = await BillingService._calculate_tax(
            subscription.billing_address,
            subtotal - discount
        )

        # Create invoice
        total = subtotal - discount + tax
        invoice = Invoice(
            organization_id=subscription.organization_id,
            subscription_id=subscription.id,
            invoice_number=await BillingService._generate_invoice_number(db),
            subtotal=int(subtotal * 100),  # Convert to cents
            tax=int(tax * 100),
            discount=int(discount * 100),
            total=int(total * 100),
            amount_due=int(total * 100),
            currency="USD",
            due_date=datetime.now(UTC) + timedelta(days=30),
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
            status=InvoiceStatus.DRAFT,
            line_items=line_items
        )

        db.add(invoice)
        await db.commit()

        return invoice

    @staticmethod
    async def process_payment(
        invoice: Invoice,
        payment_method: PaymentMethod,
        db: AsyncSession
    ) -> Payment:
        """Process payment for an invoice"""

        # Create payment intent with Stripe
        payment_intent = None
        if payment_method.stripe_payment_method_id:
            try:
                payment_intent = stripe.PaymentIntent.create(
                    amount=invoice.amount_due,
                    currency=invoice.currency.lower(),
                    payment_method=payment_method.stripe_payment_method_id,
                    customer=invoice.subscription.stripe_customer_id,
                    confirm=True,
                    metadata={
                        "invoice_id": str(invoice.id),
                        "organization_id": str(invoice.organization_id)
                    }
                )
            except stripe.error.StripeError as e:
                # Create failed payment record
                payment = Payment(
                    organization_id=invoice.organization_id,
                    invoice_id=invoice.id,
                    amount=invoice.amount_due,
                    currency=invoice.currency,
                    status=PaymentStatus.FAILED,
                    failure_reason=str(e),
                    payment_method_id=payment_method.id
                )
                db.add(payment)
                await db.commit()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Payment failed: {str(e)}"
                )

        # Create payment record
        payment = Payment(
            organization_id=invoice.organization_id,
            invoice_id=invoice.id,
            amount=invoice.amount_due,
            currency=invoice.currency,
            stripe_payment_intent_id=payment_intent.id if payment_intent else None,
            stripe_charge_id=payment_intent.charges.data[0].id if payment_intent else None,
            status=PaymentStatus.SUCCEEDED if payment_intent and payment_intent.status == "succeeded" else PaymentStatus.PENDING,
            payment_method_id=payment_method.id,
            processed_at=datetime.now(UTC) if payment_intent else None
        )
        db.add(payment)

        # Update invoice if payment succeeded
        if payment.status == PaymentStatus.SUCCEEDED:
            invoice.status = InvoiceStatus.PAID
            invoice.paid_at = datetime.now(UTC)
            invoice.amount_paid = invoice.total

        await db.commit()

        return payment

    @staticmethod
    async def handle_stripe_webhook(
        event_type: str,
        event_data: Dict,
        db: AsyncSession
    ):
        """Handle Stripe webhook events"""

        # Log event
        billing_event = BillingEvent(
            organization_id=event_data.get("metadata", {}).get("organization_id"),
            event_type=f"stripe.{event_type}",
            event_source="stripe",
            event_id=event_data.get("id"),
            data=event_data,
            processed=False
        )
        db.add(billing_event)
        await db.commit()

        # Process based on event type
        if event_type == "invoice.payment_succeeded":
            # Update invoice and payment status
            pass
        elif event_type == "invoice.payment_failed":
            # Handle failed payment
            pass
        elif event_type == "subscription.updated":
            # Sync subscription changes
            pass
        elif event_type == "subscription.deleted":
            # Handle subscription cancellation
            pass

        # Mark as processed
        billing_event.processed = True
        billing_event.processed_at = datetime.now(UTC)
        await db.commit()

    @staticmethod
    async def _calculate_tax(
        billing_address: Optional[Dict],
        amount: float
    ) -> float:
        """Calculate tax based on billing address"""

        if not billing_address:
            return 0

        country = billing_address.get("country", "").upper()
        state = billing_address.get("state", "").upper()

        # Get applicable tax rate
        tax_rate = 0
        if country == "US" and state in BillingConfig.TAX_RATES["US"]:
            tax_rate = BillingConfig.TAX_RATES["US"][state]
        elif country in BillingConfig.TAX_RATES:
            tax_rate = BillingConfig.TAX_RATES[country]
        else:
            tax_rate = BillingConfig.TAX_RATES["default"]

        return amount * tax_rate

    @staticmethod
    async def _generate_invoice_number(db: AsyncSession) -> str:
        """Generate unique invoice number"""

        # Get count of invoices this year
        year = datetime.now(UTC).year
        count_query = select(func.count(Invoice.id)).where(
            func.extract('year', Invoice.created_at) == year
        )
        count_result = await db.execute(count_query)
        count = count_result.scalar() or 0

        return f"INV-{year}-{count + 1:06d}"

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _log_billing_event(
        db: AsyncSession,
        organization_id: str,
        event_type: str,
        event_source: str,
        data: Dict,
        subscription_id: Optional[str] = None,
        invoice_id: Optional[str] = None,
        payment_id: Optional[str] = None
    ):
        """Log billing event for audit trail"""

        event = BillingEvent(
            organization_id=organization_id,
            event_type=event_type,
            event_source=event_source,
            data=data,
            subscription_id=subscription_id,
            invoice_id=invoice_id,
            payment_id=payment_id,
            processed=True,
            processed_at=datetime.now(UTC)
        )
        db.add(event)
        await db.commit()

# Export main functions
__all__ = [
    "BillingService",
    "BillingConfig",
    "CreateSubscriptionRequest",
    "UpdateSubscriptionRequest",
    "AddPaymentMethodRequest",
    "RecordUsageRequest",
    "SubscriptionResponse"
]