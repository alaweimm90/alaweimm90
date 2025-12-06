"""
Billing Routes
API endpoints for subscription and billing management
"""

from datetime import datetime, UTC
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import stripe

from database import get_db
from auth_service import get_current_user
from auth_models import User
from billing_service import (
    BillingService, CreateSubscriptionRequest, UpdateSubscriptionRequest,
    AddPaymentMethodRequest, RecordUsageRequest, SubscriptionResponse
)
from billing_models import SubscriptionPlan, Subscription, Invoice, Payment, PaymentMethod

router = APIRouter()

# Subscription plans
@router.get("/plans")
async def list_subscription_plans(
    db: AsyncSession = Depends(get_db)
):
    """List available subscription plans"""

    from sqlalchemy import select

    plans_query = select(SubscriptionPlan).where(
        SubscriptionPlan.is_active == True,
        SubscriptionPlan.is_public == True
    ).order_by(SubscriptionPlan.base_price)

    plans_result = await db.execute(plans_query)
    plans = plans_result.scalars().all()

    return [
        {
            "id": str(p.id),
            "name": p.name,
            "slug": p.slug,
            "tier": p.tier,
            "description": p.description,
            "base_price": float(p.base_price),
            "currency": p.currency,
            "billing_interval": p.billing_interval,
            "trial_days": p.trial_days,
            "features": p.features,
            "limits": p.limits,
            "is_recommended": p.is_recommended
        }
        for p in plans
    ]

@router.get("/subscription", response_model=SubscriptionResponse)
async def get_current_subscription(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current organization subscription"""

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    sub_query = select(Subscription).options(
        selectinload(Subscription.plan)
    ).where(
        Subscription.organization_id == user.organization_id,
        Subscription.status.in_(['active', 'trialing'])
    )

    sub_result = await db.execute(sub_query)
    subscription = sub_result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )

    # Calculate next invoice amount
    usage_charges, _ = await BillingService.calculate_usage_charges(subscription, db)
    next_invoice_amount = float(subscription.plan.base_price) + float(usage_charges)

    return SubscriptionResponse(
        id=str(subscription.id),
        plan_name=subscription.plan.name,
        status=subscription.status,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        is_trial=subscription.is_trial,
        usage_summary=subscription.current_usage,
        next_invoice_amount=next_invoice_amount
    )

@router.post("/subscription")
async def create_subscription(
    request: CreateSubscriptionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new subscription"""

    if not user.is_org_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization admin access required"
        )

    subscription = await BillingService.create_subscription(
        organization=user.organization,
        request=request,
        db=db
    )

    return {
        "id": str(subscription.id),
        "status": subscription.status,
        "message": "Subscription created successfully"
    }

@router.put("/subscription")
async def update_subscription(
    request: UpdateSubscriptionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update existing subscription"""

    if not user.is_org_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization admin access required"
        )

    # Get current subscription
    from sqlalchemy import select

    sub_query = select(Subscription).where(
        Subscription.organization_id == user.organization_id,
        Subscription.status.in_(['active', 'trialing'])
    )
    sub_result = await db.execute(sub_query)
    subscription = sub_result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )

    updated_subscription = await BillingService.update_subscription(
        subscription=subscription,
        request=request,
        db=db
    )

    return {
        "id": str(updated_subscription.id),
        "status": updated_subscription.status,
        "message": "Subscription updated successfully"
    }

@router.post("/subscription/cancel")
async def cancel_subscription(
    reason: Optional[str] = None,
    immediate: bool = False,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Cancel subscription"""

    if not user.is_org_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization admin access required"
        )

    # Get current subscription
    from sqlalchemy import select

    sub_query = select(Subscription).where(
        Subscription.organization_id == user.organization_id,
        Subscription.status.in_(['active', 'trialing'])
    )
    sub_result = await db.execute(sub_query)
    subscription = sub_result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )

    cancelled_subscription = await BillingService.cancel_subscription(
        subscription=subscription,
        reason=reason,
        immediate=immediate,
        db=db
    )

    return {
        "id": str(cancelled_subscription.id),
        "status": cancelled_subscription.status,
        "cancelled_at": cancelled_subscription.cancelled_at.isoformat(),
        "message": "Subscription cancelled"
    }

# Payment methods
@router.get("/payment-methods")
async def list_payment_methods(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List organization payment methods"""

    from sqlalchemy import select

    methods_query = select(PaymentMethod).where(
        PaymentMethod.organization_id == user.organization_id
    )
    methods_result = await db.execute(methods_query)
    methods = methods_result.scalars().all()

    return [
        {
            "id": str(m.id),
            "type": m.type,
            "card_brand": m.card_brand,
            "card_last4": m.card_last4,
            "is_default": m.is_default
        }
        for m in methods
    ]

@router.post("/payment-methods")
async def add_payment_method(
    request: AddPaymentMethodRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a new payment method"""

    if not user.is_org_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization admin access required"
        )

    # Get Stripe payment method details
    try:
        stripe_pm = stripe.PaymentMethod.retrieve(request.stripe_payment_method_id)
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payment method: {str(e)}"
        )

    # Create payment method record
    payment_method = PaymentMethod(
        organization_id=user.organization_id,
        stripe_payment_method_id=request.stripe_payment_method_id,
        type=stripe_pm.type,
        card_brand=stripe_pm.card.brand if stripe_pm.type == "card" else None,
        card_last4=stripe_pm.card.last4 if stripe_pm.type == "card" else None,
        card_exp_month=stripe_pm.card.exp_month if stripe_pm.type == "card" else None,
        card_exp_year=stripe_pm.card.exp_year if stripe_pm.type == "card" else None,
        billing_email=user.email,
        billing_address=request.billing_address,
        is_default=request.set_as_default
    )

    db.add(payment_method)
    await db.commit()

    return {
        "id": str(payment_method.id),
        "message": "Payment method added successfully"
    }

@router.delete("/payment-methods/{method_id}")
async def remove_payment_method(
    method_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Remove a payment method"""

    if not user.is_org_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization admin access required"
        )

    from sqlalchemy import select

    method_query = select(PaymentMethod).where(
        PaymentMethod.id == method_id,
        PaymentMethod.organization_id == user.organization_id
    )
    method_result = await db.execute(method_query)
    method = method_result.scalar_one_or_none()

    if not method:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment method not found"
        )

    await db.delete(method)
    await db.commit()

    return {"message": "Payment method removed"}

# Invoices
@router.get("/invoices")
async def list_invoices(
    limit: int = 10,
    offset: int = 0,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List organization invoices"""

    from sqlalchemy import select

    invoices_query = select(Invoice).where(
        Invoice.organization_id == user.organization_id
    ).order_by(
        Invoice.created_at.desc()
    ).limit(limit).offset(offset)

    invoices_result = await db.execute(invoices_query)
    invoices = invoices_result.scalars().all()

    return [
        {
            "id": str(i.id),
            "invoice_number": i.invoice_number,
            "total": i.total / 100,  # Convert from cents
            "currency": i.currency,
            "status": i.status,
            "invoice_date": i.invoice_date.isoformat(),
            "due_date": i.due_date.isoformat(),
            "paid_at": i.paid_at.isoformat() if i.paid_at else None
        }
        for i in invoices
    ]

@router.get("/invoices/{invoice_id}")
async def get_invoice(
    invoice_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get invoice details"""

    from sqlalchemy import select

    invoice_query = select(Invoice).where(
        Invoice.id == invoice_id,
        Invoice.organization_id == user.organization_id
    )
    invoice_result = await db.execute(invoice_query)
    invoice = invoice_result.scalar_one_or_none()

    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice not found"
        )

    return {
        "id": str(invoice.id),
        "invoice_number": invoice.invoice_number,
        "subtotal": invoice.subtotal / 100,
        "tax": invoice.tax / 100,
        "discount": invoice.discount / 100,
        "total": invoice.total / 100,
        "currency": invoice.currency,
        "status": invoice.status,
        "line_items": invoice.line_items,
        "invoice_date": invoice.invoice_date.isoformat(),
        "due_date": invoice.due_date.isoformat(),
        "paid_at": invoice.paid_at.isoformat() if invoice.paid_at else None,
        "pdf_url": invoice.pdf_url,
        "hosted_invoice_url": invoice.hosted_invoice_url
    }

# Usage tracking
@router.post("/usage")
async def record_usage(
    request: RecordUsageRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Record usage for billing"""

    usage_record = await BillingService.record_usage(
        organization=user.organization,
        request=request,
        user=user,
        db=db
    )

    return {
        "id": str(usage_record.id),
        "usage_type": usage_record.usage_type,
        "quantity": float(usage_record.quantity),
        "unit": usage_record.unit,
        "recorded_at": usage_record.recorded_at.isoformat()
    }

@router.get("/usage")
async def get_usage_summary(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current billing period usage summary"""

    from sqlalchemy import select, func
    from billing_models import UsageRecord

    # Get current subscription
    sub_query = select(Subscription).where(
        Subscription.organization_id == user.organization_id,
        Subscription.status.in_(['active', 'trialing'])
    )
    sub_result = await db.execute(sub_query)
    subscription = sub_result.scalar_one_or_none()

    if not subscription:
        return {"usage": {}, "limits": {}}

    # Get usage for current period
    usage_query = select(
        UsageRecord.usage_type,
        func.sum(UsageRecord.quantity).label('total')
    ).where(
        UsageRecord.organization_id == user.organization_id,
        UsageRecord.period_start >= subscription.current_period_start,
        UsageRecord.period_end <= subscription.current_period_end
    ).group_by(UsageRecord.usage_type)

    usage_result = await db.execute(usage_query)

    usage_summary = {
        row.usage_type: float(row.total)
        for row in usage_result
    }

    return {
        "usage": usage_summary,
        "limits": subscription.plan.limits,
        "period_start": subscription.current_period_start.isoformat(),
        "period_end": subscription.current_period_end.isoformat()
    }

# Stripe webhook
@router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """Handle Stripe webhook events"""

    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload,
            stripe_signature,
            os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Process webhook
    await BillingService.handle_stripe_webhook(
        event_type=event["type"],
        event_data=event["data"]["object"],
        db=db
    )

    return {"status": "success"}

# Export router
billing_router = router