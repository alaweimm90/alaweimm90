# Stripe Payment Integration Guide

## Overview

Live It Iconic uses Stripe for secure payment processing. This guide covers the complete integration including configuration, testing, and production deployment.

## Table of Contents

1. [Setup](#setup)
2. [Configuration](#configuration)
3. [Payment Flow](#payment-flow)
4. [API Endpoints](#api-endpoints)
5. [Testing](#testing)
6. [Webhook Configuration](#webhook-configuration)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Setup

### Prerequisites

- Node.js 18+
- Stripe account (https://dashboard.stripe.com)
- Development and production API keys

### Installation

Stripe dependencies are already installed:

```bash
npm install stripe @stripe/stripe-js @stripe/react-stripe-js @types/stripe
```

### Environment Variables

Configure the following in your `.env` file:

```bash
# Client-side (safe to expose)
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx

# Server-side (MUST be secret)
STRIPE_SECRET_KEY=sk_test_xxxxx

# Webhook verification
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
```

Get these values from: https://dashboard.stripe.com/apikeys

## Configuration

### Client-Side Configuration

The client-side Stripe instance is initialized in `src/lib/stripe.ts`:

```typescript
import { getStripe } from '@/lib/stripe';

// Use in components
const stripe = await getStripe();
```

### Server-Side Configuration

Server-side Stripe operations use `src/api/payments/create-intent.ts`:

```typescript
import Stripe from 'stripe';
import { STRIPE_CONFIG } from '@/lib/stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  apiVersion: STRIPE_CONFIG.API_VERSION,
});
```

## Payment Flow

### 1. Checkout Page Flow

```
User Clicks Checkout
    ↓
Shipping Form → ShippingForm.tsx
    ↓
Payment Form → PaymentForm.tsx
    ↓
Create Payment Intent → POST /api/payments/create-intent
    ↓
Stripe Elements Load
    ↓
User Enters Card Details
    ↓
Confirm Payment → stripe.confirmCardPayment()
    ↓
Stripe Processes Payment
    ↓
Webhook Event → Stripe Webhook Handler
    ↓
Order Confirmation Page
```

### 2. Key Components

#### PaymentForm Component

Located at: `src/components/checkout/PaymentForm.tsx`

- Wraps Stripe Elements provider
- Handles card input via CardElement
- Confirms payment intent
- Manages loading/error states

```typescript
<PaymentForm
  shippingData={shippingInfo}
  amount={99.99}
  onSuccess={(orderId) => navigate(`/order/${orderId}`)}
  onBack={() => setCurrentStep('shipping')}
/>
```

#### Checkout Page

Located at: `src/pages/Checkout.tsx`

- Multi-step checkout flow
- Manages checkout state
- Handles shipping and payment forms

#### Order Confirmation Page

Located at: `src/pages/OrderConfirmation.tsx`

- Displays after payment success
- Fetches order details from server
- Shows order summary and next steps

## API Endpoints

### Create Payment Intent

**Endpoint:** `POST /api/payments/create-intent`

**Request:**

```json
{
  "amount": 9999,
  "currency": "usd",
  "metadata": {
    "orderId": "ORD-123",
    "customerId": "CUST-456"
  },
  "description": "Purchase of merchandise"
}
```

**Response (Success):**

```json
{
  "clientSecret": "pi_xxx_secret_xxx",
  "paymentIntentId": "pi_xxx",
  "status": "requires_payment_method"
}
```

**Response (Error):**

```json
{
  "error": "Invalid amount. Must be a positive number representing cents."
}
```

### Confirm Payment

The payment confirmation happens on the client-side using Stripe.js:

```typescript
const { error, paymentIntent } = await stripe.confirmCardPayment(
  clientSecret,
  {
    payment_method: {
      card: cardElement,
      billing_details: {
        name: cardholderName,
        email: shippingData.email,
      },
    },
  }
);
```

### Webhook Endpoint

**Endpoint:** `POST /api/webhooks/stripe`

**Header Required:**
```
Stripe-Signature: <signature>
```

**Handled Events:**

- `payment_intent.succeeded` - Payment completed
- `payment_intent.payment_failed` - Payment failed
- `charge.refunded` - Refund processed
- `charge.dispute.created` - Chargeback initiated
- `customer.subscription.updated` - Subscription changed
- `customer.subscription.deleted` - Subscription cancelled

## Testing

### Test Mode Setup

Stripe provides test mode for development. Use test API keys (starting with `pk_test_` and `sk_test_`).

### Test Card Numbers

Use cards in `src/lib/stripe-test-cards.ts`:

```typescript
import { STRIPE_TEST_CARDS } from '@/lib/stripe-test-cards';

// Always succeeds
STRIPE_TEST_CARDS.success // '4242424242424242'

// Always fails
STRIPE_TEST_CARDS.declined // '4000000000000002'

// Requires authentication
STRIPE_TEST_CARDS.requiresAuthentication // '4000002500003155'
```

### Manual Testing Checklist

- [ ] Successful payment with valid card
- [ ] Declined card rejection
- [ ] Invalid expiration date
- [ ] Invalid CVC
- [ ] Insufficient funds error
- [ ] 3D Secure authentication (if applicable)
- [ ] Order confirmation displayed
- [ ] Confirmation email sent
- [ ] Order saved to database
- [ ] Webhook events logged

### Automated Testing

```bash
# Run tests
npm run test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage
```

### Local Webhook Testing

Use Stripe CLI to forward webhooks locally:

```bash
# Install Stripe CLI
# https://stripe.com/docs/stripe-cli

# Login to Stripe
stripe login

# Forward webhooks
stripe listen --forward-to localhost:3000/api/webhooks/stripe

# Trigger test events
stripe trigger payment_intent.succeeded
```

## Webhook Configuration

### Setting Up Webhooks in Stripe Dashboard

1. Go to https://dashboard.stripe.com/webhooks
2. Click "Add endpoint"
3. Enter your endpoint URL:
   - **Development:** `http://localhost:3000/api/webhooks/stripe`
   - **Production:** `https://yourdomain.com/api/webhooks/stripe`
4. Select events to listen to
5. Copy the webhook secret to `.env` as `STRIPE_WEBHOOK_SECRET`

### Important Events to Enable

- `payment_intent.succeeded`
- `payment_intent.payment_failed`
- `charge.refunded`
- `charge.dispute.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

### Webhook Signature Verification

All webhooks are verified using the webhook secret:

```typescript
const event = stripe.webhooks.constructEvent(
  body,
  signature,
  webhookSecret
);
```

This ensures the webhook came from Stripe and hasn't been tampered with.

## Security Considerations

### 1. Secret Key Management

- **Never** expose `STRIPE_SECRET_KEY` in client-side code
- **Never** commit `.env` file to version control
- Use environment variables for all sensitive data
- Rotate keys periodically

### 2. Payment Information

- Never store raw card data (PCI DSS violation)
- Use Stripe Elements or Stripe Tokenization
- Store only Stripe payment intent IDs and customer IDs
- Never log or transmit card data

### 3. HTTPS Only

- All payment endpoints must use HTTPS in production
- Stripe will reject requests over HTTP
- Certificate must be valid and not self-signed

### 4. Webhook Security

- Always verify webhook signatures
- Use `STRIPE_WEBHOOK_SECRET` for verification
- Implement idempotency keys for critical operations
- Log all webhook events for audit trail

### 5. Amount Validation

- Validate amounts on server-side
- Prevent negative or zero amounts
- Round to nearest cent (multiply by 100 for cents)
- Validate against cart total

### 6. Rate Limiting

- Implement rate limiting on payment endpoints
- Prevent brute force attacks
- Log suspicious activities
- Use Stripe's built-in fraud detection

### 7. Testing Credentials

- Use test mode for development
- Never test with real payment information
- Rotate test API keys before production deployment
- Use separate keys for development, staging, and production

## Troubleshooting

### Common Issues

#### "Missing VITE_STRIPE_PUBLISHABLE_KEY"

**Solution:** Add the key to `.env`:
```bash
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx
```

#### "Invalid API Key"

**Solution:** Verify the key format and ensure it hasn't been revoked in Stripe Dashboard.

#### "Payment Intent requires payment method"

**Solution:** Card details haven't been provided to Stripe. Check that CardElement is properly rendered and user entered valid data.

#### "Webhook signature verification failed"

**Solution:** Verify `STRIPE_WEBHOOK_SECRET` matches the webhook endpoint secret in Stripe Dashboard.

#### "Stripe is not defined"

**Solution:** Ensure `getStripe()` is properly imported and awaited:
```typescript
const stripe = await getStripe();
if (stripe) {
  // Use stripe...
}
```

### Debug Mode

Enable debug logging:

```typescript
import { getStripe } from '@/lib/stripe';

// Check Stripe loaded
const stripe = await getStripe();
console.log('Stripe instance:', stripe);

// Check Payment Intent
console.log('Client Secret:', clientSecret);
```

### Error Responses

**Invalid Amount:**
```json
{
  "error": "Invalid amount. Must be a positive number representing cents."
}
```

**Missing Secret Key:**
```json
{
  "error": "STRIPE_SECRET_KEY is not configured"
}
```

**API Error:**
```json
{
  "error": "Your card was declined"
}
```

## Production Deployment

### Pre-Production Checklist

- [ ] Switch to production API keys
- [ ] Update all environment variables
- [ ] Configure production webhook endpoint
- [ ] Enable 3D Secure if required by region
- [ ] Set up fraud prevention rules
- [ ] Configure email notifications
- [ ] Test end-to-end payment flow
- [ ] Set up monitoring and alerts
- [ ] Review security settings
- [ ] Enable PCI compliance mode

### Monitoring

Set up alerts for:
- Failed payment attempts
- Webhook delivery failures
- API rate limit errors
- Fraud detection triggers
- Refund requests

### Support

- Stripe Support: https://support.stripe.com
- API Docs: https://stripe.com/docs/api
- Testing Guide: https://stripe.com/docs/testing
- Security Best Practices: https://stripe.com/docs/security

## Additional Resources

- [Stripe JavaScript SDK Docs](https://stripe.com/docs/js)
- [Stripe Elements Guide](https://stripe.com/docs/stripe-js/elements/payment-element)
- [Payment Intent API](https://stripe.com/docs/api/payment_intents)
- [Webhook Events](https://stripe.com/docs/api/events)
- [Testing Guide](https://stripe.com/docs/testing)

## Contact

For integration support, contact the development team or refer to the Stripe documentation.
