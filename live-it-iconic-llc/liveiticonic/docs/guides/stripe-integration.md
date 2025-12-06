# Stripe Payment Integration - Live It Iconic

## Overview

This document provides a quick reference for the Stripe payment integration implemented in Live It Iconic. For detailed information, see [docs/PAYMENT_INTEGRATION.md](./docs/PAYMENT_INTEGRATION.md).

## Quick Start

### 1. Get Stripe API Keys

1. Sign up at https://stripe.com (or sign in)
2. Go to https://dashboard.stripe.com/apikeys
3. Copy your test keys (for development)

### 2. Configure Environment

Create `.env` file:

```bash
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx
STRIPE_SECRET_KEY=sk_test_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
```

### 3. Start Development

```bash
npm run dev
```

Visit: http://localhost:5173

## File Structure

```
src/
├── api/
│   ├── payments/
│   │   ├── create-intent.ts          # Create payment intent API
│   │   └── confirm-payment.ts        # Confirm payment API
│   └── webhooks/
│       └── stripe.ts                 # Stripe webhook handler
├── components/checkout/
│   └── PaymentForm.tsx               # Payment form component
├── pages/
│   ├── Checkout.tsx                  # Checkout page
│   ├── CheckoutSuccess.tsx           # Order confirmation (legacy)
│   └── OrderConfirmation.tsx         # Order confirmation (new)
├── services/
│   └── stripeService.ts              # Stripe service utilities
└── lib/
    ├── stripe.ts                     # Stripe configuration
    └── stripe-test-cards.ts          # Test card numbers

docs/
├── PAYMENT_INTEGRATION.md            # Complete integration guide
└── PAYMENT_TESTING_GUIDE.md          # Testing procedures
```

## Key Features

### Payment Flow

```
Checkout Form
  ↓
Create Payment Intent (/api/payments/create-intent)
  ↓
Stripe Elements (Card Input)
  ↓
Confirm Payment (stripe.confirmCardPayment)
  ↓
Stripe Webhook (payment_intent.succeeded)
  ↓
Order Confirmation Page
```

### Supported Features

- Credit/Debit Cards (Visa, Mastercard, Amex, Discover, etc.)
- Digital Wallets (Apple Pay, Google Pay)
- 3D Secure Authentication
- Payment Intent API
- Automatic Payment Methods
- Webhook Processing
- Error Handling & Recovery
- Test Mode for Development

### Security Features

- PCI Compliance (no raw card data handled)
- Stripe Elements for secure card input
- Webhook signature verification
- HTTPS enforcement
- Payment intent validation
- Server-side amount validation
- CSRF protection

## Testing

### Using Test Cards

```typescript
import { STRIPE_TEST_CARDS } from '@/lib/stripe-test-cards';

// Success
STRIPE_TEST_CARDS.success      // 4242 4242 4242 4242

// Failure scenarios
STRIPE_TEST_CARDS.declined     // 4000 0000 0000 0002
STRIPE_TEST_CARDS.requiresAuthentication // 4000 0025 0000 3155
STRIPE_TEST_CARDS.insufficientFunds      // 4000 0000 0000 9995
```

All test cards use: **Expiration:** 12/34, **CVC:** 567

### Manual Testing

1. Go to checkout
2. Add items to cart
3. Enter shipping information
4. Enter payment info with test card
5. Complete payment
6. View order confirmation

### Automated Testing

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# With coverage
npm run test:coverage
```

### Webhook Testing

```bash
# Install Stripe CLI
brew install stripe/stripe-cli/stripe

# Login and forward webhooks
stripe login
stripe listen --forward-to localhost:3000/api/webhooks/stripe

# Trigger test event
stripe trigger payment_intent.succeeded
```

## API Endpoints

### Create Payment Intent

```
POST /api/payments/create-intent
Content-Type: application/json

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

**Response:**
```json
{
  "clientSecret": "pi_xxx_secret_xxx",
  "paymentIntentId": "pi_xxx",
  "status": "requires_payment_method"
}
```

### Webhook Endpoint

```
POST /api/webhooks/stripe
Header: Stripe-Signature: <signature>

Handles:
- payment_intent.succeeded
- payment_intent.payment_failed
- charge.refunded
- charge.dispute.created
- customer.subscription.updated
- customer.subscription.deleted
```

## Usage Examples

### Creating a Payment Intent

```typescript
import { stripeService } from '@/services/stripeService';

const intent = await stripeService.createPaymentIntent({
  amount: 9999, // $99.99
  currency: 'usd',
  metadata: {
    orderId: 'ORD-123',
  },
  description: 'Order ORD-123',
});

console.log(intent.clientSecret); // Use in Stripe Elements
```

### Confirming Payment

```typescript
const { error, paymentIntent } = await stripeService.confirmPayment(
  stripe,
  elements,
  clientSecret,
  `${window.location.origin}/order-confirmation`
);

if (error) {
  console.error(stripeService.getErrorMessage(error));
} else if (stripeService.isSuccessful(paymentIntent)) {
  console.log('Payment successful!');
}
```

### Error Handling

```typescript
import { stripeService } from '@/services/stripeService';

try {
  const intent = await stripeService.createPaymentIntent({...});
} catch (error) {
  const message = error instanceof Error
    ? error.message
    : 'Payment failed';
  showError(message);
}
```

### Formatting Amounts

```typescript
import { stripeService } from '@/services/stripeService';

// Convert dollars to cents
const cents = stripeService.dollarsToCents(99.99); // 9999

// Convert cents to dollars
const dollars = stripeService.centsToDollars(9999); // 99.99

// Format for display
const formatted = stripeService.formatAmount(9999); // "$99.99"
```

## Configuration

### Stripe Config

Located in `src/lib/stripe.ts`:

```typescript
export const STRIPE_CONFIG = {
  API_VERSION: '2024-11-20',
  CURRENCY: 'usd',
  TAX_BEHAVIOR: 'inclusive' as const,
};
```

### Environment Variables

```bash
# Required for client
VITE_STRIPE_PUBLISHABLE_KEY

# Required for server
STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET

# Optional
STRIPE_API_VERSION
```

## Troubleshooting

### "Missing VITE_STRIPE_PUBLISHABLE_KEY"

Add to `.env`:
```bash
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx
```

### "Invalid API Key"

- Check key format (starts with `pk_test_` or `pk_live_`)
- Verify not revoked in Stripe Dashboard
- Test vs Live key mismatch

### "Webhook signature verification failed"

- Verify `STRIPE_WEBHOOK_SECRET` matches Stripe Dashboard
- Ensure webhook endpoint is correct
- Check webhook is enabled for correct events

### "Payment Intent requires payment method"

- CardElement not properly rendered
- User hasn't entered card details
- Card element lost focus before submission

## Production Deployment

### Pre-Production Checklist

- [ ] Switch to production API keys
- [ ] Update `.env` variables
- [ ] Enable production webhook endpoint
- [ ] Configure 3D Secure rules
- [ ] Set up fraud prevention
- [ ] Configure email notifications
- [ ] Test end-to-end flow
- [ ] Set up monitoring/alerts
- [ ] Review security settings
- [ ] Enable PCI compliance

### Production Keys

1. Go to https://dashboard.stripe.com/apikeys
2. Toggle "Viewing test data" to OFF
3. Copy live keys (start with `pk_live_` and `sk_live_`)
4. Update environment variables

### Webhook Configuration

1. https://dashboard.stripe.com/webhooks
2. Add endpoint: `https://yourdomain.com/api/webhooks/stripe`
3. Select events to listen
4. Copy webhook secret
5. Update `.env` with secret

## Support Resources

- **Stripe API Docs:** https://stripe.com/docs/api
- **Stripe Testing Guide:** https://stripe.com/docs/testing
- **Stripe Security:** https://stripe.com/docs/security
- **Stripe Support:** https://support.stripe.com

## Documentation

- [Payment Integration Guide](./docs/PAYMENT_INTEGRATION.md) - Complete integration details
- [Testing Guide](./docs/PAYMENT_TESTING_GUIDE.md) - Testing procedures and checklist

## Contact

For integration questions or issues:
- Email: support@liveiconic.com
- Stripe Support: https://support.stripe.com
