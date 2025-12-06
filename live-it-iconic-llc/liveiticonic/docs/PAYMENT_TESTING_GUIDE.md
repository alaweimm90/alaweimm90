# Payment Integration Testing Guide

This guide provides comprehensive testing procedures for the Stripe payment integration in Live It Iconic.

## Table of Contents

1. [Test Environment Setup](#test-environment-setup)
2. [Manual Testing Procedures](#manual-testing-procedures)
3. [Test Card Numbers](#test-card-numbers)
4. [Automated Testing](#automated-testing)
5. [Webhook Testing](#webhook-testing)
6. [Edge Cases and Error Scenarios](#edge-cases-and-error-scenarios)
7. [Performance Testing](#performance-testing)
8. [Security Testing](#security-testing)

## Test Environment Setup

### Prerequisites

- Stripe test account (created by default when you sign up)
- Test API keys from https://dashboard.stripe.com/apikeys
- Stripe CLI installed (optional but recommended)

### Environment Configuration

Create a `.env.local` file for testing:

```bash
# Use test keys (starting with pk_test_ and sk_test_)
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_xxxxxxxxxxxxx
STRIPE_SECRET_KEY=sk_test_xxxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxx

# Optional: Use different backend for testing
VITE_API_BASE_URL=http://localhost:3000
```

### Start Development Server

```bash
npm run dev
```

Visit: http://localhost:5173/checkout

## Manual Testing Procedures

### Test Case 1: Successful Payment

**Steps:**
1. Navigate to collection page
2. Add items to cart
3. Proceed to checkout
4. Enter shipping information:
   - Full name
   - Email
   - Address
   - City, state, ZIP code
5. Click "Next" to payment form
6. Enter cardholder name
7. Enter test card: `4242 4242 4242 4242`
8. Expiration: `12/34`
9. CVC: `567`
10. Click "Pay" button

**Expected Result:**
- Payment processes without error
- Order confirmation page displays
- Order details show correct totals
- Email confirmation sent

**Actual Result:**
- [ ] Payment successful
- [ ] Confirmation page displayed
- [ ] Order saved to database
- [ ] Email sent

---

### Test Case 2: Card Declined

**Steps:**
1. Follow steps 1-7 from Test Case 1
2. Enter test card: `4000 0000 0000 0002` (always declined)
3. Complete form and submit

**Expected Result:**
- Payment is rejected
- Error message displays: "Your card was declined"
- Order is not created
- User can retry with different card

**Actual Result:**
- [ ] Error message displayed
- [ ] Order not created
- [ ] User can retry

---

### Test Case 3: 3D Secure Authentication

**Steps:**
1. Follow steps 1-7 from Test Case 1
2. Enter test card: `4000 0025 0000 3155` (requires authentication)
3. Complete form and submit

**Expected Result:**
- Authentication popup appears
- Complete 3D Secure challenge (approve in test)
- Payment processes
- Order confirmation displays

**Actual Result:**
- [ ] Authentication popup shown
- [ ] Can complete challenge
- [ ] Payment successful

---

### Test Case 4: Insufficient Funds

**Steps:**
1. Follow steps 1-7 from Test Case 1
2. Enter test card: `4000 0000 0000 9995` (insufficient funds)
3. Complete form and submit

**Expected Result:**
- Payment fails
- Error message displays
- Order not created
- User notified of issue

**Actual Result:**
- [ ] Error message displayed
- [ ] Order not created

---

### Test Case 5: Invalid CVC

**Steps:**
1. Follow steps 1-7 from Test Case 1
2. Use valid test card: `4242 4242 4242 4242`
3. Enter CVC: `999` (invalid)
4. Submit form

**Expected Result:**
- Form validation shows error
- Payment not attempted
- User can correct and retry

**Actual Result:**
- [ ] Validation error shown
- [ ] Payment not attempted

---

### Test Case 6: Expired Card

**Steps:**
1. Follow steps 1-7 from Test Case 1
2. Use test card: `4000 0000 0000 0069` (expired)
3. Complete form and submit

**Expected Result:**
- Payment fails with expiration error
- Error message displayed
- Order not created

**Actual Result:**
- [ ] Error message displayed

---

### Test Case 7: Empty Fields

**Steps:**
1. Go to checkout page
2. Skip shipping form, try to access payment
3. Try to submit payment form with empty cardholder name
4. Try to submit without entering card details

**Expected Result:**
- Form validation shows errors
- Cannot submit incomplete forms
- Clear error messages for each field

**Actual Result:**
- [ ] Validation errors shown
- [ ] Cannot submit

---

### Test Case 8: Order Confirmation Email

**Steps:**
1. Complete successful payment (Test Case 1)
2. Check email inbox for confirmation

**Expected Result:**
- Email received within 1 minute
- Contains order number
- Shows order items and totals
- Includes shipping address
- Provides tracking information (when available)

**Actual Result:**
- [ ] Email received
- [ ] All information correct

---

## Test Card Numbers

### Available Test Cards

All test cards use:
- **Expiration:** 12/34
- **CVC:** 567

```javascript
import { STRIPE_TEST_CARDS } from '@/lib/stripe-test-cards';

// Success scenarios
STRIPE_TEST_CARDS.success; // Always succeeds
STRIPE_TEST_CARDS.visaDebit; // Visa debit card
STRIPE_TEST_CARDS.mastercard; // Mastercard

// Failure scenarios
STRIPE_TEST_CARDS.declined; // Card declined
STRIPE_TEST_CARDS.insufficientFunds; // Insufficient funds
STRIPE_TEST_CARDS.lostCard; // Lost card
STRIPE_TEST_CARDS.stolenCard; // Stolen card
STRIPE_TEST_CARDS.processingError; // Processing error
STRIPE_TEST_CARDS.expiredCard; // Expired card
STRIPE_TEST_CARDS.requiresAuthentication; // Needs 3D Secure

// International cards
STRIPE_TEST_CARDS.amex; // American Express
STRIPE_TEST_CARDS.discover; // Discover
STRIPE_TEST_CARDS.jcb; // JCB
```

### Using Test Card Selector

For development convenience, you can create a test card selector component:

```typescript
import { STRIPE_TEST_CARDS } from '@/lib/stripe-test-cards';

export function TestCardSelector() {
  const [selected, setSelected] = useState(STRIPE_TEST_CARDS.success);

  return (
    <div>
      <label>Select Test Card:</label>
      <select value={selected} onChange={(e) => setSelected(e.target.value)}>
        <option value={STRIPE_TEST_CARDS.success}>Always Succeeds</option>
        <option value={STRIPE_TEST_CARDS.declined}>Card Declined</option>
        <option value={STRIPE_TEST_CARDS.requiresAuthentication}>3D Secure</option>
        {/* ... more options */}
      </select>
    </div>
  );
}
```

## Automated Testing

### Unit Tests

Location: `src/components/checkout/__tests__/PaymentForm.test.tsx`

```bash
npm run test -- PaymentForm
```

Test cases:
- [ ] Component renders correctly
- [ ] Form validation works
- [ ] Stripe Elements initialize
- [ ] Error states display properly
- [ ] Success callback fires

### Integration Tests

Location: `src/__tests__/integration/payment-flow.test.ts`

```bash
npm run test -- payment-flow
```

Test scenarios:
- [ ] Complete payment flow end-to-end
- [ ] Order creation after payment
- [ ] Database state verification
- [ ] Email sending confirmation
- [ ] Error handling and recovery

### E2E Tests

Location: `e2e/payment.spec.ts`

```bash
npm run test:e2e -- payment
```

Test scenarios:
- [ ] Full checkout flow in browser
- [ ] Payment success path
- [ ] Payment failure path
- [ ] Order confirmation page
- [ ] Form validation
- [ ] Error messages

### Running All Tests

```bash
# Run all tests
npm run test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Run E2E with UI
npm run test:e2e:ui
```

## Webhook Testing

### Using Stripe CLI

#### Installation

```bash
# macOS
brew install stripe/stripe-cli/stripe

# Linux
curl https://files.stripe.com/stripe-cli/releases/latest/linux/x86_64/stripe_linux_x86_64.tar.gz -o stripe.tar.gz
tar -zxf stripe.tar.gz

# Windows
https://github.com/stripe/stripe-cli/releases
```

#### Setup

```bash
# Login to Stripe account
stripe login

# Forward webhooks to local development server
stripe listen --forward-to localhost:3000/api/webhooks/stripe

# Note the webhook signing secret and add to .env
```

#### Trigger Test Events

```bash
# Successful payment
stripe trigger payment_intent.succeeded

# Failed payment
stripe trigger payment_intent.payment_failed

# Refund
stripe trigger charge.refunded

# Chargeback
stripe trigger charge.dispute.created

# Subscription update
stripe trigger customer.subscription.updated

# View recent events
stripe logs tail
```

### Verify Webhook Handler

After triggering events, check:

1. Server logs for webhook received message
2. Database for updated order status
3. Email inbox for notifications sent

Example in logs:
```
Webhook event received: payment_intent.succeeded
Processing successful payment: {...}
Payment pi_xxx processed successfully
```

## Edge Cases and Error Scenarios

### Test Case: Network Timeout

**Steps:**
1. Start checkout
2. Disconnect internet mid-payment
3. Attempt to submit payment form

**Expected Result:**
- Network error displayed
- User can retry when connection restored
- Order not partially created

---

### Test Case: Duplicate Payment

**Steps:**
1. Complete successful payment
2. Quickly navigate back and attempt payment again

**Expected Result:**
- Second payment attempt fails
- User notified of duplicate attempt
- Only one order created

---

### Test Case: Browser Back Button

**Steps:**
1. Complete payment
2. Before order confirmation loads, press back button
3. Navigate forward again

**Expected Result:**
- Order confirmation still displays
- No duplicate orders created
- Payment intent status verified

---

### Test Case: Multiple Tabs

**Steps:**
1. Open checkout in two browser tabs
2. Complete payment in first tab
3. Try to complete payment in second tab

**Expected Result:**
- First tab shows confirmation
- Second tab fails (payment intent already processed)
- Only one order created

---

### Test Case: Invalid Amount

**Steps:**
1. Modify client-side cart total to invalid value (negative, zero, etc.)
2. Proceed to checkout

**Expected Result:**
- Server-side validation catches error
- Payment not attempted
- Error message displayed

---

## Performance Testing

### Payment Latency

Measure time from form submission to confirmation:

```javascript
console.time('payment');
// ... payment processing ...
console.timeEnd('payment');
```

Target: < 3 seconds for successful payments

### API Response Time

Monitor `/api/payments/create-intent` response time:

```bash
# Using curl
time curl -X POST http://localhost:3000/api/payments/create-intent \
  -H "Content-Type: application/json" \
  -d '{"amount": 9999}'
```

Target: < 500ms

### Webhook Processing

Monitor webhook event processing time:

```javascript
console.time('webhook-processing');
// ... webhook handler ...
console.timeEnd('webhook-processing');
```

Target: < 1 second

### Load Testing

```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:3000/checkout

# Using wrk
wrk -t12 -c400 -d30s http://localhost:3000/checkout
```

## Security Testing

### Test Case: XSS Prevention

**Steps:**
1. Try to inject JavaScript in form fields:
   ```
   <script>alert('XSS')</script>
   ```

**Expected Result:**
- Input sanitized
- JavaScript not executed
- No alert appears

---

### Test Case: CSRF Protection

**Steps:**
1. Create payment from different origin
2. Verify CSRF token validation

**Expected Result:**
- Request rejected
- 403 Forbidden error
- No payment processed

---

### Test Case: Card Data Exposure

**Steps:**
1. Monitor network requests during payment
2. Check browser console logs
3. Review server logs

**Expected Result:**
- Raw card data never transmitted
- Only payment intent ID sent
- No card data in logs
- Stripe Elements handle all card data

---

### Test Case: API Key Security

**Steps:**
1. Check client-side code for secret key exposure
2. Verify environment variables are not committed
3. Check bundle for sensitive data

**Expected Result:**
- No secret keys in client code
- Only publishable key exposed
- .env files in .gitignore
- Bundle doesn't contain secrets

---

## Testing Checklist

### Pre-Launch Testing

- [ ] Successful payment flow works
- [ ] All test cards tested
- [ ] Error handling for all failure scenarios
- [ ] Email notifications sent correctly
- [ ] Order data saved accurately
- [ ] Webhook events processed
- [ ] Form validation working
- [ ] Security checks passed
- [ ] Performance acceptable
- [ ] Mobile responsive
- [ ] Accessibility compliant
- [ ] Cross-browser compatible

### Post-Launch Monitoring

- [ ] Monitor failed payment rates
- [ ] Track webhook delivery failures
- [ ] Monitor API error rates
- [ ] Track customer support tickets
- [ ] Review refund requests
- [ ] Monitor for fraud patterns
- [ ] Track conversion rates

## Contact Support

For payment-related issues:
- Stripe Support: https://support.stripe.com
- Internal: contact@liveiconic.com
