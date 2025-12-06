# Email Templates - Quick Start Guide

A quick reference guide for using Live It Iconic email templates.

## Installation

All dependencies are already installed:
- `react-email` - Email component library
- `@react-email/components` - Pre-built email components
- `resend` - Email delivery service

## Setup

1. **Get API Key**: Sign up at [Resend.com](https://resend.com) and get your API key
2. **Configure Environment**:
   ```bash
   # Add to .env file
   RESEND_API_KEY=re_your_api_key_here
   EMAIL_FROM=hello@liveiconic.com
   EMAIL_REPLY_TO=support@liveiconic.com
   ```
3. **Verify Domain**: Add your domain to Resend dashboard

## Quick Start Examples

### 1. Send a Welcome Email

```typescript
import { sendEmail } from '@/emails/utils/sendEmail';
import { WelcomeEmail } from '@/emails/templates/WelcomeEmail';

await sendEmail({
  to: 'user@example.com',
  subject: 'Welcome to Live It Iconic',
  react: <WelcomeEmail name="John" />
});
```

### 2. Send Order Confirmation

```typescript
import { sendEmail } from '@/emails/utils/sendEmail';
import { OrderConfirmation } from '@/emails/templates/OrderConfirmation';

await sendEmail({
  to: 'customer@example.com',
  subject: 'Order #12345 Confirmed',
  react: (
    <OrderConfirmation
      orderId="12345"
      name="John Doe"
      items={[
        {
          name: 'Premium T-Shirt',
          quantity: 1,
          price: 59.99,
          image: 'https://...',
          sku: 'TSH-001'
        }
      ]}
      subtotal={59.99}
      shipping={10.00}
      tax={5.60}
      total={75.59}
      shippingAddress="123 Main St, New York, NY 10001"
      estimatedDelivery="November 20, 2025"
    />
  )
});
```

### 3. Send Shipping Notification

```typescript
import { sendEmail } from '@/emails/utils/sendEmail';
import { ShippingNotification } from '@/emails/templates/ShippingNotification';

await sendEmail({
  to: 'customer@example.com',
  subject: 'Your Order Has Shipped',
  react: (
    <ShippingNotification
      name="John Doe"
      orderId="12345"
      trackingNumber="1Z999AA1012345678"
      carrier="UPS"
      estimatedDelivery="November 20, 2025"
    />
  )
});
```

### 4. Send Password Reset

```typescript
import { sendEmail } from '@/emails/utils/sendEmail';
import { PasswordReset } from '@/emails/templates/PasswordReset';

const resetToken = generateToken();
await sendEmail({
  to: 'user@example.com',
  subject: 'Reset Your Password',
  react: (
    <PasswordReset
      name="John"
      resetUrl={`https://liveiconic.com/reset?token=${resetToken}`}
      expirationTime="24 hours"
    />
  )
});
```

### 5. Send Newsletter Confirmation

```typescript
import { sendEmail } from '@/emails/utils/sendEmail';
import { NewsletterSubscription } from '@/emails/templates/NewsletterSubscription';

const confirmToken = generateToken();
await sendEmail({
  to: 'subscriber@example.com',
  subject: 'Confirm Your Newsletter',
  react: (
    <NewsletterSubscription
      name="John"
      confirmUrl={`https://liveiconic.com/confirm?token=${confirmToken}`}
      interests={['New Releases', 'Styling Tips']}
    />
  )
});
```

### 6. Send Abandoned Cart Email

```typescript
import { sendEmail } from '@/emails/utils/sendEmail';
import { AbandonedCart } from '@/emails/templates/AbandonedCart';

await sendEmail({
  to: 'customer@example.com',
  subject: 'You Left Something Behind',
  react: (
    <AbandonedCart
      name="John"
      items={[
        {
          id: '1',
          name: 'Premium T-Shirt',
          price: 59.99,
          quantity: 1,
          image: 'https://...'
        }
      ]}
      cartUrl="https://liveiconic.com/cart/abc123"
      cartTotal={59.99}
      couponCode="CART10"
      couponDiscount={10}
    />
  )
});
```

## Email Templates Overview

| Template | File | Use Case | Key Data |
|----------|------|----------|----------|
| **Welcome** | `WelcomeEmail.tsx` | New user signup | `name` |
| **Order Confirmation** | `OrderConfirmation.tsx` | After purchase | `orderId`, `items`, `total`, `address` |
| **Shipping** | `ShippingNotification.tsx` | Order shipped | `trackingNumber`, `carrier`, `estimatedDelivery` |
| **Password Reset** | `PasswordReset.tsx` | Password recovery | `resetUrl`, `expirationTime` |
| **Newsletter** | `NewsletterSubscription.tsx` | Newsletter signup | `confirmUrl`, `interests` |
| **Abandoned Cart** | `AbandonedCart.tsx` | Cart recovery | `items`, `cartUrl`, `total` |

## Key Features by Template

### Welcome Email
- Personal greeting
- Brand introduction
- Shop link
- 10% first-time discount code
- Benefits list

### Order Confirmation
- Order number and date
- Itemized products with images
- Order totals breakdown
- Shipping address
- Estimated delivery date
- Timeline of next steps

### Shipping Notification
- Carrier tracking link
- Estimated delivery
- Delivery tips
- Support information
- Social media follow-up

### Password Reset
- Secure reset button
- Link expiration notice
- Security best practices
- Support contact
- "Didn't request?" section

### Newsletter Subscription
- Subscription confirmation
- Topics/interests
- What they'll receive
- Frequency information
- Preference management
- Social links

### Abandoned Cart
- Product images
- Cart summary
- Limited-time discount
- Return-to-cart button
- Why shop section
- Urgency messaging

## Common Tasks

### Send Email with Error Handling

```typescript
try {
  const result = await sendEmail({
    to: email,
    subject: 'Welcome',
    react: <WelcomeEmail name="John" />
  });

  if (result.success) {
    console.log('Email sent:', result.data?.id);
  } else {
    console.error('Failed:', result.error?.message);
  }
} catch (error) {
  console.error('Exception:', error);
}
```

### Send to Multiple Recipients

```typescript
const emails = ['user1@example.com', 'user2@example.com'];

for (const email of emails) {
  await sendEmail({
    to: email,
    subject: 'Welcome',
    react: <WelcomeEmail name="User" />
  });
}
```

### Track Email Delivery

```typescript
import { trackEmailEvent, EmailTemplate } from '@/emails/utils/sendEmail';

const result = await sendEmail({...});
if (result.success && result.data?.id) {
  await trackEmailEvent(
    result.data.id,
    EmailTemplate.WELCOME,
    'user@example.com'
  );
}
```

## TypeScript Types

All email templates are fully typed. Import types for better IDE support:

```typescript
import type {
  WelcomeEmailProps,
  OrderConfirmationProps,
  ShippingNotificationProps,
  PasswordResetProps,
  NewsletterSubscriptionProps,
  AbandonedCartProps
} from '@/emails/templates';

import type { SendEmailResponse, SendEmailOptions } from '@/emails/utils/sendEmail';
```

## File Structure

```
src/emails/
├── components/
│   ├── EmailLayout.tsx      # Base wrapper - handles structure
│   ├── EmailHeader.tsx       # Logo and branding
│   └── EmailFooter.tsx       # Links and footer
├── templates/
│   ├── WelcomeEmail.tsx              # New user welcome
│   ├── OrderConfirmation.tsx         # Order confirmation
│   ├── ShippingNotification.tsx      # Shipping update
│   ├── PasswordReset.tsx             # Password reset
│   ├── NewsletterSubscription.tsx    # Newsletter signup
│   └── AbandonedCart.tsx             # Cart recovery
├── utils/
│   └── sendEmail.ts          # Email sending logic
├── examples.ts               # Usage examples
└── index.ts                  # Central exports
```

## Brand Colors

```typescript
- Primary Gold: #C1A060      // Accent color
- Dark: #0B0B0C              // Text and backgrounds
- Gray: #5C6270              // Secondary text
- Border: #E6E9EF            // Light borders
- Success: #4CAF50           // Success indicators
- Light Background: #f8f8f8  // Sections
```

## Preview Emails Locally

```bash
# Start development server
npm run email:preview

# Open http://localhost:3000
# Browse and preview all email templates
```

## Best Practices

### Do's
- ✅ Use clear subject lines
- ✅ Include unsubscribe link (in footer)
- ✅ Personalize with customer name
- ✅ Add clear call-to-action buttons
- ✅ Test in multiple email clients
- ✅ Use responsive design
- ✅ Optimize images for email
- ✅ Validate email addresses before sending

### Don'ts
- ❌ Send emails without API key
- ❌ Include unverified domains
- ❌ Forget error handling
- ❌ Use overly large file sizes
- ❌ Include sensitive data
- ❌ Skip testing in actual email clients
- ❌ Use misleading subject lines
- ❌ Forget legal compliance (GDPR, CAN-SPAM)

## Environment Variables Required

```env
# Required
RESEND_API_KEY=re_your_key_here

# Optional
EMAIL_FROM=hello@liveiconic.com
EMAIL_REPLY_TO=support@liveiconic.com
```

## Resend Configuration

1. **Sign Up**: [resend.com](https://resend.com)
2. **Get API Key**: Dashboard → API Keys
3. **Verify Domain**: Add your domain for "from" address
4. **Sender Identity**: Set up From and Reply-To addresses
5. **Test**: Send test emails to confirm setup

## Testing

### Manual Testing
```bash
# Preview locally
npm run email:preview

# Send test email
const result = await sendEmail({
  to: 'your-email@example.com',
  subject: 'Test',
  react: <WelcomeEmail name="Test" />
});
```

### Automated Testing
```typescript
import { render } from '@react-email/components';
import { WelcomeEmail } from '@/emails/templates/WelcomeEmail';

test('WelcomeEmail renders', () => {
  const html = render(<WelcomeEmail name="John" />);
  expect(html).toContain('Welcome');
});
```

## Troubleshooting

### Email Not Sending?

1. **Check API Key**
   ```typescript
   // Verify in .env
   RESEND_API_KEY=re_... (should start with 're_')
   ```

2. **Verify Domain**
   - Go to Resend dashboard
   - Check domain is verified for sending

3. **Check Email Address**
   ```typescript
   // Valid format required
   user@domain.com  ✅
   invalid.email    ❌
   ```

4. **Check Logs**
   ```typescript
   const result = await sendEmail({...});
   console.log(result.error?.message);
   ```

### Styling Issues?

- Email clients render CSS differently
- Test in Gmail, Outlook, Apple Mail
- Use inline styles for reliability
- Check image URLs are absolute

### Links Not Working?

- Verify URLs are absolute (https://...)
- Check for proper URL encoding
- Test mailto: links in device
- Verify tracking doesn't break links

## Integration Checklist

- [ ] Install dependencies (`react-email`, `resend`)
- [ ] Get Resend API key
- [ ] Add API key to `.env`
- [ ] Verify domain in Resend
- [ ] Test email locally with `npm run email:preview`
- [ ] Import templates where needed
- [ ] Add error handling
- [ ] Test sending to real email
- [ ] Set up email tracking (optional)
- [ ] Document email triggers in your code

## Resources

- [React Email Docs](https://react.email/)
- [Resend Docs](https://resend.com/docs)
- [Email Best Practices](https://litmus.com/email-best-practices)
- [Live It Iconic Email Docs](./EMAIL_TEMPLATES.md)

## Support

Questions or issues?
- Check full docs: `docs/EMAIL_TEMPLATES.md`
- Review examples: `src/emails/examples.ts`
- Email: hello@liveiconic.com
