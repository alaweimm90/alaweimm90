# Email Templates Documentation

This document provides a comprehensive guide to the Live It Iconic email template system, built with React Email and Resend.

## Overview

The email template system provides professional, responsive transactional emails for all customer communications. All templates follow the Live It Iconic brand guidelines with consistent styling, tone, and structure.

## Features

- **React Email Integration**: Build emails with React components
- **Resend API**: Reliable email delivery service
- **Responsive Design**: Mobile-friendly email templates
- **Type Safety**: Full TypeScript support
- **Batch Sending**: Send emails to multiple recipients
- **Email Tracking**: Track email delivery and opens
- **Reusable Components**: Shared layout, header, and footer components

## Installation & Setup

### Prerequisites

```bash
npm install react-email @react-email/components
npm install resend
```

### Environment Configuration

1. Add the following to your `.env` file:

```env
RESEND_API_KEY=re_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
EMAIL_FROM=hello@liveiconic.com
EMAIL_REPLY_TO=support@liveiconic.com
```

2. Get your Resend API key from [Resend Dashboard](https://resend.com/dashboard)

## Project Structure

```
src/emails/
├── components/
│   ├── EmailLayout.tsx      # Base layout wrapper
│   ├── EmailHeader.tsx       # Email header with logo
│   └── EmailFooter.tsx       # Email footer with links
├── templates/
│   ├── WelcomeEmail.tsx              # Welcome email
│   ├── OrderConfirmation.tsx         # Order confirmation
│   ├── ShippingNotification.tsx      # Shipping update
│   ├── PasswordReset.tsx             # Password reset
│   ├── NewsletterSubscription.tsx    # Newsletter welcome
│   └── AbandonedCart.tsx             # Cart recovery
└── utils/
    └── sendEmail.ts          # Email sending utility
```

## Email Templates

### 1. Welcome Email

Sent when a new customer signs up.

**File**: `src/emails/templates/WelcomeEmail.tsx`

**Props**:
```typescript
interface WelcomeEmailProps {
  name: string;        // Customer's first name
}
```

**Usage**:
```typescript
import { WelcomeEmail } from '@/emails/templates/WelcomeEmail';
import { sendEmail } from '@/emails/utils/sendEmail';

await sendEmail({
  to: 'customer@example.com',
  subject: 'Welcome to Live It Iconic',
  react: <WelcomeEmail name="John" />
});
```

**Features**:
- Personal greeting
- Brand introduction
- Link to shop
- 10% discount code (WELCOME10)
- Benefits list

---

### 2. Order Confirmation

Sent after a customer successfully places an order.

**File**: `src/emails/templates/OrderConfirmation.tsx`

**Props**:
```typescript
interface OrderConfirmationProps {
  orderId: string;              // Unique order identifier
  name: string;                 // Customer's name
  items: OrderItem[];           // Array of ordered items
  subtotal: number;             // Order subtotal
  shipping: number;             // Shipping cost
  tax?: number;                 // Tax amount (optional)
  total: number;                // Total amount
  shippingAddress: string;      // Full shipping address
  estimatedDelivery?: string;   // Estimated delivery date (optional)
}

interface OrderItem {
  name: string;                 // Product name
  quantity: number;             // Quantity ordered
  price: number;                // Price per unit
  image?: string;               // Product image URL
  sku?: string;                 // SKU/Product code
}
```

**Usage**:
```typescript
import { OrderConfirmation } from '@/emails/templates/OrderConfirmation';
import { sendEmail } from '@/emails/utils/sendEmail';

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
          quantity: 2,
          price: 59.99,
          image: 'https://...',
          sku: 'TSH-001'
        }
      ]}
      subtotal={119.98}
      shipping={10.00}
      tax={10.40}
      total={140.38}
      shippingAddress="123 Main St, New York, NY 10001"
      estimatedDelivery="November 18, 2025"
    />
  )
});
```

**Features**:
- Order number and date
- Itemized product list with images
- Order totals (subtotal, shipping, tax)
- Shipping address
- Estimated delivery date
- Next steps timeline

---

### 3. Shipping Notification

Sent when an order ships with tracking information.

**File**: `src/emails/templates/ShippingNotification.tsx`

**Props**:
```typescript
interface ShippingNotificationProps {
  name: string;                 // Customer's name
  orderId: string;              // Order number
  trackingNumber: string;       // Tracking number
  carrier: 'UPS' | 'FedEx' | 'DHL' | 'USPS';  // Shipping carrier
  estimatedDelivery: string;    // Estimated delivery date
  shippingAddress?: string;     // Shipping address (optional)
}
```

**Usage**:
```typescript
import { ShippingNotification } from '@/emails/templates/ShippingNotification';
import { sendEmail } from '@/emails/utils/sendEmail';

await sendEmail({
  to: 'customer@example.com',
  subject: 'Your Order #12345 Has Shipped',
  react: (
    <ShippingNotification
      name="John Doe"
      orderId="12345"
      trackingNumber="1Z999AA1012345678"
      carrier="UPS"
      estimatedDelivery="November 18, 2025"
      shippingAddress="123 Main St, New York, NY 10001"
    />
  )
});
```

**Features**:
- Tracking number with carrier link
- Estimated delivery date
- Delivery tips
- Shipping address
- Customer support information

---

### 4. Password Reset

Sent when a customer requests to reset their password.

**File**: `src/emails/templates/PasswordReset.tsx`

**Props**:
```typescript
interface PasswordResetProps {
  name: string;                 // Customer's name
  resetUrl: string;             // Password reset link
  expirationTime?: string;      // Link expiration time (default: '24 hours')
}
```

**Usage**:
```typescript
import { PasswordReset } from '@/emails/templates/PasswordReset';
import { sendEmail } from '@/emails/utils/sendEmail';

const resetToken = generateResetToken();
const resetUrl = `https://liveiconic.com/reset-password?token=${resetToken}`;

await sendEmail({
  to: 'customer@example.com',
  subject: 'Reset Your Live It Iconic Password',
  react: (
    <PasswordReset
      name="John Doe"
      resetUrl={resetUrl}
      expirationTime="24 hours"
    />
  )
});
```

**Features**:
- Clear call-to-action button
- Direct link for email clients that don't support buttons
- Expiration time notice
- Security information
- "Didn't request this?" section
- Password best practices
- Support contact information

---

### 5. Newsletter Subscription

Sent to confirm newsletter signup.

**File**: `src/emails/templates/NewsletterSubscription.tsx`

**Props**:
```typescript
interface NewsletterSubscriptionProps {
  name: string;                 // Subscriber's name
  confirmUrl: string;           // Confirmation link
  interests?: string[];         // Topics of interest (optional)
}
```

**Usage**:
```typescript
import { NewsletterSubscription } from '@/emails/templates/NewsletterSubscription';
import { sendEmail } from '@/emails/utils/sendEmail';

const confirmToken = generateConfirmToken();
const confirmUrl = `https://liveiconic.com/confirm-newsletter?token=${confirmToken}`;

await sendEmail({
  to: 'customer@example.com',
  subject: 'Confirm Your Newsletter Subscription',
  react: (
    <NewsletterSubscription
      name="John Doe"
      confirmUrl={confirmUrl}
      interests={['New Releases', 'Styling Tips', 'Exclusive Offers']}
    />
  )
});
```

**Features**:
- Confirmation link
- List of interests/topics
- What subscribers will receive
- Social media links
- Email frequency information
- Preference management options
- Unsubscribe option

---

### 6. Abandoned Cart

Sent to remind customers of items left in their cart.

**File**: `src/emails/templates/AbandonedCart.tsx`

**Props**:
```typescript
interface AbandonedCartProps {
  name: string;                 // Customer's name
  items: CartItem[];            // Items in cart
  cartUrl: string;              // Link to resume shopping
  cartTotal: number;            // Total cart value
  couponCode?: string;          // Discount code (default: 'CART10')
  couponDiscount?: number;      // Discount percentage (default: 10)
}

interface CartItem {
  id: string;                   // Item ID
  name: string;                 // Product name
  price: number;                // Price per unit
  quantity: number;             // Quantity in cart
  image?: string;               // Product image URL
  color?: string;               // Product color
  size?: string;                // Product size
}
```

**Usage**:
```typescript
import { AbandonedCart } from '@/emails/templates/AbandonedCart';
import { sendEmail } from '@/emails/utils/sendEmail';

await sendEmail({
  to: 'customer@example.com',
  subject: 'You Left Something Behind',
  react: (
    <AbandonedCart
      name="John Doe"
      items={[
        {
          id: '1',
          name: 'Premium T-Shirt',
          price: 59.99,
          quantity: 1,
          image: 'https://...',
          color: 'Black',
          size: 'M'
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

**Features**:
- Product images with specifications
- Cart total
- Limited-time discount offer
- Clear return-to-cart button
- Why shop with us section
- Urgency messaging
- Support contact information

---

## Email Sending Utility

### Basic Usage

The `sendEmail` function handles all email sending logic.

```typescript
import { sendEmail, EmailTemplate } from '@/emails/utils/sendEmail';

// Send a single email
const result = await sendEmail({
  to: 'customer@example.com',
  subject: 'Welcome!',
  react: <WelcomeEmail name="John" />,
  replyTo: 'support@liveiconic.com'  // Optional
});

if (result.success) {
  console.log('Email sent with ID:', result.data?.id);
} else {
  console.error('Email failed:', result.error?.message);
}
```

### Batch Sending

Send the same email to multiple recipients:

```typescript
import { sendEmailBatch } from '@/emails/utils/sendEmail';

const results = await sendEmailBatch({
  to: ['customer1@example.com', 'customer2@example.com'],
  subject: 'New Collection Released',
  react: <NewsletterEmail />,
});

// results is an array of SendEmailResponse objects
results.forEach((result, index) => {
  if (result.success) {
    console.log(`Email ${index + 1} sent successfully`);
  }
});
```

### Response Types

```typescript
interface SendEmailResponse {
  success: boolean;
  data?: {
    id: string;              // Resend email ID
    from: string;            // From address
    to: string;              // To address
    created_at: string;      // Timestamp
  };
  error?: {
    message: string;         // Error message
    code?: string;           // Error code
  };
}
```

### Error Handling

```typescript
try {
  const result = await sendEmail({
    to: email,
    subject: 'Welcome',
    react: <WelcomeEmail name="John" />
  });

  if (!result.success) {
    // Handle specific error codes
    if (result.error?.code === 'INVALID_EMAIL') {
      console.error('Invalid email address');
    } else if (result.error?.code === 'MISSING_API_KEY') {
      console.error('Resend API key not configured');
    } else {
      console.error('Email send failed:', result.error?.message);
    }
  }
} catch (error) {
  console.error('Unexpected error:', error);
}
```

### Email Tracking

Track email delivery events for analytics:

```typescript
import { trackEmailEvent, EmailTemplate } from '@/emails/utils/sendEmail';

const result = await sendEmail({
  to: 'customer@example.com',
  subject: 'Welcome',
  react: <WelcomeEmail name="John" />
});

if (result.success && result.data?.id) {
  await trackEmailEvent(
    result.data.id,
    EmailTemplate.WELCOME,
    'customer@example.com'
  );
}
```

## Base Components

### EmailLayout

The base wrapper for all emails. Provides consistent structure and styling.

```typescript
import { EmailLayout } from '@/emails/components/EmailLayout';

<EmailLayout preview="Email preview text">
  {/* Email content */}
</EmailLayout>
```

**Props**:
- `preview`: Text shown in email client preview (recommended 50 chars)
- `children`: Email content (React elements)

### EmailHeader

Company logo and branding at the top of emails.

```typescript
import { EmailHeader } from '@/emails/components/EmailHeader';

<EmailHeader />
```

No props required. Displays the Live It Iconic logo and tagline.

### EmailFooter

Company information, social links, and unsubscribe option at the bottom.

```typescript
import { EmailFooter } from '@/emails/components/EmailFooter';

<EmailFooter />
```

No props required. Displays:
- Social media links
- Contact email
- Unsubscribe link
- Copyright information

## Styling & Customization

### Brand Colors

```typescript
const colors = {
  primary: '#C1A060',        // Gold accent color
  dark: '#0B0B0C',           // Dark background
  text: '#0B0B0C',           // Primary text
  textSecondary: '#5C6270',  // Secondary text
  border: '#E6E9EF',         // Borders
  background: '#f8f8f8',     // Light background
  success: '#4CAF50',        // Success indicator
};
```

### Typography

- **Headers**: 28px bold (H1), 18px bold (H3)
- **Body text**: 16px regular
- **Secondary text**: 13-14px
- **Font**: System fonts (-apple-system, Segoe UI, sans-serif)

### Responsive Design

All templates are mobile-responsive. Email clients automatically stack content on small screens. Test with:
- iOS Mail
- Gmail
- Outlook
- Apple Mail

## Integration Guide

### 1. Authentication Flows

**Sign Up Flow**:
```typescript
// After user creates account
await sendEmail({
  to: userEmail,
  subject: 'Welcome to Live It Iconic',
  react: <WelcomeEmail name={userFirstName} />
});
```

**Password Reset Flow**:
```typescript
// After user requests password reset
const resetToken = generateResetToken(userId, expiresIn: '24h');
const resetUrl = `${APP_URL}/reset-password?token=${resetToken}`;

await sendEmail({
  to: userEmail,
  subject: 'Reset Your Password',
  react: <PasswordReset name={userName} resetUrl={resetUrl} />
});
```

### 2. Order Processing

**After Order Placement**:
```typescript
// After order is created in database
await sendEmail({
  to: orderEmail,
  subject: `Order #${orderId} Confirmed`,
  react: (
    <OrderConfirmation
      orderId={order.id}
      name={order.customerName}
      items={order.items}
      subtotal={order.subtotal}
      shipping={order.shippingCost}
      tax={order.tax}
      total={order.total}
      shippingAddress={formatAddress(order.shippingAddress)}
      estimatedDelivery={calculateEstimatedDelivery()}
    />
  )
});
```

**When Order Ships**:
```typescript
// After order is handed off to carrier
await sendEmail({
  to: orderEmail,
  subject: `Your Order #${orderId} Has Shipped`,
  react: (
    <ShippingNotification
      name={order.customerName}
      orderId={order.id}
      trackingNumber={shipment.trackingNumber}
      carrier={shipment.carrier}
      estimatedDelivery={shipment.estimatedDelivery}
      shippingAddress={formatAddress(order.shippingAddress)}
    />
  )
});
```

### 3. Marketing Campaigns

**Newsletter Signup**:
```typescript
// After newsletter subscription
const confirmToken = generateToken();
const confirmUrl = `${APP_URL}/confirm?token=${confirmToken}`;

await sendEmail({
  to: emailAddress,
  subject: 'Confirm Your Subscription',
  react: (
    <NewsletterSubscription
      name={customerName}
      confirmUrl={confirmUrl}
      interests={selectedInterests}
    />
  )
});
```

**Abandoned Cart Recovery** (send after 1-2 hours):
```typescript
// Check for abandoned carts periodically
const abandonedCarts = await fetchAbandonedCarts();

for (const cart of abandonedCarts) {
  await sendEmail({
    to: cart.customerEmail,
    subject: 'You Left Something Behind',
    react: (
      <AbandonedCart
        name={cart.customerName}
        items={cart.items}
        cartUrl={`${APP_URL}/cart/${cart.id}`}
        cartTotal={cart.total}
      />
    )
  });
}
```

## Testing

### Local Testing

Preview all email templates locally:

```bash
npm run email:preview
```

This opens a development server at `http://localhost:3000` where you can view and test all email templates.

### Testing with Real Data

1. Use your test/staging Resend account
2. Send test emails to yourself
3. Check email client rendering
4. Verify links work correctly

### Unit Testing

```typescript
import { render } from '@react-email/components';
import { WelcomeEmail } from '@/emails/templates/WelcomeEmail';

test('WelcomeEmail renders correctly', () => {
  const html = render(<WelcomeEmail name="John" />);
  expect(html).toContain('Welcome to Live It Iconic');
  expect(html).toContain('John');
});
```

## Best Practices

### 1. Email Content

- Keep subject lines under 50 characters
- Use clear, benefit-focused preview text
- Include a clear call-to-action
- Ensure proper email-to-web experience
- Use personalization tokens appropriately

### 2. Performance

- Optimize images (use CDN)
- Keep HTML under 102KB
- Minimize CSS styles
- Use alt text for all images
- Test with multiple email clients

### 3. Compliance

- Always include unsubscribe link
- Provide physical mailing address (in footer)
- Honor unsubscribe requests immediately
- Don't use misleading subject lines
- Include company name/identifier

### 4. Security

- Validate email addresses before sending
- Use HTTPS for all links
- Don't include sensitive data in emails
- Implement rate limiting for email sends
- Log all email delivery for audit trails

## Troubleshooting

### Emails Not Sending

1. **Check API Key**: Verify `RESEND_API_KEY` is set correctly
2. **Verify Email Address**: Ensure recipient email is valid
3. **Domain Verification**: Confirm domain is verified in Resend
4. **Check Resend Quota**: Verify API limit hasn't been exceeded
5. **Review Logs**: Check console for error messages

### Styling Issues

1. **Test in Multiple Clients**: Gmail, Outlook, Apple Mail render differently
2. **Inline Styles**: Some properties only work when inlined
3. **Images**: Use absolute URLs and test image loading
4. **Mobile**: Test responsive layout on mobile devices

### Links Not Working

1. **Verify URLs**: Ensure all URLs are absolute and valid
2. **Track Links**: Check if link tracking is interfering
3. **UTM Parameters**: Ensure they're properly formatted
4. **Mailto Links**: Use proper mailto: format

## Resources

- [React Email Documentation](https://react.email/)
- [Resend Documentation](https://resend.com/docs)
- [Email Best Practices](https://litmus.com/email-best-practices)
- [MJML Email Framework](https://mjml.io/) (alternative)
- [Email Standards Project](https://www.campaignmonitor.com/css/)

## Support

For issues or questions:
- Email: hello@liveiconic.com
- Documentation: https://liveiconic.com/docs
- GitHub Issues: Report bugs and feature requests
