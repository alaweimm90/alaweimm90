/**
 * Email Template Usage Examples
 *
 * This file demonstrates how to use each email template
 * in different parts of your application.
 */

import {
  WelcomeEmail,
  OrderConfirmation,
  ShippingNotification,
  PasswordReset,
  NewsletterSubscription,
  AbandonedCart,
  sendEmail,
  sendEmailBatch,
  EmailTemplate,
  trackEmailEvent,
} from './index';

/**
 * Example 1: Welcome Email
 *
 * Send this when a new user creates an account
 */
export async function sendWelcomeEmail(userEmail: string, firstName: string) {
  const result = await sendEmail({
    to: userEmail,
    subject: 'Welcome to Live It Iconic',
    react: <WelcomeEmail name={firstName} />,
  });

  if (result.success && result.data?.id) {
    await trackEmailEvent(result.data.id, EmailTemplate.WELCOME, userEmail);
    console.log('Welcome email sent to', userEmail);
  } else {
    console.error('Failed to send welcome email:', result.error?.message);
  }

  return result;
}

/**
 * Example 2: Order Confirmation Email
 *
 * Send this immediately after an order is created
 */
export async function sendOrderConfirmationEmail(order: {
  id: string;
  customerEmail: string;
  customerName: string;
  items: Array<{
    name: string;
    quantity: number;
    price: number;
    image?: string;
    sku?: string;
  }>;
  subtotal: number;
  shipping: number;
  tax?: number;
  total: number;
  address: string;
  estimatedDelivery?: string;
}) {
  const result = await sendEmail({
    to: order.customerEmail,
    subject: `Order #${order.id} Confirmed - Live It Iconic`,
    react: (
      <OrderConfirmation
        orderId={order.id}
        name={order.customerName}
        items={order.items}
        subtotal={order.subtotal}
        shipping={order.shipping}
        tax={order.tax}
        total={order.total}
        shippingAddress={order.address}
        estimatedDelivery={order.estimatedDelivery}
      />
    ),
  });

  if (result.success && result.data?.id) {
    await trackEmailEvent(
      result.data.id,
      EmailTemplate.ORDER_CONFIRMATION,
      order.customerEmail
    );
  }

  return result;
}

/**
 * Example 3: Shipping Notification Email
 *
 * Send this when an order is handed off to the carrier
 */
export async function sendShippingNotificationEmail(shipment: {
  customerEmail: string;
  customerName: string;
  orderId: string;
  trackingNumber: string;
  carrier: 'UPS' | 'FedEx' | 'DHL' | 'USPS';
  estimatedDelivery: string;
  shippingAddress?: string;
}) {
  const result = await sendEmail({
    to: shipment.customerEmail,
    subject: `Order #${shipment.orderId} Has Shipped - Live It Iconic`,
    react: (
      <ShippingNotification
        name={shipment.customerName}
        orderId={shipment.orderId}
        trackingNumber={shipment.trackingNumber}
        carrier={shipment.carrier}
        estimatedDelivery={shipment.estimatedDelivery}
        shippingAddress={shipment.shippingAddress}
      />
    ),
  });

  if (result.success && result.data?.id) {
    await trackEmailEvent(
      result.data.id,
      EmailTemplate.SHIPPING_NOTIFICATION,
      shipment.customerEmail
    );
  }

  return result;
}

/**
 * Example 4: Password Reset Email
 *
 * Send this when a user requests a password reset
 */
export async function sendPasswordResetEmail(
  userEmail: string,
  firstName: string,
  resetToken: string,
  baseUrl: string = 'https://liveiconic.com'
) {
  const resetUrl = `${baseUrl}/auth/reset-password?token=${resetToken}`;

  const result = await sendEmail({
    to: userEmail,
    subject: 'Reset Your Live It Iconic Password',
    react: (
      <PasswordReset
        name={firstName}
        resetUrl={resetUrl}
        expirationTime="24 hours"
      />
    ),
  });

  if (result.success && result.data?.id) {
    await trackEmailEvent(
      result.data.id,
      EmailTemplate.PASSWORD_RESET,
      userEmail
    );
  }

  return result;
}

/**
 * Example 5: Newsletter Subscription Email
 *
 * Send this when a user subscribes to the newsletter
 */
export async function sendNewsletterSubscriptionEmail(
  subscriberEmail: string,
  subscriberName: string,
  confirmToken: string,
  interests: string[] = ['New Releases', 'Styling Tips', 'Exclusive Offers'],
  baseUrl: string = 'https://liveiconic.com'
) {
  const confirmUrl = `${baseUrl}/newsletter/confirm?token=${confirmToken}`;

  const result = await sendEmail({
    to: subscriberEmail,
    subject: 'Confirm Your Newsletter Subscription',
    react: (
      <NewsletterSubscription
        name={subscriberName}
        confirmUrl={confirmUrl}
        interests={interests}
      />
    ),
  });

  if (result.success && result.data?.id) {
    await trackEmailEvent(
      result.data.id,
      EmailTemplate.NEWSLETTER_SUBSCRIPTION,
      subscriberEmail
    );
  }

  return result;
}

/**
 * Example 6: Abandoned Cart Email
 *
 * Send this when a cart has been abandoned for 1-2 hours
 */
export async function sendAbandonedCartEmail(cart: {
  customerId: string;
  customerEmail: string;
  customerName: string;
  items: Array<{
    id: string;
    name: string;
    price: number;
    quantity: number;
    image?: string;
    color?: string;
    size?: string;
  }>;
  total: number;
  cartToken: string;
  baseUrl?: string;
}) {
  const baseUrl = cart.baseUrl || 'https://liveiconic.com';
  const cartUrl = `${baseUrl}/cart?token=${cart.cartToken}`;

  const result = await sendEmail({
    to: cart.customerEmail,
    subject: 'You left something behind - Live It Iconic',
    react: (
      <AbandonedCart
        name={cart.customerName}
        items={cart.items}
        cartUrl={cartUrl}
        cartTotal={cart.total}
        couponCode="CART10"
        couponDiscount={10}
      />
    ),
  });

  if (result.success && result.data?.id) {
    await trackEmailEvent(
      result.data.id,
      EmailTemplate.ABANDONED_CART,
      cart.customerEmail
    );
  }

  return result;
}

/**
 * Example 7: Batch Email Sending
 *
 * Send the same email to multiple customers
 */
export async function sendBulkWelcomeEmails(
  customers: Array<{ email: string; name: string }>
) {
  const emailComponents = customers.map((customer) => ({
    to: customer.email,
    subject: 'Welcome to Live It Iconic',
    react: <WelcomeEmail name={customer.name} />,
  }));

  // Send emails sequentially with small delays
  const results = [];
  for (const email of emailComponents) {
    const result = await sendEmail(email);
    results.push(result);

    // Add a small delay between sends to avoid rate limiting
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  const successCount = results.filter((r) => r.success).length;
  console.log(`Sent ${successCount} of ${results.length} welcome emails`);

  return results;
}

/**
 * Example 8: Integration with User Auth
 *
 * Demonstrate how to hook into authentication flow
 */
export async function handleUserSignup(userData: {
  email: string;
  firstName: string;
  lastName: string;
}) {
  // 1. Create user in database
  const userId = 'user-' + Math.random().toString(36).substr(2, 9);

  // 2. Send welcome email
  const welcomeResult = await sendWelcomeEmail(userData.email, userData.firstName);

  if (!welcomeResult.success) {
    console.error('Welcome email failed:', welcomeResult.error?.message);
    // Decide if you want to proceed or reject signup
  }

  // 3. Subscribe to newsletter if opted in
  // const confirmToken = generateConfirmToken();
  // await sendNewsletterSubscriptionEmail(
  //   userData.email,
  //   userData.firstName,
  //   confirmToken
  // );

  return {
    userId,
    emailSent: welcomeResult.success,
  };
}

/**
 * Example 9: Integration with Order Management
 *
 * Demonstrate full order lifecycle email notifications
 */
export async function handleOrderLifecycle(orderData: {
  id: string;
  customerId: string;
  customerEmail: string;
  customerName: string;
  items: Array<{
    name: string;
    quantity: number;
    price: number;
    image?: string;
    sku?: string;
  }>;
  subtotal: number;
  shipping: number;
  tax?: number;
  total: number;
  address: string;
  estimatedDelivery?: string;
  trackingData?: {
    trackingNumber: string;
    carrier: 'UPS' | 'FedEx' | 'DHL' | 'USPS';
    estimatedDelivery: string;
  };
}) {
  // Step 1: Order placed
  const confirmationResult = await sendOrderConfirmationEmail(orderData);
  console.log('Order confirmation sent:', confirmationResult.success);

  // Step 2: Order ships (happens later)
  if (orderData.trackingData) {
    const shippingResult = await sendShippingNotificationEmail({
      customerEmail: orderData.customerEmail,
      customerName: orderData.customerName,
      orderId: orderData.id,
      trackingNumber: orderData.trackingData.trackingNumber,
      carrier: orderData.trackingData.carrier,
      estimatedDelivery: orderData.trackingData.estimatedDelivery,
      shippingAddress: orderData.address,
    });
    console.log('Shipping notification sent:', shippingResult.success);
  }
}

/**
 * Example 10: Error Handling Best Practices
 */
export async function sendEmailWithErrorHandling(email: {
  to: string;
  subject: string;
  template: 'welcome' | 'order' | 'shipping';
  data: Record<string, unknown>;
}) {
  try {
    // Validate input
    if (!email.to || !email.subject) {
      throw new Error('Missing required email fields');
    }

    // Build appropriate component based on template
    let reactComponent;
    switch (email.template) {
      case 'welcome':
        reactComponent = <WelcomeEmail name={email.data.name as string} />;
        break;
      // ... other cases
      default:
        throw new Error(`Unknown template: ${email.template}`);
    }

    // Send email
    const result = await sendEmail({
      to: email.to,
      subject: email.subject,
      react: reactComponent,
    });

    // Handle result
    if (!result.success) {
      if (result.error?.code === 'INVALID_EMAIL') {
        throw new Error(`Invalid email address: ${email.to}`);
      } else if (result.error?.code === 'MISSING_API_KEY') {
        throw new Error('Email service is not configured');
      } else {
        throw new Error(result.error?.message || 'Failed to send email');
      }
    }

    return result;
  } catch (error) {
    console.error('Email sending error:', error);
    // Log to error tracking service (e.g., Sentry)
    // Notify admin if critical
    throw error;
  }
}

export default {
  sendWelcomeEmail,
  sendOrderConfirmationEmail,
  sendShippingNotificationEmail,
  sendPasswordResetEmail,
  sendNewsletterSubscriptionEmail,
  sendAbandonedCartEmail,
  sendBulkWelcomeEmails,
  handleUserSignup,
  handleOrderLifecycle,
  sendEmailWithErrorHandling,
};
