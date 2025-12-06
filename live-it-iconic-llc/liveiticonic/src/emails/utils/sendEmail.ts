import { Resend } from 'resend';
import { render } from '@react-email/components';

const resend = new Resend(process.env.RESEND_API_KEY);

interface SendEmailOptions {
  to: string;
  subject: string;
  react: React.ReactElement;
  replyTo?: string;
}

interface SendEmailResponse {
  success: boolean;
  data?: {
    id: string;
    from: string;
    to: string;
    created_at: string;
  };
  error?: {
    message: string;
    code?: string;
  };
}

/**
 * Send an email using React Email and Resend
 * @param options - Email sending options
 * @returns Promise with success status and response data
 */
export async function sendEmail({
  to,
  subject,
  react,
  replyTo,
}: SendEmailOptions): Promise<SendEmailResponse> {
  // Validate email address
  if (!to || !isValidEmail(to)) {
    return {
      success: false,
      error: {
        message: 'Invalid recipient email address',
        code: 'INVALID_EMAIL',
      },
    };
  }

  // Check if API key is configured
  if (!process.env.RESEND_API_KEY) {
    console.error('RESEND_API_KEY is not configured');
    return {
      success: false,
      error: {
        message: 'Email service is not configured',
        code: 'MISSING_API_KEY',
      },
    };
  }

  try {
    // Convert React component to HTML
    const html = render(react);

    // Send email via Resend
    const response = await resend.emails.send({
      from: 'Live It Iconic <hello@liveiconic.com>',
      to,
      subject,
      html,
      ...(replyTo && { reply_to: replyTo }),
    });

    if (response.error) {
      console.error(`Error sending email to ${to}:`, response.error);
      return {
        success: false,
        error: {
          message: response.error.message,
          code: 'SEND_FAILED',
        },
      };
    }

    console.log(`Email sent successfully to ${to} with ID: ${response.data?.id}`);
    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error(`Exception sending email to ${to}:`, errorMessage);
    return {
      success: false,
      error: {
        message: errorMessage,
        code: 'SEND_ERROR',
      },
    };
  }
}

/**
 * Send emails to multiple recipients
 * @param options - Email sending options with array of recipients
 * @returns Promise with array of results
 */
export async function sendEmailBatch(
  options: Omit<SendEmailOptions, 'to'> & { to: string[] }
): Promise<SendEmailResponse[]> {
  const results = await Promise.all(
    options.to.map((recipient) =>
      sendEmail({
        ...options,
        to: recipient,
      })
    )
  );
  return results;
}

/**
 * Simple email validation
 * @param email - Email address to validate
 * @returns true if email is valid, false otherwise
 */
function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Email template types for better type safety
 */
export enum EmailTemplate {
  WELCOME = 'welcome',
  ORDER_CONFIRMATION = 'order_confirmation',
  SHIPPING_NOTIFICATION = 'shipping_notification',
  PASSWORD_RESET = 'password_reset',
  NEWSLETTER_SUBSCRIPTION = 'newsletter_subscription',
  ABANDONED_CART = 'abandoned_cart',
}

/**
 * Track email send events for analytics
 * @param emailId - Resend email ID
 * @param template - Email template type
 * @param recipient - Recipient email address
 */
export async function trackEmailEvent(
  emailId: string,
  template: EmailTemplate,
  recipient: string
): Promise<void> {
  try {
    // Log to console in development
    if (process.env.NODE_ENV !== 'production') {
      console.log(`Email tracked - ID: ${emailId}, Template: ${template}, Recipient: ${recipient}`);
    }

    // TODO: Implement analytics tracking
    // This could integrate with your analytics service (e.g., PostHog, Mixpanel, etc.)
  } catch (error) {
    console.error('Error tracking email event:', error);
  }
}
