/**
 * Email Templates & Utilities
 *
 * Central export file for all email-related functionality.
 * Makes it easy to import templates and utilities throughout the app.
 */

// Components
export { EmailLayout } from './components/EmailLayout';
export { EmailHeader } from './components/EmailHeader';
export { EmailFooter } from './components/EmailFooter';

// Templates
export { WelcomeEmail } from './templates/WelcomeEmail';
export type { } from './templates/WelcomeEmail';

export { OrderConfirmation } from './templates/OrderConfirmation';
export type { } from './templates/OrderConfirmation';

export { ShippingNotification } from './templates/ShippingNotification';
export type { } from './templates/ShippingNotification';

export { PasswordReset } from './templates/PasswordReset';
export type { } from './templates/PasswordReset';

export { NewsletterSubscription } from './templates/NewsletterSubscription';
export type { } from './templates/NewsletterSubscription';

export { AbandonedCart } from './templates/AbandonedCart';
export type { } from './templates/AbandonedCart';

// Utilities
export {
  sendEmail,
  sendEmailBatch,
  trackEmailEvent,
  EmailTemplate,
} from './utils/sendEmail';
export type { } from './utils/sendEmail';
