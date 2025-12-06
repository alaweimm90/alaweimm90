import { Order } from '@/types/order';
import { CartItem } from '@/types/cart';

interface EmailTemplateData {
  order?: Order;
  cartItems?: CartItem[];
  trackingNumber?: string;
  [key: string]: unknown;
}

interface EmailParams {
  to: string;
  subject: string;
  template: string;
  data: EmailTemplateData;
}

export const emailService = {
  async sendOrderConfirmation(email: string, order: Order): Promise<boolean> {
    try {
      const response = await fetch('/api/email/order-confirmation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          order,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Failed to send order confirmation email:', error);
      return false;
    }
  },

  async sendShippingNotification(
    email: string,
    order: Order,
    trackingNumber: string
  ): Promise<boolean> {
    try {
      const response = await fetch('/api/email/shipping-notification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          order,
          trackingNumber,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Failed to send shipping notification:', error);
      return false;
    }
  },

  async sendAbandonedCartEmail(email: string, cartItems: CartItem[]): Promise<boolean> {
    try {
      const response = await fetch('/api/email/abandoned-cart', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          cartItems,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Failed to send abandoned cart email:', error);
      return false;
    }
  },

  async subscribeNewsletter(email: string): Promise<boolean> {
    try {
      const response = await fetch('/api/email/newsletter', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Failed to subscribe newsletter:', error);
      return false;
    }
  },
};
