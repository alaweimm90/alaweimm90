import { ShippingAddress } from '@/types/order';

interface CreatePaymentIntentParams {
  amount: number;
  currency?: string;
  orderData?: {
    items: Array<{ id: string; name: string; quantity: number; price: number }>;
    shippingAddress: ShippingAddress;
  };
}

interface PaymentIntentResponse {
  clientSecret: string;
  paymentIntentId: string;
}

interface ConfirmPaymentParams {
  paymentIntentId: string;
  paymentMethodId: string;
}

export const paymentService = {
  async createPaymentIntent(params: CreatePaymentIntentParams): Promise<PaymentIntentResponse> {
    try {
      const response = await fetch('/api/payments/create-intent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          amount: Math.round(params.amount * 100), // Convert to cents
          currency: params.currency || 'usd',
          orderData: params.orderData,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to create payment intent');
      }

      return await response.json();
    } catch (error) {
      console.error('Payment intent creation failed:', error);
      throw error;
    }
  },

  async confirmPayment(
    params: ConfirmPaymentParams
  ): Promise<{ success: boolean; orderId?: string }> {
    try {
      const response = await fetch('/api/payments/confirm-payment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          paymentIntentId: params.paymentIntentId,
          paymentMethodId: params.paymentMethodId,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Payment confirmation failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Payment confirmation failed:', error);
      throw error;
    }
  },
};
