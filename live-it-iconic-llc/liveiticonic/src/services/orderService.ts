import { Order, ShippingAddress } from '@/types/order';

interface CreateOrderParams {
  shippingData: ShippingAddress;
  amount: number;
  paymentMethod: string;
  cardLast4?: string;
}

export const orderService = {
  async createOrder(params: CreateOrderParams): Promise<string> {
    try {
      // Get cart items from context (would be passed as param in real implementation)
      const response = await fetch('/api/orders/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          shippingData: params.shippingData,
          amount: params.amount,
          paymentMethod: params.paymentMethod,
          status: 'pending',
          paymentStatus: 'paid',
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to create order');
      }

      const order = await response.json();
      return order.id;
    } catch (error) {
      console.error('Failed to create order:', error);
      throw error;
    }
  },

  async getOrder(orderId: string): Promise<Order | null> {
    try {
      const response = await fetch(`/api/orders/${orderId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return null;
      }

      const order = await response.json();
      return order;
    } catch (error) {
      console.error('Failed to get order:', error);
      return null;
    }
  },

  async getCustomerOrders(customerId: string): Promise<Order[]> {
    try {
      const response = await fetch(`/api/orders/customer/${customerId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      return data.orders || [];
    } catch (error) {
      console.error('Failed to get customer orders:', error);
      return [];
    }
  },

  async updateOrderStatus(orderId: string, status: string): Promise<Order | null> {
    try {
      const response = await fetch(`/api/orders/${orderId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status }),
      });

      if (!response.ok) {
        return null;
      }

      const order = await response.json();
      return order;
    } catch (error) {
      console.error('Failed to update order status:', error);
      return null;
    }
  },
};
