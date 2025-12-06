interface CheckAvailabilityParams {
  productId: string;
  quantity: number;
  variantId?: string;
}

interface ReserveInventoryParams {
  productId: string;
  quantity: number;
  orderId: string;
  variantId?: string;
}

interface ConfirmReservationParams {
  productId: string;
  quantity: number;
  variantId?: string;
}

export const inventoryService = {
  async checkAvailability(params: CheckAvailabilityParams): Promise<boolean> {
    try {
      const response = await fetch('/api/inventory/check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productId: params.productId,
          quantity: params.quantity,
          variantId: params.variantId,
        }),
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      return data.available || false;
    } catch (error) {
      console.error('Inventory check failed:', error);
      return false;
    }
  },

  async reserveInventory(params: ReserveInventoryParams): Promise<boolean> {
    try {
      const response = await fetch('/api/inventory/reserve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productId: params.productId,
          quantity: params.quantity,
          orderId: params.orderId,
          variantId: params.variantId,
        }),
      });

      if (!response.ok) {
        return false;
      }

      return true;
    } catch (error) {
      console.error('Inventory reservation failed:', error);
      return false;
    }
  },

  async confirmReservation(params: ConfirmReservationParams): Promise<boolean> {
    try {
      const response = await fetch('/api/inventory/confirm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productId: params.productId,
          quantity: params.quantity,
          variantId: params.variantId,
        }),
      });

      if (!response.ok) {
        return false;
      }

      return true;
    } catch (error) {
      console.error('Inventory confirmation failed:', error);
      return false;
    }
  },

  async getLowStockProducts(
    threshold: number = 5
  ): Promise<Array<{ productId: string; quantity: number }>> {
    try {
      const response = await fetch(`/api/inventory/low-stock?threshold=${threshold}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      return data.products || [];
    } catch (error) {
      console.error('Failed to get low stock products:', error);
      return [];
    }
  },
};
