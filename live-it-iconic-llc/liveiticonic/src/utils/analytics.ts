// Analytics utility for tracking events
// In production, integrate with Segment, Google Analytics, or similar

type AnalyticsValue = string | number | boolean | null | undefined;
type AnalyticsProperties = Record<string, AnalyticsValue | AnalyticsValue[]>;

interface AnalyticsEvent {
  event: string;
  properties?: AnalyticsProperties;
}

class Analytics {
  private initialized = false;

  init() {
    // In production, initialize analytics SDK here
    this.initialized = true;
    console.log('Analytics initialized');
  }

  track(event: string, properties?: AnalyticsProperties) {
    if (!this.initialized) {
      this.init();
    }

    // In production, send to analytics service
    console.log('Analytics Event:', event, properties);

    // Store in localStorage for debugging
    const events = JSON.parse(localStorage.getItem('analytics_events') || '[]');
    events.push({
      event,
      properties,
      timestamp: new Date().toISOString(),
    });
    localStorage.setItem('analytics_events', JSON.stringify(events.slice(-100))); // Keep last 100
  }

  trackPageView(path: string) {
    this.track('Page View', { path });
  }

  trackPurchase(
    orderId: string,
    total: number,
    items: Array<{ id: string; name: string; quantity: number; price: number }>
  ) {
    this.track('Purchase', {
      orderId,
      total,
      revenue: total,
      items,
      currency: 'USD',
    });
  }

  trackAddToCart(productId: string, quantity: number, price: number, productName?: string) {
    this.track('Add to Cart', {
      productId,
      quantity,
      price,
      productName,
      value: price * quantity,
    });
  }

  trackRemoveFromCart(productId: string, quantity: number) {
    this.track('Remove from Cart', {
      productId,
      quantity,
    });
  }

  trackProductView(productId: string, name: string, price: number) {
    this.track('View Product', {
      productId,
      name,
      price,
    });
  }

  trackCheckoutStarted(
    items: Array<{ id: string; quantity: number; price: number }>,
    total: number
  ) {
    this.track('Checkout Started', {
      items,
      total,
      itemCount: items.reduce((sum, item) => sum + item.quantity, 0),
    });
  }

  trackCheckoutStep(step: string, stepNumber: number) {
    this.track('Checkout Step', {
      step,
      stepNumber,
    });
  }
}

export const analytics = new Analytics();

// Initialize on module load
analytics.init();

export const trackEvents = {
  viewProduct: (productId: string, name: string, price: number) => {
    analytics.trackProductView(productId, name, price);
  },
  addToCart: (productId: string, quantity: number, price: number, productName?: string) => {
    analytics.trackAddToCart(productId, quantity, price, productName);
  },
  removeFromCart: (productId: string, quantity: number) => {
    analytics.trackRemoveFromCart(productId, quantity);
  },
  purchase: (
    orderId: string,
    total: number,
    items: Array<{ id: string; name: string; quantity: number; price: number }>
  ) => {
    analytics.trackPurchase(orderId, total, items);
  },
  checkoutStarted: (
    items: Array<{ id: string; quantity: number; price: number }>,
    total: number
  ) => {
    analytics.trackCheckoutStarted(items, total);
  },
  checkoutStep: (step: string, stepNumber: number) => {
    analytics.trackCheckoutStep(step, stepNumber);
  },
};
