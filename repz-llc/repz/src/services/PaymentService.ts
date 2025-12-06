import { supabase } from '@/integrations/supabase/client';
import { getTierKey } from '@/constants/tiers';

export interface CheckoutData {
  tier: string;
  tierPrice: number;
  billingPeriod?: 'monthly' | 'annual';
  formData: Record<string, unknown>;
  accessToken: string;
}

export interface CheckoutResult {
  url?: string;
  error?: string;
}

export class PaymentService {
  static async createCheckoutSession(data: CheckoutData): Promise<CheckoutResult> {
    try {
      const traceId = crypto.randomUUID();
      const { data: response, error } = await supabase.functions.invoke('create-checkout', {
        body: {
          tier: getTierKey(data.tier),
          billing_period: data.billingPeriod || 'monthly',
          tierPrice: data.tierPrice,
          formData: data.formData,
          trace_id: traceId,
          paymentMethodId: null
        },
        headers: {
          Authorization: `Bearer ${data.accessToken}`,
        }
      });

      if (error) {
        return { error: error.message };
      }

      if (!response.url) {
        return { error: 'Failed to create checkout session' };
      }

      return { url: response.url };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Unknown payment error'
      };
    }
  }

  static redirectToCheckout(url: string): void {
    window.open(url, '_blank');
  }

  static clearFormDraft(): void {
    localStorage.removeItem('intake-form-draft');
  }
}
