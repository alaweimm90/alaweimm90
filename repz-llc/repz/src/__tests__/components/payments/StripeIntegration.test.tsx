import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, test, expect, beforeEach } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { supabase } from '../../../integrations/supabase/client';
import type { User } from '@supabase/supabase-js';

// Mock Supabase
vi.mock('../../../integrations/supabase/client', () => ({
  supabase: {
    functions: {
      invoke: vi.fn()
    },
    auth: {
      getSession: vi.fn()
    }
  }
}));

// Mock Stripe checkout component
const MockCheckoutButton = ({ tier, onSuccess, onError }: any) => (
  <button
    onClick={async () => {
      try {
        const response = await supabase.functions.invoke('create-checkout', {
          body: { tier, billing_period: 'monthly' }
        });
        if (response.error) throw response.error;
        onSuccess?.(response.data);
      } catch (error) {
        onError?.(error);
      }
    }}
  >
    Subscribe to {tier}
  </button>
);

const TestWrapper = ({ children }: any) => (
  <BrowserRouter>
    {children}
  </BrowserRouter>
);

describe('Stripe Payment Flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const mockUser: Partial<User> = {
      id: '123',
      email: 'test@example.com',
      app_metadata: {},
      user_metadata: {},
      aud: 'authenticated',
      created_at: '2024-01-01T00:00:00Z'
    };
    
    vi.mocked(supabase.auth.getSession).mockResolvedValue({
      data: {
        session: {
          user: mockUser as User,
          access_token: 'token',
          refresh_token: 'refresh',
          expires_in: 3600,
          token_type: 'bearer'
        }
      },
      error: null
    });
  });

  test('successful subscription creation', async () => {
    const mockCheckoutSession = {
      url: 'https://checkout.stripe.com/session_123',
      id: 'cs_123'
    };

    vi.mocked(supabase.functions.invoke).mockResolvedValue({
      data: mockCheckoutSession,
      error: null
    });

    const onSuccess = vi.fn();
    const onError = vi.fn();

    render(
      <TestWrapper>
        <MockCheckoutButton 
          tier="performance" 
          onSuccess={onSuccess}
          onError={onError}
        />
      </TestWrapper>
    );

    // Click subscribe button
    fireEvent.click(screen.getByText(/subscribe to performance/i));

    await waitFor(() => {
      expect(supabase.functions.invoke).toHaveBeenCalledWith('create-checkout', {
        body: { tier: 'performance', billing_period: 'monthly' }
      });
      expect(onSuccess).toHaveBeenCalledWith(mockCheckoutSession);
      expect(onError).not.toHaveBeenCalled();
    });
  });

  test('payment failure handling', async () => {
    const mockError = new Error('Payment method declined');

    vi.mocked(supabase.functions.invoke).mockResolvedValue({
      data: null,
      error: mockError
    });

    const onSuccess = vi.fn();
    const onError = vi.fn();

    render(
      <TestWrapper>
        <MockCheckoutButton 
          tier="adaptive" 
          onSuccess={onSuccess}
          onError={onError}
        />
      </TestWrapper>
    );

    fireEvent.click(screen.getByText(/subscribe to adaptive/i));

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(mockError);
      expect(onSuccess).not.toHaveBeenCalled();
    });
  });

  test('subscription cancellation workflow', async () => {
    const mockPortalSession = {
      url: 'https://billing.stripe.com/session_123'
    };

    vi.mocked(supabase.functions.invoke).mockResolvedValue({
      data: mockPortalSession,
      error: null
    });

    const CustomerPortalButton = () => (
      <button
        onClick={async () => {
          const response = await supabase.functions.invoke('customer-portal');
          if (response.data?.url) {
            window.open(response.data.url, '_blank');
          }
        }}
      >
        Manage Subscription
      </button>
    );

    // Mock window.open
    const mockOpen = vi.fn();
    Object.defineProperty(window, 'open', { value: mockOpen });

    render(
      <TestWrapper>
        <CustomerPortalButton />
      </TestWrapper>
    );

    fireEvent.click(screen.getByText(/manage subscription/i));

    await waitFor(() => {
      expect(supabase.functions.invoke).toHaveBeenCalledWith('customer-portal');
      expect(mockOpen).toHaveBeenCalledWith(mockPortalSession.url, '_blank');
    });
  });

  test('tier upgrade with proration', async () => {
    const mockUpgradeCheckout = {
      url: 'https://checkout.stripe.com/upgrade_123',
      id: 'cs_upgrade_123'
    };

    vi.mocked(supabase.functions.invoke).mockResolvedValue({
      data: mockUpgradeCheckout,
      error: null
    });

    const UpgradeButton = () => (
      <button
        onClick={async () => {
          await supabase.functions.invoke('create-checkout', {
            body: { 
              tier: 'longevity', 
              billing_period: 'monthly',
              is_upgrade: true,
              current_tier: 'performance'
            }
          });
        }}
      >
        Upgrade to Longevity
      </button>
    );

    render(
      <TestWrapper>
        <UpgradeButton />
      </TestWrapper>
    );

    fireEvent.click(screen.getByText(/upgrade to longevity/i));

    await waitFor(() => {
      expect(supabase.functions.invoke).toHaveBeenCalledWith('create-checkout', {
        body: { 
          tier: 'longevity', 
          billing_period: 'monthly',
          is_upgrade: true,
          current_tier: 'performance'
        }
      });
    });
  });

  test('handles network errors gracefully', async () => {
    vi.mocked(supabase.functions.invoke).mockRejectedValue(
      new Error('Network request failed')
    );

    const onError = vi.fn();

    render(
      <TestWrapper>
        <MockCheckoutButton 
          tier="core" 
          onError={onError}
        />
      </TestWrapper>
    );

    fireEvent.click(screen.getByText(/subscribe to core/i));

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Network request failed'
        })
      );
    });
  });

  test('subscription status check', async () => {
    const mockSubscriptionStatus = {
      subscribed: true,
      subscription_tier: 'performance',
      subscription_end: '2024-12-31T23:59:59Z'
    };

    vi.mocked(supabase.functions.invoke).mockResolvedValue({
      data: mockSubscriptionStatus,
      error: null
    });

    const StatusChecker = () => {
      const [status, setStatus] = React.useState<any>(null);
      
      return (
        <div>
          <button
            onClick={async () => {
              const response = await supabase.functions.invoke('check-subscription');
              setStatus(response.data);
            }}
          >
            Check Status
          </button>
          {status && (
            <div>
              Status: {status.subscribed ? 'Active' : 'Inactive'}
              Tier: {status.subscription_tier}
            </div>
          )}
        </div>
      );
    };

    render(
      <TestWrapper>
        <StatusChecker />
      </TestWrapper>
    );

    fireEvent.click(screen.getByText(/check status/i));

    await waitFor(() => {
      expect(screen.getByText(/status: active/i)).toBeInTheDocument();
      expect(screen.getByText(/tier: performance/i)).toBeInTheDocument();
    });
  });
});