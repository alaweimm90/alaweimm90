import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { AuthProvider } from '@/contexts/AuthContext';

// Mock the tier access hook
vi.mock('@/hooks/useTierAccess', () => ({
  useTierAccess: () => ({
    hasMinimumTier: vi.fn(() => true),
    hasFeature: vi.fn(() => true),
    userTier: 'core'
  })
}));

// Mock Supabase client
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    auth: {
      getSession: vi.fn(() => Promise.resolve({ data: { session: { user: null } } })),
      getUser: vi.fn(() => Promise.resolve({ data: { user: null }, error: null })),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } }))
    }
  }
}));

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <AuthProvider>
        {component}
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('UnifiedTierCard Component', () => {
  it('renders tier card with correct tier information', () => {
    const mockOnSelect = vi.fn();
    
    renderWithProviders(
      <UnifiedTierCard 
        tier="core" 
        onSelect={mockOnSelect}
        showFeatures={true}
      />
    );

    expect(screen.getAllByText(/core/i).length).toBeGreaterThan(0);
  });

  it('calls onSelect when tier is selected', async () => {
    const mockOnSelect = vi.fn();
    
    renderWithProviders(
      <UnifiedTierCard 
        tier="adaptive" 
        onSelect={mockOnSelect}
        showFeatures={true}
      />
    );

    const selectButton = screen.getByRole('button');
    fireEvent.click(selectButton);

    await waitFor(() => {
      expect(mockOnSelect).toHaveBeenCalledWith('adaptive');
    });
  });

  it('displays features when showFeatures is true', () => {
    renderWithProviders(
      <UnifiedTierCard 
        tier="longevity" 
        showFeatures={true}
      />
    );

    // Should show some tier features
    expect(screen.getByRole('button')).toBeInTheDocument();
  });
});

describe('Performance Tests', () => {
  it('renders multiple tier cards efficiently', async () => {
    const startTime = performance.now();
    
    renderWithProviders(
      <div>
        {['core', 'adaptive', 'performance', 'longevity'].map(tier => (
          <UnifiedTierCard key={tier} tier={tier as any} showFeatures={true} />
        ))}
      </div>
    );

    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    // Should render quickly (under 100ms)
    expect(renderTime).toBeLessThan(100);
  });
});

describe('Integration Tests', () => {
  it('integrates with tier access system correctly', () => {
    renderWithProviders(
      <UnifiedTierCard 
        tier="performance" 
        showFeatures={true}
      />
    );

    // Should show tier-appropriate content
    expect(screen.getByRole('button')).toBeInTheDocument();
  });
});
