import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';

// Mock the tier data
vi.mock('@/constants/tiers', () => ({
  tierData: {
    'core': {
      name: 'Core Program',
      price: '$97',
      monthlyPrice: '$97',
      annualPrice: '$970',
      description: 'Essential performance optimization',
      features: ['Training program', 'Nutrition plan', 'Q&A access']
    },
    'adaptive': {
      name: 'Adaptive Engine',
      price: '$199',
      monthlyPrice: '$199', 
      annualPrice: '$1990',
      description: 'Advanced coaching & tracking',
      features: ['Weekly check-ins', 'Wearable integration', 'Sleep optimization'],
      popular: true
    },
    'performance': {
      name: 'Performance Suite',
      price: '$299',
      monthlyPrice: '$299',
      annualPrice: '$2990', 
      description: 'AI-powered optimization',
      features: ['AI coaching', 'Live sessions', 'Advanced analytics']
    },
    'longevity': {
      name: 'Longevity Protocol',
      price: '$449',
      monthlyPrice: '$449',
      annualPrice: '$4490',
      description: 'Complete longevity system',
      features: ['Premium protocols', 'Biomarker tracking', 'Expert access']
    }
  },
  getTierConfigByType: (tier: 'core' | 'adaptive' | 'performance' | 'longevity') => {
    const configs: Record<string, any> = {
      core: {
        displayName: 'Core Program',
        monthlyPrice: 97,
        annualPrice: 970,
        quarterlyPrice: 270,
        semiannualPrice: 540,
        features: ['Training program', 'Nutrition plan', 'Q&A access'],
        savings: { annual: 20, quarterly: 10, semiannual: 15 },
        gradient: 'from-blue-500 to-indigo-500',
        badge: undefined,
        tagline: 'Essential performance optimization',
        description: 'Essential performance optimization',
        ctaText: 'Select Core',
        isPopular: false,
      },
      adaptive: {
        displayName: 'Adaptive Engine',
        monthlyPrice: 199,
        annualPrice: 1990,
        quarterlyPrice: 570,
        semiannualPrice: 1140,
        features: ['Weekly check-ins', 'Wearable integration', 'Sleep optimization'],
        savings: { annual: 20, quarterly: 5, semiannual: 10 },
        gradient: 'from-purple-500 to-pink-500',
        badge: 'Most Popular',
        tagline: 'Advanced coaching & tracking',
        description: 'Advanced coaching & tracking',
        ctaText: 'Select Adaptive',
        isPopular: true,
      },
      performance: {
        displayName: 'Performance Suite',
        monthlyPrice: 299,
        annualPrice: 2990,
        quarterlyPrice: 870,
        semiannualPrice: 1740,
        features: ['AI coaching', 'Live sessions', 'Advanced analytics'],
        savings: { annual: 20, quarterly: 5, semiannual: 10 },
        gradient: 'from-amber-500 to-orange-500',
        badge: 'Pro',
        tagline: 'AI-powered optimization',
        description: 'AI-powered optimization',
        ctaText: 'Select Performance',
        isPopular: false,
      },
      longevity: {
        displayName: 'Longevity Protocol',
        monthlyPrice: 449,
        annualPrice: 4490,
        quarterlyPrice: 1290,
        semiannualPrice: 2580,
        features: ['Premium protocols', 'Biomarker tracking', 'Expert access'],
        savings: { annual: 25, quarterly: 5, semiannual: 12 },
        gradient: 'from-green-500 to-teal-500',
        badge: 'Elite',
        tagline: 'Complete longevity system',
        description: 'Complete longevity system',
        ctaText: 'Select Longevity',
        isPopular: false,
      },
    }
    return configs[tier]
  }
}));

// Mock the useAuth hook
const mockUseAuth = vi.fn();
vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => mockUseAuth()
}));

describe('TierCard Component', () => {
  const mockOnSelect = vi.fn();

  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: null,
      session: null,
      loading: false
    });
    mockOnSelect.mockClear();
  });

  it('displays correct pricing for each tier', () => {
    render(<UnifiedTierCard tier="performance" isCurrentTier onSelect={mockOnSelect} />);
    
    expect(screen.getByText('$299')).toBeInTheDocument();
    expect(screen.getByText('Performance Suite')).toBeInTheDocument();
  });

  it('shows upgrade prompt for locked features', () => {
    render(<UnifiedTierCard tier="core" onSelect={mockOnSelect} />);
    
    expect(screen.getAllByText(/Core Program/i).length).toBeGreaterThan(0);
  });

  it('handles tier selection correctly', () => {
    render(<UnifiedTierCard tier="core" onSelect={mockOnSelect} />);
    
    const button = screen.getByRole('button');
    fireEvent.click(button);
    
    expect(mockOnSelect).toHaveBeenCalledWith('core');
  });

  it('renders basic component structure', () => {
    const { container } = render(<UnifiedTierCard tier="core" onSelect={mockOnSelect} />);
    
    expect(container.firstChild).toBeDefined();
  });

  it('displays feature list correctly', () => {
    render(<UnifiedTierCard tier="adaptive" onSelect={mockOnSelect} />);
    
    expect(screen.getByText('Weekly check-ins')).toBeInTheDocument();
    expect(screen.getByText('Wearable integration')).toBeInTheDocument();
  });

  it('handles current plan indicator', () => {
    mockUseAuth.mockReturnValue({
      user: { id: '123' },
      session: { access_token: 'token' },
      loading: false,
      userTier: 'performance'
    });

    render(<UnifiedTierCard tier="performance" isCurrentTier onSelect={mockOnSelect} />);
    
    expect(screen.getByText(/current/i)).toBeInTheDocument();
  });
});
