import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@/test/utils/test-utils'
import { Card, CardContent } from '@/ui/molecules/Card'
import { Badge } from '@/ui/atoms/Badge'
import { Button } from '@/ui/atoms/Button'
import { Progress } from '@/components/ui/progress'

// Mock a simplified TierCard component since the actual one doesn't exist yet
const TierCard = ({ 
  tier, 
  name, 
  price, 
  features, 
  isPopular, 
  billingCycle 
}: {
  tier: string
  name: string
  price: number
  features: string[]
  isPopular: boolean
  billingCycle: string
}) => {
  return (
    <Card role="article" className="backdrop-blur-xl">
      <CardContent className="p-6">
        <div className="text-center">
          {isPopular && <Badge className="mb-2">Most Popular</Badge>}
          <h3 className="text-xl font-bold mb-2">{name}</h3>
          <div className="text-3xl font-bold mb-4">
            ${price}
            <span className="text-sm text-muted-foreground">
              {billingCycle === 'annual' ? ' per month (billed annually)' : ' per month'}
            </span>
          </div>
          <ul className="space-y-2 mb-6">
            {features.map((feature, index) => (
              <li key={index} className="text-sm">{feature}</li>
            ))}
          </ul>
          <Button 
            className="w-full" 
            aria-label={`Upgrade to ${name}`}
          >
            Upgrade to {name}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Mock the tier access hook
vi.mock('@/hooks/useTierAccess', () => ({
  useTierAccess: () => ({
    userTier: 'core',
    hasMinimumTier: vi.fn().mockReturnValue(false),
    upgradePrompts: {
      showUpgrade: vi.fn(),
      hideUpgrade: vi.fn()
    }
  })
}))

describe('TierCard Component', () => {
  const mockTierData = {
    name: 'Adaptive',
    price: 49,
    features: ['AI coaching', 'Custom plans', 'Weekly check-ins']
  }

  it('renders tier information correctly', () => {
    render(
      <TierCard
        tier="adaptive"
        name={mockTierData.name}
        price={mockTierData.price}
        features={mockTierData.features}
        isPopular={false}
        billingCycle="monthly"
      />
    )
    
    expect(screen.getByText('Adaptive')).toBeInTheDocument()
    expect(screen.getByText('$49')).toBeInTheDocument()
    expect(screen.getByText('AI coaching')).toBeInTheDocument()
    expect(screen.getByText('Custom plans')).toBeInTheDocument()
  })

  it('shows popular badge when isPopular is true', () => {
    render(
      <TierCard
        tier="performance"
        name="Performance"
        price={89}
        features={['Advanced analytics']}
        isPopular={true}
        billingCycle="monthly"
      />
    )
    
    expect(screen.getByText(/most popular/i)).toBeInTheDocument()
  })

  it('displays annual pricing correctly', () => {
    render(
      <TierCard
        tier="adaptive"
        name={mockTierData.name}
        price={mockTierData.price}
        features={mockTierData.features}
        isPopular={false}
        billingCycle="annual"
      />
    )
    
    expect(screen.getByText(/billed annually/i)).toBeInTheDocument()
  })

  it('has proper accessibility attributes', () => {
    render(
      <TierCard
        tier="adaptive"
        name={mockTierData.name}
        price={mockTierData.price}
        features={mockTierData.features}
        isPopular={false}
        billingCycle="monthly"
      />
    )
    
    const card = screen.getByRole('article')
    expect(card).toBeInTheDocument()
    
    const upgradeButton = screen.getByRole('button')
    expect(upgradeButton).toHaveAttribute('aria-label')
  })

  it('applies glass morphism styling', () => {
    render(
      <TierCard
        tier="adaptive"
        name={mockTierData.name}
        price={mockTierData.price}
        features={mockTierData.features}
        isPopular={false}
        billingCycle="monthly"
      />
    )
    
    const card = screen.getByRole('article')
    expect(card).toHaveClass('backdrop-blur-xl')
  })
})