import React from 'react'
import { cn } from '@/lib/utils'
import { motion } from 'framer-motion'
import { Badge } from '@/ui/atoms/Badge'
import { EnhancedCard, EnhancedHeading, EnhancedText, EnhancedBadge } from './enhanced-design-system'

// ============================================================================
// ENHANCED TIER DISPLAY SYSTEM
// Beautiful tier cards with perfect visual hierarchy
// ============================================================================

interface TierDisplayProps {
  tier: 'core' | 'adaptive' | 'performance' | 'longevity'
  title: string
  price: string
  period?: string
  features: string[]
  isPopular?: boolean
  isRecommended?: boolean
  className?: string
}

export const EnhancedTierCard: React.FC<TierDisplayProps> = ({
  tier,
  title,
  price,
  period = 'month',
  features,
  isPopular = false,
  isRecommended = false,
  className
}) => {
  const tierConfig = {
    core: {
      gradient: 'from-tier-core/20 to-tier-core/10',
      border: 'border-tier-core/30',
      accent: 'tier-core',
      glow: 'shadow-lg shadow-tier-core/20'
    },
    adaptive: {
      gradient: 'from-tier-adaptive/20 to-tier-adaptive/10', 
      border: 'border-tier-adaptive/30',
      accent: 'tier-adaptive',
      glow: 'shadow-lg shadow-tier-adaptive/20'
    },
    performance: {
      gradient: 'from-tier-performance/20 to-tier-performance/10',
      border: 'border-tier-performance/30', 
      accent: 'tier-performance',
      glow: 'shadow-lg shadow-tier-performance/20'
    },
    longevity: {
      gradient: 'from-tier-longevity/20 to-tier-longevity/10',
      border: 'border-tier-longevity/30',
      accent: 'tier-longevity', 
      glow: 'shadow-lg shadow-tier-longevity/20'
    }
  }
  
  const config = tierConfig[tier]
  
  return (
    <motion.div
      className={cn(
        "relative group",
        className
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, scale: 1.02 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      {/* Popular/Recommended Badge */}
      {(isPopular || isRecommended) && (
        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 z-10">
          <EnhancedBadge 
            variant="tier" 
            tier={tier}
            glow
            pulse
          >
            {isRecommended ? 'Recommended' : 'Most Popular'}
          </EnhancedBadge>
        </div>
      )}
      
      <EnhancedCard
        variant="glass"
        className={cn(
          "p-8 h-full bg-gradient-to-br",
          config.gradient,
          config.border,
          config.glow,
          "hover:shadow-xl transition-all duration-500",
          isPopular && "ring-2 ring-tier-adaptive/50",
          isRecommended && "ring-2 ring-tier-performance/50"
        )}
      >
        {/* Header */}
        <div className="text-center mb-8">
          <EnhancedHeading 
            variant="subtitle" 
            className={`text-${config.accent} mb-2`}
          >
            {title}
          </EnhancedHeading>
          
          <div className="flex items-baseline justify-center gap-1">
            <span className={`text-4xl font-montserrat font-bold text-${config.accent}`}>
              {price}
            </span>
            <EnhancedText variant="muted" size="sm">
              /{period}
            </EnhancedText>
          </div>
        </div>
        
        {/* Features */}
        <div className="space-y-4 mb-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="flex items-start gap-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className={`w-2 h-2 rounded-full bg-${config.accent} mt-2 flex-shrink-0`} />
              <EnhancedText variant="elegant" size="sm">
                {feature}
              </EnhancedText>
            </motion.div>
          ))}
        </div>
        
        {/* CTA Button */}
        <div className="mt-auto">
          <motion.button
            className={cn(
              "w-full py-4 rounded-xl font-montserrat font-semibold",
              "bg-gradient-to-r transition-all duration-300",
              `from-${config.accent}/20 to-${config.accent}/10`,
              `border border-${config.accent}/30 text-${config.accent}`,
              `hover:from-${config.accent}/30 hover:to-${config.accent}/20`,
              `hover:border-${config.accent}/50 hover:shadow-lg hover:shadow-${config.accent}/20`,
              "backdrop-blur-sm"
            )}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Get Started
          </motion.button>
        </div>
      </EnhancedCard>
    </motion.div>
  )
}

// Tier Comparison Table
interface TierComparisonProps {
  tiers: Array<{
    tier: 'core' | 'adaptive' | 'performance' | 'longevity'
    name: string
    price: string
    features: Record<string, boolean | string>
  }>
  featureCategories: Array<{
    category: string
    features: string[]
  }>
}

export const TierComparison: React.FC<TierComparisonProps> = ({
  tiers,
  featureCategories
}) => {
  return (
    <EnhancedCard variant="frost" className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          {/* Header */}
          <thead>
            <tr className="border-b border-white/10">
              <th className="text-left p-6">
                <EnhancedText variant="premium" weight="semibold">
                  Features
                </EnhancedText>
              </th>
              {tiers.map(tier => (
                <th key={tier.tier} className="text-center p-6 min-w-[200px]">
                  <EnhancedHeading variant="subtitle" tier={tier.tier} gradient>
                    {tier.name}
                  </EnhancedHeading>
                  <EnhancedText variant="muted" className="mt-1">
                    {tier.price}
                  </EnhancedText>
                </th>
              ))}
            </tr>
          </thead>
          
          {/* Body */}
          <tbody>
            {featureCategories.map(category => (
              <React.Fragment key={category.category}>
                <tr className="border-b border-white/5">
                  <td colSpan={tiers.length + 1} className="p-4">
                    <EnhancedText variant="premium" weight="medium" className="text-repz-orange">
                      {category.category}
                    </EnhancedText>
                  </td>
                </tr>
                {category.features.map(feature => (
                  <tr key={feature} className="border-b border-white/5 hover:bg-white/5">
                    <td className="p-4">
                      <EnhancedText variant="elegant">
                        {feature}
                      </EnhancedText>
                    </td>
                    {tiers.map(tier => (
                      <td key={tier.tier} className="p-4 text-center">
                        {renderFeatureValue(tier.features[feature])}
                      </td>
                    ))}
                  </tr>
                ))}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </EnhancedCard>
  )
}

function renderFeatureValue(value: boolean | string | undefined) {
  if (typeof value === 'boolean') {
    return (
      <div className="flex justify-center">
        {value ? (
          <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-green-400" />
          </div>
        ) : (
          <div className="w-6 h-6 rounded-full bg-gray-500/20 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-gray-500" />
          </div>
        )}
      </div>
    )
  }
  
  if (typeof value === 'string') {
    return (
      <EnhancedBadge variant="feature">
        {value}
      </EnhancedBadge>
    )
  }
  
  return (
    <EnhancedText variant="muted">
      —
    </EnhancedText>
  )
}

// Tier Status Indicator
interface TierStatusProps {
  currentTier: 'core' | 'adaptive' | 'performance' | 'longevity'
  className?: string
}

export const TierStatus: React.FC<TierStatusProps> = ({
  currentTier,
  className
}) => {
  const tierNames = {
    core: 'Core',
    adaptive: 'Adaptive', 
    performance: 'Performance',
    longevity: 'Longevity'
  }
  
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <EnhancedBadge variant="tier" tier={currentTier} glow>
        {tierNames[currentTier]}
      </EnhancedBadge>
      <EnhancedText variant="muted" size="sm">
        Current Plan
      </EnhancedText>
    </div>
  )
}