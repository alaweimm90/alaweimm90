import React from 'react'
import { cn } from '@/lib/utils'
import { motion } from 'framer-motion'
import { EnhancedCard, EnhancedHeading, EnhancedText, EnhancedBadge } from './enhanced-design-system'

// ============================================================================
// ENHANCED DASHBOARD COMPONENTS
// Professional, beautiful dashboard elements
// ============================================================================

// Metric Card with Beautiful Animations
interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  change?: number
  changeType?: 'positive' | 'negative' | 'neutral'
  icon?: React.ReactNode
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  variant?: 'default' | 'compact' | 'hero'
  className?: string
}

export const EnhancedMetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  change,
  changeType = 'neutral',
  icon,
  tier,
  variant = 'default',
  className
}) => {
  const changeColors = {
    positive: 'text-green-400',
    negative: 'text-red-400', 
    neutral: 'text-white/60'
  }
  
  const changeSymbol = {
    positive: '+',
    negative: '',
    neutral: ''
  }
  
  const variants = {
    default: 'p-6',
    compact: 'p-4',
    hero: 'p-8'
  }
  
  return (
    <EnhancedCard 
      variant="glass" 
      tier={tier}
      animation="float"
      className={cn(variants[variant], className)}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <EnhancedText variant="muted" size="sm" className="mb-1">
            {title}
          </EnhancedText>
          <motion.div
            className="flex items-baseline gap-2"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <EnhancedHeading 
              variant={variant === 'hero' ? 'hero' : 'subtitle'}
              tier={tier}
              gradient
            >
              {value}
            </EnhancedHeading>
            {change !== undefined && (
              <span className={cn('text-sm font-medium', changeColors[changeType])}>
                {changeSymbol[changeType]}{change}%
              </span>
            )}
          </motion.div>
        </div>
        
        {icon && (
          <div className={cn(
            "flex items-center justify-center rounded-xl",
            tier ? `bg-tier-${tier}/20 text-tier-${tier}` : 'bg-surface-frost text-white/70',
            variant === 'compact' ? 'w-8 h-8' : 'w-12 h-12'
          )}>
            {icon}
          </div>
        )}
      </div>
      
      {/* Subtitle */}
      {subtitle && (
        <EnhancedText variant="caption">
          {subtitle}
        </EnhancedText>
      )}
    </EnhancedCard>
  )
}

// Progress Card with Beautiful Animations
interface ProgressCardProps {
  title: string
  current: number
  target: number
  unit?: string
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  showPercentage?: boolean
  className?: string
}

export const EnhancedProgressCard: React.FC<ProgressCardProps> = ({
  title,
  current,
  target,
  unit = '',
  tier,
  showPercentage = true,
  className
}) => {
  const percentage = Math.min((current / target) * 100, 100)
  
  return (
    <EnhancedCard variant="glass" tier={tier} className={cn('p-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <EnhancedText variant="premium" weight="medium">
          {title}
        </EnhancedText>
        {showPercentage && (
          <EnhancedBadge variant="tier" tier={tier}>
            {Math.round(percentage)}%
          </EnhancedBadge>
        )}
      </div>
      
      {/* Values */}
      <div className="flex items-baseline gap-2 mb-4">
        <span className={cn(
          'text-2xl font-montserrat font-bold',
          tier ? `text-tier-${tier}` : 'text-primary'
        )}>
          {current}
        </span>
        <EnhancedText variant="muted">
          / {target} {unit}
        </EnhancedText>
      </div>
      
      {/* Progress Bar */}
      <div className="relative">
        <div className="w-full h-2 bg-surface-glass rounded-full overflow-hidden">
          <motion.div
            className={cn(
              'h-full rounded-full',
              tier ? `bg-tier-${tier}` : 'bg-primary'
            )}
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-subtle-shimmer" />
      </div>
    </EnhancedCard>
  )
}

// Activity Timeline
interface ActivityItem {
  id: string
  title: string
  description: string
  timestamp: string
  type: 'workout' | 'nutrition' | 'progress' | 'achievement'
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
}

interface ActivityTimelineProps {
  activities: ActivityItem[]
  className?: string
}

export const EnhancedActivityTimeline: React.FC<ActivityTimelineProps> = ({
  activities,
  className
}) => {
  const typeConfig = {
    workout: { color: 'tier-adaptive', icon: '💪' },
    nutrition: { color: 'tier-core', icon: '🥗' },
    progress: { color: 'tier-performance', icon: '📈' },
    achievement: { color: 'tier-longevity', icon: '🏆' }
  }
  
  return (
    <EnhancedCard variant="frost" className={cn('p-6', className)}>
      <EnhancedHeading variant="subtitle" className="mb-6">
        Recent Activity
      </EnhancedHeading>
      
      <div className="space-y-4">
        {activities.map((activity, index) => {
          const config = typeConfig[activity.type]
          
          return (
            <motion.div
              key={activity.id}
              className="flex gap-4 p-4 rounded-lg bg-surface-glass hover:bg-surface-frost transition-all"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              {/* Icon */}
              <div className={cn(
                'flex items-center justify-center w-10 h-10 rounded-full',
                `bg-${config.color}/20 text-${config.color}`
              )}>
                <span className="text-lg">{config.icon}</span>
              </div>
              
              {/* Content */}
              <div className="flex-1 min-w-0">
                <EnhancedText variant="premium" weight="medium" className="mb-1">
                  {activity.title}
                </EnhancedText>
                <EnhancedText variant="muted" size="sm" className="mb-2">
                  {activity.description}
                </EnhancedText>
                <EnhancedText variant="caption">
                  {activity.timestamp}
                </EnhancedText>
              </div>
              
              {/* Tier Badge */}
              {activity.tier && (
                <EnhancedBadge variant="tier" tier={activity.tier}>
                  {activity.tier}
                </EnhancedBadge>
              )}
            </motion.div>
          )
        })}
      </div>
    </EnhancedCard>
  )
}

// Quick Stats Grid
interface QuickStat {
  label: string
  value: string
  change?: number
  changeType?: 'positive' | 'negative' | 'neutral'
}

interface QuickStatsProps {
  stats: QuickStat[]
  tier?: 'core' | 'adaptive' | 'performance' | 'longevity'
  className?: string
}

export const EnhancedQuickStats: React.FC<QuickStatsProps> = ({
  stats,
  tier,
  className
}) => {
  return (
    <div className={cn('grid grid-cols-2 md:grid-cols-4 gap-4', className)}>
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          className={cn(
            'p-4 rounded-xl backdrop-blur-sm border text-center',
            tier ? `bg-tier-${tier}/10 border-tier-${tier}/20` : 'bg-surface-glass border-white/10'
          )}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          whileHover={{ scale: 1.02 }}
        >
          <motion.div
            className={cn(
              'text-2xl font-montserrat font-bold mb-1',
              tier ? `text-tier-${tier}` : 'text-primary'
            )}
            initial={{ scale: 0.5 }}
            animate={{ scale: 1 }}
            transition={{ delay: index * 0.1 + 0.2, type: "spring" }}
          >
            {stat.value}
          </motion.div>
          
          <EnhancedText variant="muted" size="sm" className="mb-1">
            {stat.label}
          </EnhancedText>
          
          {stat.change !== undefined && (
            <div className={cn(
              'text-xs font-medium',
              stat.changeType === 'positive' ? 'text-green-400' :
              stat.changeType === 'negative' ? 'text-red-400' : 'text-white/60'
            )}>
              {stat.changeType === 'positive' ? '+' : ''}{stat.change}%
            </div>
          )}
        </motion.div>
      ))}
    </div>
  )
}

// Feature Spotlight Card
interface FeatureSpotlightProps {
  title: string
  description: string
  tier: 'core' | 'adaptive' | 'performance' | 'longevity'
  available: boolean
  children?: React.ReactNode
  className?: string
}

export const EnhancedFeatureSpotlight: React.FC<FeatureSpotlightProps> = ({
  title,
  description,
  tier,
  available,
  children,
  className
}) => {
  return (
    <EnhancedCard 
      variant="glass" 
      tier={tier}
      className={cn(
        'p-6 relative overflow-hidden',
        !available && 'opacity-60',
        className
      )}
    >
      {/* Glow Effect */}
      {available && (
        <div className={cn(
          'absolute inset-0 opacity-20',
          `bg-gradient-to-br from-tier-${tier}/30 to-transparent`
        )} />
      )}
      
      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <EnhancedHeading variant="subtitle" tier={tier} gradient>
              {title}
            </EnhancedHeading>
            <EnhancedText variant="muted" className="mt-2">
              {description}
            </EnhancedText>
          </div>
          
          <EnhancedBadge 
            variant="status" 
            status={available ? 'active' : 'inactive'}
          >
            {available ? 'Available' : 'Upgrade Required'}
          </EnhancedBadge>
        </div>
        
        {children}
        
        {!available && (
          <motion.button
            className={cn(
              'mt-4 w-full py-3 rounded-lg font-montserrat font-semibold',
              'bg-gradient-to-r transition-all duration-300',
              `from-tier-${tier}/20 to-tier-${tier}/10`,
              `border border-tier-${tier}/30 text-tier-${tier}`,
              'hover:from-tier-${tier}/30 hover:to-tier-${tier}/20'
            )}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Upgrade to {tier.charAt(0).toUpperCase() + tier.slice(1)}
          </motion.button>
        )}
      </div>
    </EnhancedCard>
  )
}