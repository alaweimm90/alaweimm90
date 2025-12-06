import React from 'react';
import { Card, CardContent, CardHeader } from "@/ui/molecules/Card";
import { Button } from "@/components/ui/unified-button";
import { ComingSoonOverlay } from "@/components/ui/coming-soon-overlay";
import { Badge } from "@/ui/atoms/Badge";
import { Check, X, Sprout, TrendingUp, Zap, Crown, Building, Home, MapPin } from 'lucide-react';
import { cn } from '@/lib/utils';

// =====================================================================
// UNIFIED PRICING CARD ARCHITECTURE - SUPERPROMPT IMPLEMENTATION
// Modular component system preserving contextual identity
// =====================================================================

export type PricingCardVariant = 'monthly' | 'gym';
export type TierType = 'core' | 'adaptive' | 'performance' | 'longevity';
export type BadgeType = 'popular' | 'premium' | 'coach-pick' | 'limited';
export type GymType = 'other' | 'city-sports' | 'home';

export interface PricingCardFeature {
  text: string;
  included: boolean;
  icon?: React.ReactNode;
  iconColor?: string;
  highlight?: boolean;
}

export interface PricingCardProps {
  // Core Content
  title: string;
  subtitle: string;
  price: string;
  priceSubtext?: string;
  priceSubtextColor?: string;
  description?: string;
  
  // Features
  features: PricingCardFeature[];
  
  // Visual Design
  tier: TierType;
  variant: PricingCardVariant;
  gymType?: GymType; // For gym-specific styling
  badge?: BadgeType;
  badgePosition?: 'top-left' | 'top-right';
  
  // Special Content
  specialSection?: React.ReactNode;
  
  // Interaction
  buttonText?: string;
  buttonVariant?: 'primary' | 'secondary' | 'ghost' | 'destructive';
  onClick?: () => void;
  
  // State
  isSelected?: boolean;
  isDimmed?: boolean;
  
  // Coming soon overlay
  comingSoon?: boolean;
  comingSoonMessage?: string;
  
  // Styling
  className?: string;
}

// UNIFIED ARCHITECTURAL TOKENS
const architecturalTokens = {
  // Typography - Consistent across all cards and views
  typography: {
    title: 'text-xl font-semibold leading-tight', // Removed text-white - will be conditional
    subtitle: 'text-gray-300 text-sm',
    price: 'text-2xl font-bold text-white mb-1', // Price always white
    priceSubtext: 'text-sm font-semibold',
    description: 'text-gray-300 text-sm mt-2',
    featureText: 'text-sm' // Removed text-white - will be conditional
  },
  
  // Spacing - Unified layout grid with consistent tokens
  spacing: {
    cardPadding: 'p-6',
    headerPadding: 'pb-6',
    contentPadding: 'pt-0',
    iconGap: 'gap-4',
    featureSpacing: 'space-y-0', // Removed automatic spacing, using manual mb-4
    sectionMargin: 'mb-6',
    checklistItemGap: 'gap-3', // 12px gap between icon and text
    buttonMargin: 'mt-6' // Consistent button spacing
  },
  
  // Visual Elements - Consistent across variants  
  visual: {
    cornerRadius: 'rounded-2xl',
    shadow: 'shadow-2xl',
    borderWidth: 'border-2',
    iconSize: 'w-12 h-12',
    iconRadius: 'rounded-xl',
    checkIconSize: 'w-5 h-5 flex-shrink-0',
    cardDimensions: 'w-full max-w-[360px] min-h-[580px]' // Unified min-height: 580px
  },
  
  // Layout - Flex system for consistent card structure
  layout: {
    cardContainer: 'flex flex-col justify-between h-full',
    cardContent: 'flex-1 flex flex-col',
    cardBody: 'flex-1', // Body grows to fill space
    cardFooter: 'mt-auto pt-6', // Footer pinned to bottom
    featuresContainer: 'flex-1'
  },
  
  // Badge System - Global positioning rule
  badge: {
    position: 'absolute top-[-12px] right-3 z-20',
    transform: 'transform -translate-y-1/2',
    style: 'px-3 py-1 rounded-full text-xs font-bold',
    shadow: 'shadow-lg'
  }
};

// TIER-SPECIFIC TOKENS (Monthly Coaching)
const tierConfig = {
  core: {
    icon: Sprout,
    gradient: 'from-gray-500/20 via-gray-600/15 to-gray-700/10',
    borderColor: 'border-gray-500/30',
    iconBg: 'bg-gray-500',
    iconColor: 'text-gray-400',
    accentColor: 'text-gray-400'
  },
  adaptive: {
    icon: TrendingUp,
    gradient: 'from-pink-300/20 via-pink-400/15 to-pink-500/10',
    borderColor: 'border-pink-400/30',
    iconBg: 'bg-pink-400',
    iconColor: 'text-pink-300',
    accentColor: 'text-pink-300'
  },
  performance: {
    icon: Zap,
    gradient: 'from-purple-800/20 via-purple-900/15 to-indigo-900/10',
    borderColor: 'border-purple-600/30',
    iconBg: 'bg-purple-700',
    iconColor: 'text-purple-400',
    accentColor: 'text-purple-400'
  },
  longevity: {
    icon: Crown,
    gradient: 'from-black via-black to-black',
    borderColor: 'border-yellow-400/60 shadow-[0_0_20px_rgba(250,204,21,0.4),0_0_40px_rgba(250,204,21,0.2)]', // Always glowing - removed hover condition
    iconBg: 'bg-yellow-500',
    iconColor: 'text-yellow-400',
    accentColor: 'text-yellow-400',
    textColor: 'text-yellow-400' // Gold text for Elite Concierge
  }
};

// GYM-SPECIFIC TOKENS (Preserving Visual Identity)
const gymConfig = {
  other: {
    icon: MapPin,
    gradient: 'from-gray-600/25 via-gray-700/20 to-gray-800/15',
    borderColor: 'border-gray-500/40',
    iconBg: 'bg-gray-600',
    iconColor: 'text-gray-300',
    accentColor: 'text-gray-400',
    theme: 'External, variable, third-party dependent'
  },
  'city-sports': {
    icon: Building,
    gradient: 'from-orange-600/25 via-orange-700/20 to-amber-600/15',
    borderColor: 'border-orange-500/40',
    iconBg: 'bg-orange-600',
    iconColor: 'text-orange-300',
    accentColor: 'text-orange-400',
    theme: 'Structured, urban, high-access'
  },
  home: {
    icon: Home,
    gradient: 'from-purple-600/25 via-purple-700/20 to-indigo-600/15',
    borderColor: 'border-purple-500/40',
    iconBg: 'bg-purple-600',
    iconColor: 'text-purple-300',
    accentColor: 'text-purple-400',
    theme: 'Private, flexible, personalized'
  }
};

// BADGE CONFIGURATIONS - Global positioning and styling
const badgeConfig = {
  popular: {
    text: 'Most Popular',
    className: 'bg-gradient-to-r from-purple-600 to-purple-700 text-white'
  },
  premium: {
    text: 'Premium',
    className: 'bg-gradient-to-r from-yellow-500 to-yellow-600 text-black'
  },
  'coach-pick': {
    text: "Coach's Pick",
    className: 'bg-gradient-to-r from-orange-600 to-orange-700 text-white'
  },
  limited: {
    text: 'LIMITED 5 CLIENTS',
    className: 'bg-gradient-to-r from-yellow-500 to-yellow-600 text-black'
  }
};

export const UnifiedPricingCard: React.FC<PricingCardProps> = ({
  title,
  subtitle,
  price,
  priceSubtext,
  priceSubtextColor = 'text-green-400',
  description,
  features,
  tier,
  variant,
  gymType,
  badge,
  badgePosition = 'top-right',
  specialSection,
  buttonText = 'Select',
  buttonVariant = 'primary',
  onClick,
  isSelected = false,
  isDimmed = false,
  comingSoon = false,
  comingSoonMessage,
  className
}) => {
  // Choose configuration based on variant
  const config = variant === 'gym' && gymType 
    ? gymConfig[gymType] 
    : tierConfig[tier];
  
  const Icon = config.icon;
  const badgeInfo = badge ? badgeConfig[badge] : null;

  return (
    <ComingSoonOverlay disabled={comingSoon} message={comingSoonMessage || "A modern, data‑rich dashboard experience is on the way. Monthly plans will open soon."}>
    <Card className={cn(
      // UNIFIED ARCHITECTURE - Fixed dimensions and flex layout
      "relative transition-all duration-500 group",
      architecturalTokens.visual.cardDimensions,
      architecturalTokens.layout.cardContainer,
      architecturalTokens.visual.cornerRadius,
      architecturalTokens.visual.borderWidth,
      "backdrop-blur-xl overflow-hidden",
      
      // CONTEXTUAL STYLING - Apply proper tier/gym colors
      `bg-gradient-to-br ${config.gradient}`,
      config.borderColor,
      
      // INTERACTIVE STATES
      isSelected && "ring-2 ring-offset-2 ring-offset-background",
      isSelected && `ring-${config.iconColor.split('-')[1]}-500`,
      
      // HOVER EFFECTS
      architecturalTokens.visual.shadow,
      "hover:scale-105",
      !isDimmed && "hover:border-opacity-50",
      
      // DIMMED STATE
      isDimmed && "opacity-50",
      
      className
    )}>
      
      {/* UNIFIED BADGE SYSTEM - Global positioning */}
      {badge && badgeInfo && (
        <div className={cn(
          architecturalTokens.badge.position,
          architecturalTokens.badge.transform,
          architecturalTokens.badge.style,
          architecturalTokens.badge.shadow,
          badgeInfo.className
        )}>
          {badgeInfo.text}
        </div>
      )}

      {/* UNIFIED BACKGROUND EFFECTS */}
      <div className="absolute inset-0 bg-gradient-to-br from-black/60 via-gray-950/80 to-black/60 rounded-2xl" />
      
      <CardHeader className={cn("relative z-10 flex-shrink-0", architecturalTokens.spacing.cardPadding, architecturalTokens.spacing.headerPadding)}>
        {/* UNIFIED HEADER LAYOUT - Consistent top alignment */}
        <div className={cn("flex items-center", architecturalTokens.spacing.iconGap, "mb-6")}>
          <div className={cn(
            architecturalTokens.visual.iconSize,
            architecturalTokens.visual.iconRadius,
            "flex items-center justify-center shadow-lg",
            config.iconBg,
            "group-hover:shadow-xl transition-all duration-300"
          )}>
            <Icon className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className={cn(
              architecturalTokens.typography.title,
              tier === 'longevity' ? 'text-yellow-400' : 'text-white'
            )}>
              {title}
            </h3>
            <p className={cn(
              architecturalTokens.typography.subtitle,
              tier === 'longevity' ? 'text-yellow-300' : 'text-gray-300'
            )}>
              {subtitle}
            </p>
          </div>
        </div>

        {/* UNIFIED PRICE SECTION - Consistent spacing */}
        <div className="mb-6">
          <div className={cn(
            "text-2xl font-bold mb-1",
            tier === 'longevity' ? 'text-yellow-400' : 'text-white'
          )}>
            {price}
          </div>
          {priceSubtext && (
            <div className={cn(
              architecturalTokens.typography.priceSubtext, 
              // Keep save text green for all tiers, including Elite Concierge
              priceSubtextColor
            )}>
              {priceSubtext}
            </div>
          )}
          {description && (
            <p className={cn(
              architecturalTokens.typography.description,
              tier === 'longevity' ? 'text-yellow-400' : 'text-gray-300'
            )}>
              {description}
            </p>
          )}
        </div>
      </CardHeader>

      <CardContent className={cn("relative z-10", architecturalTokens.layout.cardContent, architecturalTokens.spacing.contentPadding)}>
        {/* UNIFIED FEATURES LIST - Clean checklist with equal spacing */}
        <div className={cn(architecturalTokens.layout.cardBody, architecturalTokens.spacing.sectionMargin)}>
          {features.map((feature, index) => (
            <div key={index} className={cn(
              "flex items-start", // Consistent alignment
              architecturalTokens.spacing.checklistItemGap,
              index !== features.length - 1 && "mb-4" // 16px spacing between items (not on last item)
            )}>
              {/* Standard inline icon + text format - conditional colors for Elite Concierge */}
              <div className="flex-shrink-0">
                {feature.icon || (
                  feature.included ? (
                    <Check className={cn(
                      architecturalTokens.visual.checkIconSize, 
                      tier === 'longevity' ? 'text-yellow-400' : (feature.iconColor || config.accentColor)
                    )} />
                  ) : (
                    <X className={cn(architecturalTokens.visual.checkIconSize, "text-red-400")} />
                  )
                )}
              </div>
              <span className={cn(
                architecturalTokens.typography.featureText,
                "leading-relaxed", // Consistent line height (1.6)
                tier === 'longevity' 
                  ? 'text-yellow-400' 
                  : (feature.included ? "text-white" : "text-gray-400")
              )}>
                {feature.text}
              </span>
            </div>
          ))}
        </div>

        {/* UNIFIED SPECIAL SECTION */}
        {specialSection && (
          <div className={architecturalTokens.spacing.sectionMargin}>
            {specialSection}
          </div>
        )}

        {/* UNIFIED BUTTON TREATMENT - Pinned to bottom with consistent spacing */}
        <div className={cn(architecturalTokens.layout.cardFooter)}>
          <Button
            onClick={comingSoon ? undefined : onClick}
            disabled={comingSoon}
            aria-disabled={comingSoon}
            variant={buttonVariant}
            className={cn(
              "w-full font-semibold transition-all duration-300",
              isSelected && "ring-2 ring-offset-2 ring-offset-background",
              isSelected && `ring-${config.iconColor.split('-')[1]}-500`,
              comingSoon && "cursor-not-allowed"
            )}
          >
            {buttonText}
          </Button>
        </div>
      </CardContent>
    </Card>
    </ComingSoonOverlay>
  );
};

// CONVENIENCE COMPONENTS
export const MonthlyCoachingCard: React.FC<Omit<PricingCardProps, 'variant'>> = (props) => (
  <UnifiedPricingCard {...props} variant="monthly" />
);

export const GymTierCard: React.FC<Omit<PricingCardProps, 'variant'> & { gymType: GymType }> = (props) => (
  <UnifiedPricingCard {...props} variant="gym" />
);