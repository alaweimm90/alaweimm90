import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardFooter, CardHeader } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Check, Sparkles, Zap, Crown, Shield } from 'lucide-react';
import { TIER_CONFIGS, TierType, getTierConfigByType } from '@/constants/tiers';
import { cn } from "@/lib/utils";
import { motion } from 'framer-motion';

interface UnifiedTierCardProps {
  tier: TierType;
  isPopular?: boolean;
  className?: string;
}

/**
 * UnifiedTierCard - Single source of truth for monthly subscription tier display
 * Automatically pulls configuration from constants/tiers.ts
 */
export function UnifiedTierCard({ tier, isPopular = false, className }: UnifiedTierCardProps) {
  const navigate = useNavigate();
  const [isHovered, setIsHovered] = useState(false);
  const config = getTierConfigByType(tier);
  
  // Tier-specific styling
  const tierStyles = {
    core: {
      gradient: 'from-blue-500/20 to-blue-600/20',
      border: 'border-blue-500/50',
      badge: 'bg-blue-500/20 text-blue-300',
      button: 'bg-blue-500 hover:bg-blue-600',
      icon: <Shield className="w-6 h-6" />,
      glow: 'hover:shadow-blue-500/20'
    },
    adaptive: {
      gradient: 'from-repz-primary/20 to-orange-600/20',
      border: 'border-repz-primary/50',
      badge: 'bg-repz-primary/20 text-orange-300',
      button: 'bg-repz-primary hover:bg-orange-600',
      icon: <Zap className="w-6 h-6" />,
      glow: 'hover:shadow-repz-primary/20'
    },
    performance: {
      gradient: 'from-purple-500/20 to-purple-600/20',
      border: 'border-purple-500/50',
      badge: 'bg-purple-500/20 text-purple-300',
      button: 'bg-purple-500 hover:bg-purple-600',
      icon: <Sparkles className="w-6 h-6" />,
      glow: 'hover:shadow-purple-500/20'
    },
    longevity: {
      gradient: 'from-yellow-500/20 to-yellow-600/20',
      border: 'border-yellow-500/50',
      badge: 'bg-yellow-500/20 text-yellow-300',
      button: 'bg-yellow-500 hover:bg-yellow-600 text-black',
      icon: <Crown className="w-6 h-6" />,
      glow: 'hover:shadow-yellow-500/20'
    }
  };

  const style = tierStyles[tier];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      whileHover={{ y: -4 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      <Card 
        className={cn(
          "relative overflow-hidden transition-all duration-300",
          "bg-gray-900/50 backdrop-blur-sm",
          style.border,
          style.glow,
          "hover:shadow-2xl",
          isPopular && "scale-105 shadow-2xl",
          className
        )}
      >
        {/* Background Gradient */}
        <div className={cn(
          "absolute inset-0 bg-gradient-to-br opacity-10",
          style.gradient
        )} />
        
        {/* Popular Badge */}
        {(isPopular || config.badge) && (
          <div className="absolute -top-4 -right-4">
            <Badge className={cn(
              "px-4 py-1 text-sm font-bold transform rotate-12",
              style.badge
            )}>
              {config.badge || 'MOST POPULAR'}
            </Badge>
          </div>
        )}

        <CardHeader className="relative z-10 pb-4">
          <div className="flex items-center justify-between mb-2">
            <div className={cn("text-3xl", style.badge.replace('bg-', 'text-').replace('/20', ''))}>
              {style.icon}
            </div>
            {config.isLimited && (
              <Badge variant="outline" className="text-xs">
                LIMITED SPOTS
              </Badge>
            )}
          </div>
          
          <h3 className="text-2xl font-bold text-white">
            {config.displayName}
          </h3>
          <p className="text-gray-400 text-sm mt-1">
            {config.tagline}
          </p>
        </CardHeader>

        <CardContent className="relative z-10 space-y-6">
          {/* Pricing */}
          <div className="space-y-2">
            <div className="flex items-baseline">
              <span className="text-4xl font-bold text-white">
                ${config.monthlyPrice}
              </span>
              <span className="text-gray-400 ml-2">/month</span>
            </div>
            <p className="text-sm text-gray-500">
              or ${config.annualPrice}/mo billed annually (save 20%)
            </p>
          </div>

          {/* Description */}
          <p className="text-gray-300 text-sm">
            {config.description}
          </p>

          {/* Features */}
          <ul className="space-y-3">
            {config.features.slice(0, 6).map((feature, index) => (
              <li key={index} className="flex items-start">
                <Check className={cn(
                  "w-5 h-5 mt-0.5 mr-3 flex-shrink-0",
                  style.badge.replace('bg-', 'text-').replace('/20', '')
                )} />
                <span className="text-gray-300 text-sm">{feature}</span>
              </li>
            ))}
          </ul>

          {/* Limited Features Preview */}
          {config.limitedFeatures.length > 0 && (
            <div className="pt-2 border-t border-gray-800">
              <p className="text-xs text-gray-500 mb-2">
                Includes {config.limitedFeatures.length} exclusive features
              </p>
            </div>
          )}
        </CardContent>

        <CardFooter className="relative z-10 pt-4">
          <Button
            className={cn(
              "w-full py-6 text-lg font-semibold transition-all",
              style.button,
              isHovered && "scale-105"
            )}
            onClick={() => navigate(`/auth?tier=${tier}`)}
          >
            {config.ctaText}
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  );
}