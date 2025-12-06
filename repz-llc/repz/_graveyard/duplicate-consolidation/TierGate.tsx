import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierType, TierFeatures } from '@/types/fitness';
import { Lock, ArrowUp, Star } from 'lucide-react';

interface TierGateProps {
  children: React.ReactNode;
  requiredTier: TierType;
  feature: keyof TierFeatures;
  fallback?: React.ReactNode;
  showUpgrade?: boolean;
}

interface UpgradePromptProps {
  currentTier: TierType;
  requiredTier: TierType;
  feature: string;
  nextTier?: TierType;
}

// Simple tier styling function to replace the missing import
const getTierStyling = (tier: TierType) => {
  switch (tier) {
    case 'core':
      return { badge: 'bg-blue-100 text-blue-700 border-blue-200' };
    case 'adaptive':
      return { badge: 'bg-orange-100 text-orange-700 border-orange-200' };
    case 'performance':  
      return { badge: 'bg-purple-100 text-purple-700 border-purple-200' };
    case 'longevity':
      return { badge: 'bg-yellow-100 text-yellow-700 border-yellow-200' };
    default:
      return { badge: 'bg-gray-100 text-gray-700 border-gray-200' };
  }
};

// Simple tier icon function
const getTierIcon = (tier: TierType) => {
  switch (tier) {
    case 'core': return '🌱';
    case 'adaptive': return '📈';
    case 'performance': return '⚡';
    case 'longevity': return '👑';
    default: return '🌱';
  }
};

const UpgradePrompt: React.FC<UpgradePromptProps> = ({ 
  currentTier, 
  requiredTier, 
  feature, 
  nextTier 
}) => {
  const navigate = useNavigate();
  const currentStyling = getTierStyling(currentTier);
  const requiredStyling = getTierStyling(requiredTier);
  
  const tierPrices = {
    core: 96,
    adaptive: 179,
    performance: 299,
    longevity: 449
  };

  return (
    <Card className="border-dashed border-2 bg-muted/20">
      <CardHeader className="text-center">
        <div className="flex items-center justify-center mb-2">
          <Lock className="h-8 w-8 text-muted-foreground" />
        </div>
        <CardTitle className="text-lg">Feature Locked</CardTitle>
        <CardDescription>
          <span className="capitalize">{feature.replace(/_/g, ' ')}</span> requires {getTierIcon(requiredTier)} {requiredTier} tier or higher
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="flex items-center justify-center space-x-4">
          <div className="text-center">
            <Badge variant="outline" className={currentStyling.badge}>
              {getTierIcon(currentTier)} Current: {currentTier}
            </Badge>
            <p className="text-sm text-muted-foreground mt-1">
              ${tierPrices[currentTier]}/month
            </p>
          </div>
          
          <ArrowUp className="h-4 w-4 text-muted-foreground" />
          
          <div className="text-center">
            <Badge variant="outline" className={requiredStyling.badge}>
              {getTierIcon(requiredTier)} Required: {requiredTier}
            </Badge>
            <p className="text-sm text-muted-foreground mt-1">
              ${tierPrices[requiredTier]}/month
            </p>
          </div>
        </div>

        <div className="text-center">
          <Button 
            className="w-full" 
            variant="default"
            onClick={() => navigate('/')}
          >
            <Star className="h-4 w-4 mr-2" />
            Upgrade to {requiredTier} Tier
          </Button>
          
          {nextTier && nextTier !== requiredTier && (
            <p className="text-xs text-muted-foreground mt-2">
              Or upgrade to {getTierIcon(nextTier)} {nextTier} (${tierPrices[nextTier]}/month) to unlock this feature
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

/**
 * TierGate component that conditionally renders content based on subscription tier
 */
export const TierGate: React.FC<TierGateProps> = ({ 
  children, 
  requiredTier, 
  feature, 
  fallback,
  showUpgrade = true
}) => {
  const { hasMinimumTier, userTier } = useTierAccess();
  const hasAccess = hasMinimumTier(requiredTier);

  if (hasAccess) {
    return <>{children}</>;
  }

  if (fallback) {
    return <>{fallback}</>;
  }

  if (showUpgrade) {
    return (
      <UpgradePrompt 
        currentTier={userTier || 'core'}
        requiredTier={requiredTier}
        feature={feature as string}
        nextTier={requiredTier}
      />
    );
  }

  return null;
};