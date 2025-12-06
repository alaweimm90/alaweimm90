import React from 'react';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierType } from '@/types/fitness';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Lock, ArrowUp } from 'lucide-react';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { TIER_FEATURES } from '@/constants/featureMatrix';
import { useNavigate } from 'react-router-dom';

interface TierGateProps {
  children: React.ReactNode;
  requiredTier: TierType;
  feature?: string;
  fallback?: React.ReactNode;
  showUpgrade?: boolean; // optional, accepted for compatibility
}

export function TierGate({ children, requiredTier, feature, fallback }: TierGateProps) {
  const { hasMinimumTier, userTier, nextTier } = useTierAccess();
  const navigate = useNavigate();

  const hasAccess = hasMinimumTier(requiredTier);

  if (hasAccess) {
    return <>{children}</>;
  }

  if (fallback) {
    return <>{fallback}</>;
  }

  // Default upgrade prompt
  return (
    <Card className="glass-tier-card border-border/20">
      <CardHeader className="text-center">
        <div className="mx-auto w-12 h-12 rounded-full bg-gradient-to-br from-orange-500/20 to-purple-500/20 flex items-center justify-center mb-4">
          <Lock className="w-6 h-6 text-orange-500" />
        </div>
        <CardTitle className="text-xl">
          {feature ? `${feature} Requires Upgrade` : 'Upgrade Required'}
        </CardTitle>
        <CardDescription>
          This feature is available starting with the <span className="font-semibold text-orange-500">{requiredTier}</span> tier.
          {userTier && (
            <span className="block mt-1">
              You're currently on the <span className="font-semibold">{userTier}</span> tier.
            </span>
          )}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {nextTier && (
          <div className="space-y-4">
            <div className="text-sm text-muted-foreground text-center">
              Upgrade to unlock this feature:
            </div>
            <UnifiedTierCard
              tier={nextTier}
              variant="compact"
              showFeatures={false}
            />
          </div>
        )}
        
        <div className="flex gap-3">
          <Button 
            onClick={() => navigate('/')}
            className="flex-1"
          >
            <ArrowUp className="w-4 h-4 mr-2" />
            View Plans
          </Button>
          <Button 
            variant="outline" 
            onClick={() => window.history.back()}
            className="flex-1"
          >
            Go Back
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper functions
function getTheoreticalPrice(tier: TierType): number {
  const pricing = {
    core: 89,
    adaptive: 149,
    performance: 229,
    longevity: 349
  };
  return pricing[tier];
}

function getTierColor(tier: TierType): string {
  const colors = {
    core: '#3B82F6',
    adaptive: '#F15B23',
    performance: '#A855F7',
    longevity: '#EAB308'
  };
  return colors[tier];
}