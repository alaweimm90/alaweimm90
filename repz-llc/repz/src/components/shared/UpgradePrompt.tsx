import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Lock, Crown, Zap, Award } from 'lucide-react';
import { TierType } from '@/types/fitness';
import { getTierDisplayName } from '@/utils/tierMigration';

interface UpgradePromptProps {
  feature: string;
  currentTier: TierType | null;
  suggestedTier?: TierType;
  onUpgrade?: () => void;
  className?: string;
  variant?: 'card' | 'inline' | 'modal';
}

const tierIcons = {
  core: Zap,
  adaptive: Award,  
  performance: Crown,
  longevity: Crown
};

const tierColors = {
  core: 'bg-blue-500',
  adaptive: 'bg-emerald-500',
  performance: 'bg-orange-500',
  longevity: 'bg-black'
};

export const UpgradePrompt: React.FC<UpgradePromptProps> = ({
  feature,
  currentTier,
  suggestedTier = 'adaptive',
  onUpgrade,
  className = '',
  variant = 'card'
}) => {
  const currentTierName = currentTier ? getTierDisplayName(currentTier) : 'Free';
  const suggestedTierName = getTierDisplayName(suggestedTier);
  const SuggestedIcon = tierIcons[suggestedTier];
  const tierColorClass = tierColors[suggestedTier];

  const handleUpgradeClick = () => {
    if (onUpgrade) {
      onUpgrade();
    } else {
      // Default behavior - navigate to pricing
      window.location.href = '/pricing';
    }
  };

  if (variant === 'inline') {
    return (
      <div className={`flex items-center gap-2 p-2 bg-gray-50 rounded-lg border ${className}`}>
        <Lock className="h-4 w-4 text-gray-400" />
        <span className="text-sm text-gray-600">
          {feature} requires {suggestedTierName}
        </span>
        <Button size="sm" variant="outline" onClick={handleUpgradeClick}>
          Upgrade
        </Button>
      </div>
    );
  }

  if (variant === 'modal') {
    return (
      <div className={`text-center p-6 ${className}`}>
        <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full ${tierColorClass} mb-4`}>
          <SuggestedIcon className="h-8 w-8 text-white" />
        </div>
        <h3 className="text-xl font-semibold mb-2">Upgrade Required</h3>
        <p className="text-gray-600 mb-4">
          Access to <strong>{feature}</strong> requires the {suggestedTierName} tier or higher.
        </p>
        <div className="flex gap-2 justify-center">
          <Button variant="outline" onClick={() => window.history.back()}>
            Go Back
          </Button>
          <Button onClick={handleUpgradeClick}>
            Upgrade to {suggestedTierName}
          </Button>
        </div>
      </div>
    );
  }

  // Default card variant
  return (
    <Card className={`border-dashed border-2 border-gray-300 ${className}`}>
      <CardHeader className="text-center">
        <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full ${tierColorClass} mb-2`}>
          <Lock className="h-6 w-6 text-white" />
        </div>
        <CardTitle className="text-lg">Upgrade Required</CardTitle>
        <CardDescription>
          <strong>{feature}</strong> is available in {suggestedTierName} tier and above
        </CardDescription>
      </CardHeader>
      <CardContent className="text-center space-y-3">
        <div className="flex items-center justify-center gap-2">
          <Badge variant="secondary">Current: {currentTierName}</Badge>
          <span className="text-gray-400">→</span>
          <Badge className={tierColorClass}>Suggested: {suggestedTierName}</Badge>
        </div>
        
        <Button 
          className="w-full" 
          onClick={handleUpgradeClick}
        >
          <SuggestedIcon className="h-4 w-4 mr-2" />
          Upgrade to {suggestedTierName}
        </Button>
        
        <p className="text-xs text-gray-500">
          Unlock this feature and more with a higher tier subscription
        </p>
      </CardContent>
    </Card>
  );
};

// Convenience component for specific features
export const WeeklyCheckinsUpgrade: React.FC<Omit<UpgradePromptProps, 'feature' | 'suggestedTier'>> = (props) => (
  <UpgradePrompt 
    {...props} 
    feature="Weekly Check-ins & Photo Reviews" 
    suggestedTier="adaptive" 
  />
);

export const PeptidesUpgrade: React.FC<Omit<UpgradePromptProps, 'feature' | 'suggestedTier'>> = (props) => (
  <UpgradePrompt 
    {...props} 
    feature="Peptides Protocols" 
    suggestedTier="performance" 
  />
);

export const PedsUpgrade: React.FC<Omit<UpgradePromptProps, 'feature' | 'suggestedTier'>> = (props) => (
  <UpgradePrompt 
    {...props} 
    feature="PEDs Protocols" 
    suggestedTier="longevity" 
  />
);

export const AIAssistantUpgrade: React.FC<Omit<UpgradePromptProps, 'feature' | 'suggestedTier'>> = (props) => (
  <UpgradePrompt 
    {...props} 
    feature="AI Fitness Assistant" 
    suggestedTier="adaptive" 
  />
);