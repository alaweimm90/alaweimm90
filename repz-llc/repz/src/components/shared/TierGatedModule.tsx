import React from 'react';
import { TierType } from '@/types/fitness';
import { useTierAccess } from '@/hooks/useTierAccess';
import { UpgradePrompt } from './UpgradePrompt';

interface TierGatedModuleProps {
  moduleName: string;
  userTier: TierType;
  accessLevel: 'none' | 'readonly' | 'interactive' | 'advanced' | 'premium';
  children: React.ReactNode;
  fallbackComponent?: React.ComponentType<Record<string, unknown>>;
  requiredTier?: TierType;
  featureVariable?: string;
}

export const TierGatedModule: React.FC<TierGatedModuleProps> = ({
  moduleName,
  userTier,
  accessLevel,
  children,
  fallbackComponent: FallbackComponent,
  requiredTier,
  featureVariable
}) => {
  const { hasFeature, hasMinimumTier } = useTierAccess();

  // Check access based on feature variable or required tier
  let hasAccess = false;

  if (featureVariable) {
    hasAccess = hasFeature(featureVariable as string);
  } else if (requiredTier) {
    hasAccess = hasMinimumTier(requiredTier);
  } else {
    // Fallback to access level check
    hasAccess = accessLevel !== 'none';
  }

  if (!hasAccess) {
    if (FallbackComponent) {
      return <FallbackComponent moduleName={moduleName} currentTier={userTier} />;
    }
    return <UpgradePrompt feature={moduleName} currentTier={userTier} />;
  }

  // Wrap children with access level context if needed
  return (
    <div 
      className="tier-gated-module" 
      data-module={moduleName}
      data-access-level={accessLevel}
      data-user-tier={userTier}
    >
      {children}
    </div>
  );
};

// Higher-order component version
export function withTierGating<P extends object>(
  Component: React.ComponentType<P>,
  options: {
    moduleName: string;
    requiredTier?: TierType;
    featureVariable?: string;
    fallbackComponent?: React.ComponentType<Record<string, unknown>>;
  }
) {
  return function TierGatedComponent(props: P) {
    const { userTier } = useTierAccess();
    
    if (!userTier) {
      return <UpgradePrompt feature={options.moduleName} currentTier={null} />;
    }

    return (
      <TierGatedModule
        moduleName={options.moduleName}
        userTier={userTier}
        accessLevel="interactive"
        requiredTier={options.requiredTier}
        featureVariable={options.featureVariable}
        fallbackComponent={options.fallbackComponent}
      >
        <Component {...props} />
      </TierGatedModule>
    );
  };
}