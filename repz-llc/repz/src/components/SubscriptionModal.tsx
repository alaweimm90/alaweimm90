import React from 'react';
import { MultiStepIntakeForm } from './intake/MultiStepIntakeForm';
import { TierType } from '@/constants/tiers';

interface SubscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedTier: string;
  tierPrice: string;
  onChangeTier?: () => void;
}

export const SubscriptionModal: React.FC<SubscriptionModalProps> = ({ 
  isOpen, 
  onClose, 
  selectedTier, 
  tierPrice, 
  onChangeTier
}) => {
  // Map display names to tier keys for proper styling
  const tierMapping: Record<string, TierType> = {
    'Core Program': 'core',
    'Adaptive Engine': 'adaptive', 
    'Prime Suite': 'performance',
    'Elite Concierge': 'longevity',
    // Also handle legacy names
    'core': 'core',
    'adaptive': 'adaptive',
    'performance': 'performance', 
    'longevity': 'longevity'
  };

  const mappedTier = tierMapping[selectedTier] || 'core';

  return (
    <MultiStepIntakeForm
      selectedTier={mappedTier}
      onComplete={(data) => {
        console.log('Form completed with data:', data);
        onClose();
      }}
      onTierChange={(tier) => {
        if (onChangeTier) onChangeTier();
      }}
      isOpen={isOpen}
      onClose={onClose}
    />
  );
};