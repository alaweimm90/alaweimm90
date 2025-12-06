// src/components/pricing/MobileElegantPricing.tsx
// Mobile-optimized elegant pricing with swipe and progressive disclosure

import React, { useState } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import { Check, X, ChevronDown, ChevronUp, Sparkles, Shield, Zap, Activity, Crown } from 'lucide-react';
import { TierType, BillingCycle, getTierConfig, getTierPrice, calculateSavings } from '@/constants/tiers';
import { ComingSoonOverlay } from '@/components/ui/coming-soon-overlay';

// Design constants
const MOBILE_DESIGN = {
  tiers: {
    core: { icon: Shield, gradient: 'from-blue-500 to-blue-600' },
    adaptive: { icon: Zap, gradient: 'from-orange-500 to-orange-600' },
    performance: { icon: Activity, gradient: 'from-purple-500 to-purple-600' },
    longevity: { icon: Crown, gradient: 'from-yellow-500 to-yellow-600' }
  }
};

// Tier features for mobile
const MOBILE_FEATURES = {
  core: {
    highlights: [
      'Macro-based training',
      'Basic nutrition plan',
      'Q&A support (72hr)',
      '5 messages/month'
    ],
    locked: ['Weekly check-ins', 'Biomarker tracking', 'AI Assistant']
  },
  adaptive: {
    highlights: [
      'Everything in Core +',
      'Biomarker tracking',
      'Auto grocery lists',
      'Weekly check-ins',
      '15 messages (48hr)'
    ],
    locked: ['AI Assistant', 'PEDs protocols']
  },
  performance: {
    highlights: [
      'Everything in Adaptive +',
      'AI fitness assistant',
      'PEDs & peptides',
      'Nootropics guide',
      'Unlimited messaging (24hr)'
    ],
    locked: ['Bioregulators', 'In-person training']
  },
  longevity: {
    highlights: [
      'Everything in Performance +',
      'Bioregulators protocols',
      '2x weekly in-person',
      'Custom cycling',
      '12hr elite response'
    ],
    locked: []
  }
};

interface MobileElegantPricingProps {
  onSelectTier: (tier: TierType, billing: BillingCycle) => void;
}

interface MobileTierCardProps {
  tier: TierType;
  index: number;
  onSelect: (tier: TierType, billing: BillingCycle) => void;
}

const MobileTierCard: React.FC<MobileTierCardProps> = ({ tier, index, onSelect }) => {
  const [selectedBilling, setSelectedBilling] = useState<BillingCycle>('annual');
  const [isExpanded, setIsExpanded] = useState(false);
  const [showLocked, setShowLocked] = useState(false);
  
  const config = getTierConfig(tier);
  const design = MOBILE_DESIGN.tiers[tier];
  const features = MOBILE_FEATURES[tier];
  const Icon = design.icon;
  
  const currentPrice = getTierPrice(tier, selectedBilling);
  const monthlyEquivalent = selectedBilling === 'monthly' ? currentPrice : 
    Math.round(currentPrice / (selectedBilling === 'quarterly' ? 3 : 
    selectedBilling === 'semiannual' ? 6 : 12));
  const annualSavings = calculateSavings(tier, 'annual');
  
  // Swipe handling
  const x = useMotionValue(0);
  const opacity = useTransform(x, [-100, 0, 100], [0.5, 1, 0.5]);
  const scale = useTransform(x, [-100, 0, 100], [0.95, 1, 0.95]);
  
  return (
    <ComingSoonOverlay disabled={true} message="New, modern dashboards launching soon for all subscription plans.">
      <motion.div
        className="mb-4"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        style={{ x, opacity, scale }}
        drag="x"
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.2}
      >
      <div
        className={`rounded-2xl overflow-hidden ${
          tier === 'longevity' ? 'bg-black' : 'bg-gray-900'
        }`}
        style={{
          boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
          border: tier === 'longevity' ? '2px solid #D4AF37' : '1px solid rgba(255, 255, 255, 0.1)'
        }}
      >
        {/* Header with gradient */}
        <div className={`p-6 bg-gradient-to-r ${design.gradient}`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <motion.div
                className="p-2 rounded-lg bg-white/20"
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <Icon className="w-6 h-6 text-white" />
              </motion.div>
              <div>
                <h3 className="text-xl font-bold text-white">{config?.displayName || tier}</h3>
                {tier === 'performance' && (
                  <span className="text-xs text-white/80">MOST POPULAR</span>
                )}
                {tier === 'longevity' && (
                  <span className="text-xs text-white/80">LIMITED AVAILABILITY</span>
                )}
              </div>
            </div>
            
            <motion.button
              onClick={() => setIsExpanded(!isExpanded)}
              whileTap={{ scale: 0.9 }}
            >
              {isExpanded ? 
                <ChevronUp className="w-5 h-5 text-white" /> : 
                <ChevronDown className="w-5 h-5 text-white" />
              }
            </motion.button>
          </div>
          
          {/* Price */}
          <div className="text-center">
            <div className="text-4xl font-bold text-white mb-1">
              ${monthlyEquivalent}
              <span className="text-lg font-normal opacity-80">/mo</span>
            </div>
            {selectedBilling === 'annual' && (
              <p className="text-sm text-white/90">
                Save ${annualSavings}/year
              </p>
            )}
          </div>
        </div>
        
        {/* Content */}
        <div className={`p-6 ${tier === 'longevity' ? 'bg-gray-900' : ''}`}>
          {/* Billing selector */}
          <div className="grid grid-cols-4 gap-2 mb-6">
            {(['monthly', 'quarterly', 'semiannual', 'annual'] as BillingCycle[]).map((cycle) => {
              const isSelected = selectedBilling === cycle;
              const discount = cycle === 'quarterly' ? 5 : cycle === 'semiannual' ? 10 : cycle === 'annual' ? 20 : 0;
              
              return (
                <motion.button
                  key={cycle}
                  onClick={() => setSelectedBilling(cycle)}
                  className={`py-2 rounded-lg text-xs font-medium ${
                    isSelected
                      ? tier === 'longevity'
                        ? 'bg-yellow-600 text-black'
                        : 'bg-white text-gray-900'
                      : tier === 'longevity'
                        ? 'bg-gray-800 text-gray-400'
                        : 'bg-gray-800 text-gray-400'
                  }`}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="block">{cycle.charAt(0).toUpperCase() + cycle.slice(1, 3)}</span>
                  {discount > 0 && <span className="text-xs">-{discount}%</span>}
                </motion.button>
              );
            })}
          </div>
          
          {/* Loss aversion for monthly */}
          {selectedBilling === 'monthly' && (
            <motion.p
              className="text-sm text-red-400 text-center mb-4"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              You could save ${annualSavings}/year with annual
            </motion.p>
          )}
          
          {/* Features preview */}
          <AnimatePresence>
            {!isExpanded ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {/* Show first 3 features */}
                <div className="space-y-2 mb-4">
                  {features.highlights.slice(0, 3).map((feature, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                      <span className={`text-sm ${
                        tier === 'longevity' ? 'text-yellow-100' : 'text-gray-300'
                      }`}>
                        {feature}
                      </span>
                    </div>
                  ))}
                </div>
                
                <p className={`text-xs ${
                  tier === 'longevity' ? 'text-yellow-400' : 'text-gray-500'
                }`}>
                  +{features.highlights.length - 3} more features
                </p>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {/* All features */}
                <div className="space-y-2 mb-4">
                  {features.highlights.map((feature, idx) => (
                    <motion.div
                      key={idx}
                      className="flex items-center gap-2"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                    >
                      <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                      <span className={`text-sm ${
                        tier === 'longevity' ? 'text-yellow-100' : 'text-gray-300'
                      }`}>
                        {feature}
                      </span>
                    </motion.div>
                  ))}
                </div>
                
                {/* Locked features */}
                {features.locked.length > 0 && (
                  <div className="pt-4 border-t border-gray-700">
                    <button
                      onClick={() => setShowLocked(!showLocked)}
                      className={`text-sm font-medium ${
                        tier === 'longevity' ? 'text-yellow-400' : 'text-gray-400'
                      }`}
                    >
                      {features.locked.length} features at higher tiers
                    </button>
                    
                    {showLocked && (
                      <div className="mt-2 space-y-1">
                        {features.locked.map((locked, idx) => (
                          <div key={idx} className="flex items-center gap-2">
                            <X className="w-4 h-4 text-red-400" />
                            <span className="text-sm text-gray-500">{locked}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* CTA Button */}
          <motion.button
            className={`w-full py-4 rounded-xl font-bold mt-6 ${
              tier === 'longevity'
                ? 'bg-gradient-to-r from-yellow-500 to-yellow-600 text-black'
                : 'bg-gradient-to-r ' + design.gradient + ' text-white'
            }`}
            style={{
              boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)'
            }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onSelect(tier, selectedBilling)}
            disabled={true}
          >
            <Sparkles className="inline w-4 h-4 mr-2" />
            Get Started
          </motion.button>
        </div>
      </div>
      </motion.div>
    </ComingSoonOverlay>
  );
};

export const MobileElegantPricing: React.FC<MobileElegantPricingProps> = ({ onSelectTier }) => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black">
      {/* Header */}
      <div className="sticky top-0 z-40 bg-black/80 backdrop-blur-lg border-b border-gray-800">
        <div className="px-4 py-6 text-center">
          <h1 className="text-3xl font-bold mb-2">
            <span style={{
              background: 'linear-gradient(135deg, #fff 0%, #F15B23 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              Choose Your Path
            </span>
          </h1>
          <p className="text-sm text-gray-400">Swipe to explore • Tap to expand</p>
        </div>
      </div>
      
      {/* Tier cards */}
      <div className="p-4 pb-20">
        {(['core', 'adaptive', 'performance', 'longevity'] as TierType[]).map((tier, idx) => (
          <MobileTierCard
            key={tier}
            tier={tier}
            index={idx}
            onSelect={onSelectTier}
          />
        ))}
      </div>
      
      {/* Bottom bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-black/80 backdrop-blur-lg border-t border-gray-800 p-4">
        <div className="flex justify-center gap-4 text-xs text-gray-400">
          <span>Cancel anytime</span>
          <span>Secure checkout</span>
        </div>
      </div>
    </div>
  );
};

export default MobileElegantPricing;