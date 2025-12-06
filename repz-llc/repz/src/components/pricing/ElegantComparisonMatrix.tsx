import React, { useState } from 'react';
import { Check, X, Crown, ChevronDown, ChevronUp, TrendingUp, Users, Clock, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { TierType, BillingCycle, CONVERSION_OPTIMIZED_TIERS, getTierPrice, calculateSavings } from '@/constants/tiers';
import { TIER_FEATURES, FEATURE_DISPLAY_NAMES, FEATURE_CATEGORIES } from '@/constants/featureMatrix';

// Tier data for comparison
const COMPARISON_DATA = {
  tiers: ['core', 'adaptive', 'performance', 'longevity'] as TierType[],
  tierInfo: {
    core: {
      name: 'Core Program',
      subtitle: 'Essential foundation',
      color: '#0091FF',
      gradient: 'linear-gradient(135deg, #0091FF 0%, #0066CC 100%)'
    },
    adaptive: {
      name: 'Adaptive Engine',
      subtitle: 'Smart progression',
      color: '#E87900',
      gradient: 'linear-gradient(135deg, #E87900 0%, #CC6600 100%)'
    },
    performance: {
      name: 'Performance Suite',
      subtitle: 'AI-powered excellence',
      color: '#6300A6',
      gradient: 'linear-gradient(135deg, #6300A6 0%, #4A0080 100%)',
      badge: 'MOST POPULAR'
    },
    longevity: {
      name: 'Longevity Concierge',
      subtitle: 'Elite biohacking',
      color: '#D4AF37',
      gradient: 'linear-gradient(135deg, #D4AF37 0%, #B8941F 100%)',
      badge: 'LIMITED – 5 SPOTS'
    }
  }
};

// Billing options
const BILLING_OPTIONS = [
  { id: 'monthly' as BillingCycle, label: 'Monthly', discount: 0 },
  { id: 'quarterly' as BillingCycle, label: 'Quarterly', discount: 5 },
  { id: 'semiannual' as BillingCycle, label: 'Semi-Annual', discount: 10 },
  { id: 'annual' as BillingCycle, label: 'Annual', discount: 20 }
];

interface ElegantComparisonMatrixProps {
  onSelectTier: (tier: TierType, billing: BillingCycle) => void;
}

export const ElegantComparisonMatrix: React.FC<ElegantComparisonMatrixProps> = ({ onSelectTier }) => {
  const [selectedBilling, setSelectedBilling] = useState<BillingCycle>('annual');
  const [hoveredTier, setHoveredTier] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['Core Platform', 'Convenience & Tech']));
  
  const getFeatureDisplay = (tier: TierType, featureKey: string) => {
    const features = TIER_FEATURES[tier];
    if (!features) return null;
    
    const value = features[featureKey as keyof typeof features];
    const tierColor = COMPARISON_DATA.tierInfo[tier].color;
    
    if (value === true) return <Check className="w-5 h-5" style={{ color: tierColor }} />;
    if (value === false) return <X className="w-5 h-5 text-gray-600" />;
    if (typeof value === 'string') {
      return <span className="text-sm font-medium" style={{ color: tierColor }}>{value}</span>;
    }
    if (typeof value === 'number' && value > 0) {
      return <span className="text-sm font-medium" style={{ color: tierColor }}>{value}</span>;
    }
    return <X className="w-5 h-5 text-gray-600" />;
  };
  
  const calculatePrice = (tier: TierType) => {
    return Math.round(getTierPrice(tier, selectedBilling) / (selectedBilling === 'monthly' ? 1 : 
      selectedBilling === 'quarterly' ? 3 : selectedBilling === 'semiannual' ? 6 : 12));
  };

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };
  
  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #0D0D0D 0%, #1A1A1A 100%)' }}>
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute w-[600px] h-[600px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(241, 91, 35, 0.1) 0%, transparent 50%)',
            filter: 'blur(100px)',
            top: '-200px',
            right: '-200px'
          }}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 py-16">
        {/* Header */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-4xl font-bold mb-4" style={{
            background: 'linear-gradient(135deg, #fff 0%, #F15B23 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Complete Feature Comparison
          </h1>
          <p className="text-lg text-gray-400">
            Choose your path to evidence-based transformation
          </p>
        </motion.div>
        
        {/* Billing Selector */}
        <div className="flex justify-center mb-12">
          <div className="inline-flex rounded-lg p-1" style={{ background: 'rgba(255, 255, 255, 0.05)' }}>
            {BILLING_OPTIONS.map((option) => (
              <motion.button
                key={option.id}
                onClick={() => setSelectedBilling(option.id)}
                className="relative px-6 py-2 rounded-lg font-medium text-sm transition-all"
                style={{
                  background: selectedBilling === option.id ? 'rgba(241, 91, 35, 0.2)' : 'transparent',
                  color: selectedBilling === option.id ? '#F15B23' : 'rgba(255, 255, 255, 0.6)',
                  border: selectedBilling === option.id ? '1px solid #F15B23' : '1px solid transparent'
                }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {option.label}
                {option.discount > 0 && (
                  <span className="ml-1 text-xs text-green-400">{option.discount}%</span>
                )}
              </motion.button>
            ))}
          </div>
        </div>
        
        {/* Comparison Table */}
        <div className="rounded-2xl overflow-hidden" style={{
          background: 'rgba(255, 255, 255, 0.03)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.06)'
        }}>
          {/* Tier Headers */}
          <div className="grid grid-cols-5">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-400">Features</h3>
            </div>
            {COMPARISON_DATA.tiers.map((tier) => {
              const info = COMPARISON_DATA.tierInfo[tier];
              return (
                <motion.div
                  key={tier}
                  className="p-6 text-center relative border-l"
                  style={{ borderColor: 'rgba(255, 255, 255, 0.06)' }}
                  onHoverStart={() => setHoveredTier(tier)}
                  onHoverEnd={() => setHoveredTier(null)}
                  animate={{
                    background: hoveredTier === tier ? 'rgba(255, 255, 255, 0.02)' : 'transparent'
                  }}
                >
                  {'badge' in info && info.badge && (
                    <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                      <div className="px-3 py-1 rounded-full text-xs font-bold text-black"
                        style={{ background: info.gradient }}>
                        {info.badge}
                      </div>
                    </div>
                  )}
                  
                  <div className="mb-2">
                    <div className="w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center"
                      style={{
                        background: info.gradient,
                        boxShadow: hoveredTier === tier ? `0 0 20px ${info.color}50` : 'none'
                      }}>
                      {tier === 'longevity' && <Crown className="w-6 h-6 text-black" />}
                      {tier === 'performance' && <TrendingUp className="w-6 h-6 text-white" />}
                      {tier === 'adaptive' && <Zap className="w-6 h-6 text-white" />}
                      {tier === 'core' && <Users className="w-6 h-6 text-white" />}
                    </div>
                    <h4 className="font-bold text-lg mb-1" style={{ color: info.color }}>
                      {info.name}
                    </h4>
                    <p className="text-xs text-gray-500 mb-3">{info.subtitle}</p>
                  </div>
                  
                  <div className="mb-4">
                    <motion.div
                      key={`${tier}-${selectedBilling}`}
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="text-3xl font-bold"
                      style={{ color: tier === 'longevity' ? '#D4AF37' : '#fff' }}
                    >
                      ${calculatePrice(tier)}
                      <span className="text-sm font-normal opacity-70">/month</span>
                    </motion.div>
                    {selectedBilling === 'annual' && (
                      <p className="text-xs text-green-400 mt-1">
                        Save ${calculateSavings(tier, 'annual')}/year
                      </p>
                    )}
                  </div>
                  
                  <motion.button
                    className="px-4 py-2 rounded-lg text-sm font-medium w-full"
                    style={{
                      background: info.gradient,
                      color: tier === 'longevity' ? '#000' : '#fff'
                    }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => onSelectTier(tier, selectedBilling)}
                  >
                    Choose {info.name}
                  </motion.button>
                </motion.div>
              );
            })}
          </div>
          
          {/* Feature Categories */}
          {Object.entries(FEATURE_CATEGORIES).map(([categoryName, features], catIdx) => (
            <div key={catIdx}>
              <motion.div 
                className="grid grid-cols-5 bg-black bg-opacity-30 cursor-pointer"
                onClick={() => toggleCategory(categoryName)}
                whileHover={{ backgroundColor: 'rgba(0, 0, 0, 0.4)' }}
              >
                <div className="col-span-5 px-6 py-3 flex items-center justify-between">
                  <h4 className="text-xs font-bold text-gray-400 tracking-wider">
                    {categoryName.toUpperCase()}
                  </h4>
                  {expandedCategories.has(categoryName) ? 
                    <ChevronUp className="w-4 h-4 text-gray-400" /> : 
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  }
                </div>
              </motion.div>
              
              <AnimatePresence>
                {expandedCategories.has(categoryName) && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    {features.map((featureKey, featIdx) => {
                      const displayName = FEATURE_DISPLAY_NAMES[featureKey as keyof typeof FEATURE_DISPLAY_NAMES] || featureKey;
                      
                      return (
                        <motion.div
                          key={featIdx}
                          className="grid grid-cols-5"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: featIdx * 0.05 }}
                          style={{
                            borderTop: '1px solid rgba(255, 255, 255, 0.03)'
                          }}
                        >
                          <div className="p-4 flex items-center">
                            <span className="text-sm text-gray-300">{displayName}</span>
                          </div>
                          
                          {COMPARISON_DATA.tiers.map((tier) => (
                            <div 
                              key={tier}
                              className="p-4 flex items-center justify-center border-l"
                              style={{ 
                                borderColor: 'rgba(255, 255, 255, 0.06)',
                                background: hoveredTier === tier ? 'rgba(255, 255, 255, 0.01)' : 'transparent'
                              }}
                            >
                              {getFeatureDisplay(tier, featureKey)}
                            </div>
                          ))}
                        </motion.div>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
          
          {/* Bottom CTA Row */}
          <div className="grid grid-cols-5 p-6" style={{ borderTop: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <div />
            {COMPARISON_DATA.tiers.map((tier) => {
              const info = COMPARISON_DATA.tierInfo[tier];
              return (
                <div key={tier} className="text-center">
                  <motion.button
                    className="px-6 py-3 rounded-xl text-sm font-medium w-full"
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${info.color}30`,
                      color: info.color
                    }}
                    whileHover={{ 
                      scale: 1.05,
                      background: `${info.color}20`,
                      borderColor: info.color
                    }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => onSelectTier(tier, selectedBilling)}
                  >
                    Choose {info.name}
                  </motion.button>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Trust indicators */}
        <motion.div
          className="mt-16 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex justify-center gap-8 text-sm text-gray-500">
            <span>Secure checkout via Stripe</span>
            <span>Cancel anytime</span>
            <span>Start immediately</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ElegantComparisonMatrix;