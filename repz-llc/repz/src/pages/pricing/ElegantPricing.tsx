// src/pages/pricing/ElegantPricing.tsx
// Main pricing page with elegant design system

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { TierType, BillingCycle } from '@/constants/tiers';
import { supabase } from '@/integrations/supabase/client';
import { ElegantComparisonMatrix } from '@/components/pricing/ElegantComparisonMatrix';
import { MobileElegantPricing } from '@/components/pricing/MobileElegantPricing';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { Sparkles } from 'lucide-react';

// Floating orb component for background
const FloatingOrb: React.FC<{ delay: number; size: string; position: { top?: string; bottom?: string; left?: string; right?: string } }> = ({ delay, size, position }) => (
  <motion.div
    className={`absolute ${size} rounded-full pointer-events-none`}
    style={{
      ...position,
      background: 'radial-gradient(circle, rgba(241, 91, 35, 0.15) 0%, transparent 50%)',
      filter: 'blur(80px)'
    }}
    animate={{
      x: [0, 100, 0],
      y: [0, -50, 0],
      scale: [1, 1.2, 1]
    }}
    transition={{
      duration: 15 + delay,
      repeat: Infinity,
      ease: "easeInOut",
      delay
    }}
  />
);

export const ElegantPricing: React.FC = () => {
  const [showComparison, setShowComparison] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<'glass' | 'floating' | 'minimal'>('glass');
  const [selectedBilling, setSelectedBilling] = useState<BillingCycle>('annual');
  const isMobile = useMediaQuery('(max-width: 768px)');
  const isTablet = useMediaQuery('(max-width: 1024px)');
  
  // Handle tier selection using canonical structure
  const handleTierSelection = async (tier: TierType, billing: BillingCycle) => {
    try {
      // Track conversion event
      if (typeof window !== 'undefined' && (window as Window & { analytics?: { track: (event: string, data: Record<string, unknown>) => void } }).analytics) {
        (window as Window & { analytics: { track: (event: string, data: Record<string, unknown>) => void } }).analytics.track('tier_selected', {
          tier,
          billing,
          variant: selectedVariant,
          device: isMobile ? 'mobile' : isTablet ? 'tablet' : 'desktop'
        });
      }
      
      // Use Supabase edge function for checkout
      const { data, error } = await fetch('/functions/v1/create-comprehensive-checkout', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`
        },
        body: JSON.stringify({ 
          tier, 
          billingCycle: billing,
          returnUrl: window.location.origin
        })
      }).then(res => res.json());
      
      if (error) {
        console.error('Checkout error:', error);
        return;
      }
      
      // Redirect to Stripe
      if (data?.url) {
        window.location.href = data.url;
      }
    } catch (error) {
      console.error('Failed to create checkout session:', error);
    }
  };
  
  // Mobile view
  if (isMobile) {
    return <MobileElegantPricing onSelectTier={handleTierSelection} />;
  }
  
  return (
    <div 
      className="min-h-screen relative overflow-hidden"
      style={{
        background: 'radial-gradient(ellipse at top, #0D0D0D 0%, #1A1A1A 50%, #0D0D0D 100%)'
      }}
    >
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <FloatingOrb delay={0} size="w-96 h-96" position={{ top: '-200px', left: '10%' }} />
        <FloatingOrb delay={5} size="w-64 h-64" position={{ bottom: '-100px', right: '20%' }} />
        <FloatingOrb delay={10} size="w-80 h-80" position={{ top: '50%', left: '50%' }} />
      </div>
      
      <div className="relative z-10">
        {/* Header */}
        <motion.div 
          className="text-center pt-20 pb-16 px-4"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.h1 
            className="text-7xl font-bold mb-6"
            style={{
              background: 'linear-gradient(135deg, #fff 0%, #F15B23 50%, #fff 100%)',
              backgroundSize: '200% 200%',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              animation: 'gradientShift 6s ease infinite',
              textShadow: '0 2px 20px rgba(241, 91, 35, 0.3)'
            }}
          >
            Elite Transformation Awaits
          </motion.h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
            Science-driven protocols. Proven results. Choose your path to excellence.
          </p>
          
          {/* View toggle */}
          <div className="inline-flex gap-4 p-1 rounded-lg" style={{ background: 'rgba(255, 255, 255, 0.05)' }}>
            <button
              onClick={() => setShowComparison(false)}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                !showComparison 
                  ? 'bg-gradient-to-r from-orange-500 to-orange-600 text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Sparkles className="inline w-4 h-4 mr-2" />
              Tier Cards
            </button>
            <button
              onClick={() => setShowComparison(true)}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                showComparison 
                  ? 'bg-gradient-to-r from-orange-500 to-orange-600 text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Comparison View
            </button>
          </div>
        </motion.div>
        
        {/* Content */}
        <div className="px-4 pb-20">
          <AnimatePresence mode="wait">
            {!showComparison ? (
              <motion.div
                key="cards"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -30 }}
                className="max-w-7xl mx-auto"
              >
                {/* Design variant selector (optional - for demo) */}
                {process.env.NODE_ENV === 'development' && (
                  <div className="flex justify-center gap-2 mb-8">
                    {(['glass', 'floating', 'minimal'] as const).map((variant) => (
                      <button
                        key={variant}
                        onClick={() => setSelectedVariant(variant)}
                        className={`px-4 py-1 rounded text-sm ${
                          selectedVariant === variant 
                            ? 'bg-orange-500 text-white' 
                            : 'bg-gray-800 text-gray-400'
                        }`}
                      >
                        {variant}
                      </button>
                    ))}
                  </div>
                )}
                
                {/* Tier Cards Grid */}
                <div className={`grid gap-6 ${isTablet ? 'grid-cols-2' : 'grid-cols-4'}`}>
                  {(['core', 'adaptive', 'performance', 'longevity'] as TierType[]).map((tier, idx) => (
                    <motion.div
                      key={tier}
                      initial={{ opacity: 0, y: 50 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.1, duration: 0.5 }}
                    >
                      <UnifiedTierCard
                        tier={tier}
                        variant="detailed"
                        onSelect={() => handleTierSelection(tier, selectedBilling)}
                        comingSoon={true}
                        comingSoonMessage="New, modern dashboards launching soon for all subscription plans."
                      />
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="comparison"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
              >
                <ElegantComparisonMatrix onSelectTier={handleTierSelection} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Trust indicators */}
        <motion.div 
          className="text-center pb-16 text-gray-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <div className="flex justify-center gap-8 text-sm">
            <span>✓ Cancel anytime</span>
            <span>✓ Secure checkout</span>
            <span>✓ Start immediately</span>
          </div>
        </motion.div>
      </div>
      
      {/* CSS for gradient animation */}
      <style>{`
        @keyframes gradientShift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
      `}</style>
    </div>
  );
};

export default ElegantPricing;