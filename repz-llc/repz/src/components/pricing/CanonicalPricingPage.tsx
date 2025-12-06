// CANONICAL PRICING PAGE COMPONENT
// Single source of truth for all pricing displays

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { TierType, BillingCycle } from '@/constants/tiers';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { ElegantComparisonMatrix } from './ElegantComparisonMatrix';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Sparkles, Shield, Users, Zap, Star } from 'lucide-react';

interface CanonicalPricingPageProps {
  variant?: 'default' | 'compact' | 'elegant';
  showComparison?: boolean;
  title?: string;
  subtitle?: string;
}

export const CanonicalPricingPage: React.FC<CanonicalPricingPageProps> = ({
  variant = 'default',
  showComparison: initialShowComparison = false,
  title = "Choose Your Elite Coaching Path",
  subtitle = "Science-based optimization programs designed for serious athletes, biohackers, and longevity enthusiasts"
}) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [showComparison, setShowComparison] = useState(initialShowComparison);
  const [processing, setProcessing] = useState(false);

  // Canonical tier order
  const tiers: TierType[] = ['core', 'adaptive', 'performance', 'longevity'];

  const handleTierSelection = async (tier: TierType, billing: BillingCycle) => {
    if (!user) {
      toast.error('Please create an account first');
      navigate('/signup');
      return;
    }

    setProcessing(true);
    
    try {
      const { data, error } = await supabase.functions.invoke('create-comprehensive-checkout', {
        body: {
          tier,
          billingCycle: billing,
          returnUrl: window.location.origin + '/payment-success'
        }
      });

      if (error) throw error;

      if (data?.url) {
        window.location.href = data.url;
      } else {
        throw new Error('No checkout URL received');
      }
    } catch (error) {
      console.error('Checkout error:', error);
      toast.error('Failed to start checkout process');
    } finally {
      setProcessing(false);
    }
  };

  const renderHeader = () => (
    <motion.div 
      className="text-center pt-20 pb-16 px-4"
      initial={{ opacity: 0, y: -30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      {/* Brand Logo */}
      <div className="flex items-center justify-center gap-4 mb-8">
        <div className="lovable-logo" role="banner" aria-label="REPZ Coach Pro logo">
          <div className="lovable-bar bar-left"></div>
          <div className="lovable-bar bar-center"></div>
          <div className="lovable-bar bar-right"></div>
        </div>
        <span className="text-orange-400 font-bold text-5xl tracking-wider">REPZ</span>
      </div>

      <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-gray-100 to-yellow-400 bg-clip-text text-transparent mb-6">
        {title}
      </h1>
      <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed mb-8">
        {subtitle}
      </p>

      {/* Trust Indicators */}
      <div className="flex justify-center gap-8 md:gap-16 mt-12 mb-12">
        <div className="text-center">
          <div className="text-3xl md:text-4xl font-bold text-yellow-400">2,847</div>
          <div className="text-gray-400 text-sm md:text-base">Athletes Transformed</div>
        </div>
        <div className="text-center">
          <div className="text-3xl md:text-4xl font-bold text-yellow-400">96%</div>
          <div className="text-gray-400 text-sm md:text-base">Client Retention</div>
        </div>
        <div className="text-center">
          <div className="text-3xl md:text-4xl font-bold text-yellow-400">4.9/5</div>
          <div className="text-gray-400 text-sm md:text-base">Average Rating</div>
        </div>
      </div>

      {/* View Toggle */}
      <div className="inline-flex gap-4 p-1 rounded-lg bg-gray-800/50 backdrop-blur-sm border border-gray-700">
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
          Detailed Comparison
        </button>
      </div>
    </motion.div>
  );

  const renderTierCards = () => (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16"
    >
      {tiers.map((tier, idx) => (
        <motion.div
          key={tier}
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: idx * 0.1, duration: 0.5 }}
        >
          <UnifiedTierCard
            tier={tier}
            variant="detailed"
            onSelect={(selectedTier) => handleTierSelection(selectedTier, 'monthly')}
          />
        </motion.div>
      ))}
    </motion.div>
  );

  const renderSocialProof = () => (
    <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl border border-gray-700 p-8 mb-16">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-4">Trusted by Elite Athletes</h3>
        <div className="flex justify-center gap-2 mb-4">
          {[...Array(5)].map((_, i) => (
            <Star key={i} className="w-6 h-6 fill-yellow-400 text-yellow-400" />
          ))}
        </div>
        <p className="text-gray-300 italic">
          "The most comprehensive and scientifically-backed coaching platform I've ever used. 
          The longevity protocols are game-changing."
        </p>
        <div className="text-gray-400 mt-2">- Professional Athlete, Performance Suite</div>
      </div>
    </div>
  );

  const renderGuarantees = () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
      <div className="text-center">
        <Shield className="w-12 h-12 text-green-400 mx-auto mb-4" />
        <h4 className="text-lg font-semibold text-white mb-2">Secure & Private</h4>
        <p className="text-gray-400 text-sm">Your health data is encrypted and never shared</p>
      </div>
      <div className="text-center">
        <Users className="w-12 h-12 text-blue-400 mx-auto mb-4" />
        <h4 className="text-lg font-semibold text-white mb-2">Expert Coaches</h4>
        <p className="text-gray-400 text-sm">Certified professionals with advanced degrees</p>
      </div>
      <div className="text-center">
        <Zap className="w-12 h-12 text-purple-400 mx-auto mb-4" />
        <h4 className="text-lg font-semibold text-white mb-2">Proven Results</h4>
        <p className="text-gray-400 text-sm">Average 23% performance improvement</p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-800">
      <div className="container mx-auto px-4 py-16">
        {renderHeader()}
        
        <div className="px-4 pb-20">
          {!showComparison ? (
            renderTierCards()
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <ElegantComparisonMatrix onSelectTier={handleTierSelection} />
            </motion.div>
          )}
        </div>

        {renderSocialProof()}
        {renderGuarantees()}

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
    </div>
  );
};

export default CanonicalPricingPage;