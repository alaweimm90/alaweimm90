import React from 'react';
import { UnifiedTierCard } from './unified-tier-card';
import { FloatingParticles } from './floating-particles';
import { TierType } from '@/constants/tiers';

interface DesignSystemShowcaseProps {
  className?: string;
}

export const DesignSystemShowcase: React.FC<DesignSystemShowcaseProps> = ({ className = "" }) => {
  const tiers: TierType[] = ['core', 'adaptive', 'performance', 'longevity'];

  return (
    <div className={`min-h-screen bg-primary-elegant py-20 ${className}`}>
      <div className="container-premium">
        <div className="text-center mb-16">
          <h1 className="heading-hero mb-6">
            REPZ Design System
          </h1>
          <p className="text-premium max-w-3xl mx-auto">
            Premium glass morphism effects with sophisticated tier theming and elegant animations.
            Experience the signature visual identity of the REPZ platform.
          </p>
        </div>

        {/* Glass Morphism Tier Cards Showcase */}
        <div className="grid-premium mb-20">
          {tiers.map((tier) => (
            <div key={tier} className="relative">
              <FloatingParticles tier={tier} count={3} />
              <UnifiedTierCard
                tier={tier}
                variant="detailed"
                billingCycle="monthly"
                showFeatures={true}
                interactive={true}
              />
            </div>
          ))}
        </div>

        {/* Animation Showcase */}
        <div className="mb-20">
          <h2 className="heading-section text-center mb-12">Premium Animations</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="panel-glass p-8 text-center animate-elegant-float">
              <h3 className="text-xl font-bold text-white mb-4">Elegant Float</h3>
              <p className="text-white/80">8-second sophisticated movement</p>
            </div>
            <div className="panel-frost p-8 text-center animate-sophisticated-pulse">
              <h3 className="text-xl font-bold text-white mb-4">Sophisticated Pulse</h3>
              <p className="text-white/80">Premium breathing effect</p>
            </div>
            <div className="panel-velvet p-8 text-center animate-premium-glow">
              <h3 className="text-xl font-bold text-white mb-4">Premium Glow</h3>
              <p className="text-white/80">Tier-based highlighting</p>
            </div>
          </div>
        </div>

        {/* Glass Panel Variations */}
        <div className="mb-20">
          <h2 className="heading-section text-center mb-12">Glass Panel System</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="panel-glass p-6 text-center">
              <h3 className="text-lg font-bold text-white mb-2">Glass</h3>
              <p className="text-white/70 text-sm">Ultra-premium blur</p>
            </div>
            <div className="panel-frost p-6 text-center">
              <h3 className="text-lg font-bold text-white mb-2">Frost</h3>
              <p className="text-white/70 text-sm">Subtle elegance</p>
            </div>
            <div className="panel-velvet p-6 text-center">
              <h3 className="text-lg font-bold text-white mb-2">Velvet</h3>
              <p className="text-white/70 text-sm">Rich depth</p>
            </div>
            <div className="panel-hero p-6 text-center">
              <h3 className="text-lg font-bold text-white mb-2">Hero</h3>
              <p className="text-white/70 text-sm">Maximum impact</p>
            </div>
          </div>
        </div>

        {/* Color System */}
        <div className="mb-20">
          <h2 className="heading-section text-center mb-12">REPZ Color System</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-20 h-20 rounded-full mx-auto mb-4 bg-tier-core shadow-glow-blue"></div>
              <h3 className="font-bold text-white">Core</h3>
              <p className="text-sm text-white/70">Trust Blue</p>
            </div>
            <div className="text-center">
              <div className="w-20 h-20 rounded-full mx-auto mb-4 bg-tier-adaptive shadow-glow-orange"></div>
              <h3 className="font-bold text-white">Adaptive</h3>
              <p className="text-sm text-white/70">REPZ Orange</p>
            </div>
            <div className="text-center">
              <div className="w-20 h-20 rounded-full mx-auto mb-4 bg-tier-performance shadow-glow-purple"></div>
              <h3 className="font-bold text-white">Performance</h3>
              <p className="text-sm text-white/70">Sophistication Purple</p>
            </div>
            <div className="text-center">
              <div className="w-20 h-20 rounded-full mx-auto mb-4 bg-tier-longevity shadow-glow-gold"></div>
              <h3 className="font-bold text-white">Longevity</h3>
              <p className="text-sm text-white/70">Luxury Gold</p>
            </div>
          </div>
        </div>

        {/* REPZ Logo Showcase */}
        <div className="text-center">
          <h2 className="heading-section mb-12">REPZ Brand Identity</h2>
          <div className="flex flex-col md:flex-row items-center justify-center gap-8">
            <div className="repz-brand repz-brand-md">
              <div className="repz-logo">
                <div className="repz-bar repz-bar-left"></div>
                <div className="repz-bar repz-bar-center"></div>
                <div className="repz-bar repz-bar-right"></div>
              </div>
              <span className="repz-text">REPZ</span>
            </div>
            <div className="text-left">
              <p className="text-white font-semibold">Official Brand Colors</p>
              <p className="text-white/70">REPZ Orange: #F15B23</p>
              <p className="text-white/70">REPZ Black: #000000</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DesignSystemShowcase;