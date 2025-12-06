import React, { useState } from 'react';
import {
  LayoutDashboard, Activity, Utensils, Camera, BarChart3,
  Bot, ShoppingCart, Plane, Brain, TestTube, Users,
  Calendar, Lock, Sparkles, TrendingUp, Clock, Zap,
  Crown, Shield, ChevronRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTierAccess } from '@/hooks/useTierAccess';

type TierType = 'core' | 'adaptive' | 'performance' | 'longevity';

// Design system matching your elegant aesthetic
const DASHBOARD_DESIGN = {
  tiers: {
    core: {
      primary: '#0091FF',
      gradient: 'linear-gradient(135deg, #0091FF 0%, #0066CC 100%)',
      glow: 'rgba(0, 145, 255, 0.3)',
      icon: Shield
    },
    adaptive: {
      primary: '#E87900',
      gradient: 'linear-gradient(135deg, #E87900 0%, #CC6600 100%)',
      glow: 'rgba(232, 121, 0, 0.3)',
      icon: Zap
    },
    performance: {
      primary: '#6300A6',
      gradient: 'linear-gradient(135deg, #6300A6 0%, #4A0080 100%)',
      glow: 'rgba(99, 0, 166, 0.3)',
      icon: Activity
    },
    longevity: {
      primary: '#D4AF37',
      gradient: 'linear-gradient(135deg, #D4AF37 0%, #B8941F 100%)',
      glow: 'rgba(212, 175, 55, 0.6)',
      icon: Crown
    }
  },
  glass: {
    background: 'rgba(255, 255, 255, 0.03)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.06)',
    hover: 'rgba(255, 255, 255, 0.05)'
  }
};

// Dashboard modules with tier requirements
const DASHBOARD_MODULES = [
  // Core modules
  { id: 'overview', name: 'Overview', icon: LayoutDashboard, requiredTier: 'core', description: 'Your fitness journey at a glance' },
  { id: 'training', name: 'Training Program', icon: Activity, requiredTier: 'core', description: 'Personalized workout plans' },
  { id: 'nutrition', name: 'Nutrition Plan', icon: Utensils, requiredTier: 'core', description: 'Meal plans and macro tracking' },
  
  // Adaptive+ modules
  { id: 'checkins', name: 'Weekly Check-ins', icon: Calendar, requiredTier: 'adaptive', description: 'Track progress with your coach' },
  { id: 'progress-photos', name: 'Progress Photos', icon: Camera, requiredTier: 'adaptive', description: 'Visual transformation tracking' },
  { id: 'auto-grocery', name: 'Auto Grocery Lists', icon: ShoppingCart, requiredTier: 'adaptive', description: 'Save 2 hours/week', isNew: true },
  { id: 'travel-workouts', name: 'Travel Workouts', icon: Plane, requiredTier: 'adaptive', description: 'Stay on track anywhere', isNew: true },
  { id: 'biomarkers', name: 'Biomarker Tracking', icon: TestTube, requiredTier: 'adaptive', description: 'Monitor health metrics', isNew: true },
  
  // Performance+ modules
  { id: 'ai-assistant', name: 'AI Fitness Assistant', icon: Bot, requiredTier: 'performance', description: 'Your 24/7 intelligent coach', isNew: true },
  { id: 'analytics', name: 'Advanced Analytics', icon: BarChart3, requiredTier: 'performance', description: 'Deep insights into progress' },
  { id: 'peds-protocols', name: 'PEDs Protocols', icon: Zap, requiredTier: 'performance', description: 'Advanced enhancement guidance', isNew: true },
  { id: 'nootropics', name: 'Nootropics Guide', icon: Brain, requiredTier: 'performance', description: 'Cognitive enhancement', isNew: true },
  
  // Longevity exclusive modules
  { id: 'bioregulators', name: 'Bioregulators', icon: Sparkles, requiredTier: 'longevity', description: 'Exclusive longevity compounds' },
  { id: 'in-person', name: 'In-Person Training', icon: Users, requiredTier: 'longevity', description: '2x weekly personal sessions' }
];

const TIER_HIERARCHY = ['core', 'adaptive', 'performance', 'longevity'];

interface ModuleCardProps {
  module: typeof DASHBOARD_MODULES[0];
  userTier: string;
  hasAccess: boolean;
  onSelect: () => void;
  onUpgrade: () => void;
}

const ModuleCard: React.FC<ModuleCardProps> = ({ module, userTier, hasAccess, onSelect, onUpgrade }) => {
  const [isHovered, setIsHovered] = useState(false);
  const Icon = module.icon;
  const design = DASHBOARD_DESIGN.tiers[userTier as keyof typeof DASHBOARD_DESIGN.tiers];
  
  return (
    <motion.div
      className="relative"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: hasAccess ? 1.02 : 1 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      {/* Glow effect */}
      <motion.div
        className="absolute inset-0 rounded-xl opacity-0"
        style={{ background: design.gradient, filter: 'blur(20px)' }}
        animate={{ opacity: isHovered && hasAccess ? 0.3 : 0 }}
      />
      
      <div
        className={`relative rounded-xl p-6 cursor-pointer transition-all ${
          hasAccess ? 'hover:border-opacity-30' : 'opacity-60'
        }`}
        style={{
          ...DASHBOARD_DESIGN.glass,
          borderColor: hasAccess ? design.primary : 'rgba(255, 255, 255, 0.06)',
          background: isHovered ? DASHBOARD_DESIGN.glass.hover : DASHBOARD_DESIGN.glass.background
        }}
        onClick={hasAccess ? onSelect : onUpgrade}
      >
        {/* New badge */}
        {module.isNew && hasAccess && (
          <motion.div
            className="absolute -top-2 -right-2 px-2 py-0.5 rounded-full bg-green-500 text-black text-xs font-bold"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", delay: 0.3 }}
          >
            NEW
          </motion.div>
        )}
        
        {/* Lock overlay for locked modules */}
        {!hasAccess && (
          <div className="absolute inset-0 rounded-xl bg-black/20 backdrop-blur-sm flex items-center justify-center">
            <Lock className="w-8 h-8 text-gray-400" />
          </div>
        )}
        
        {/* Icon */}
        <motion.div
          className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
          style={{
            background: hasAccess ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}
          animate={{
            boxShadow: isHovered && hasAccess ? `0 0 20px ${design.glow}` : 'none'
          }}
        >
          <Icon className="w-6 h-6" style={{ color: hasAccess ? design.primary : '#666' }} />
        </motion.div>
        
        {/* Content */}
        <h3 className="font-semibold mb-1 text-white">{module.name}</h3>
        <p className="text-sm text-gray-400 mb-3">{module.description}</p>
        
        {/* Access indicator */}
        {!hasAccess && (
          <p className="text-xs" style={{ color: design.primary }}>
            Requires {module.requiredTier}+ tier
          </p>
        )}
      </div>
    </motion.div>
  );
};

interface ElegantDashboardProps {
  userName?: string;
}

export const ElegantDashboard: React.FC<ElegantDashboardProps> = ({ userName = 'Athlete' }) => {
  const { userTier, hasMinimumTier } = useTierAccess();
  const [selectedModule, setSelectedModule] = useState('overview');
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [targetModule, setTargetModule] = useState<typeof DASHBOARD_MODULES[0] | null>(null);
  
  const currentTier = userTier || 'core';
  const design = DASHBOARD_DESIGN.tiers[currentTier as keyof typeof DASHBOARD_DESIGN.tiers];
  const TierIcon = design.icon;
  
  const hasAccess = (module: typeof DASHBOARD_MODULES[0]): boolean => {
    return hasMinimumTier(module.requiredTier as TierType);
  };
  
  const handleModuleClick = (module: typeof DASHBOARD_MODULES[0]) => {
    if (hasAccess(module)) {
      setSelectedModule(module.id);
    } else {
      setTargetModule(module);
      setShowUpgradeModal(true);
    }
  };
  
  // Group modules by tier
  const modulesByTier = {
    core: DASHBOARD_MODULES.filter(m => m.requiredTier === 'core'),
    adaptive: DASHBOARD_MODULES.filter(m => m.requiredTier === 'adaptive'),
    performance: DASHBOARD_MODULES.filter(m => m.requiredTier === 'performance'),
    longevity: DASHBOARD_MODULES.filter(m => m.requiredTier === 'longevity')
  };
  
  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(180deg, #0D0D0D 0%, #1A1A1A 100%)' }}>
      {/* Header */}
      <motion.header
        className="sticky top-0 z-40 border-b"
        style={{
          ...DASHBOARD_DESIGN.glass,
          borderColor: design.primary + '30'
        }}
        initial={{ y: -100 }}
        animate={{ y: 0 }}
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-white">Welcome back, {userName}</h1>
            <motion.div
              className="px-4 py-1.5 rounded-full flex items-center gap-2"
              style={{
                background: design.gradient,
                boxShadow: `0 0 20px ${design.glow}`
              }}
              whileHover={{ scale: 1.05 }}
            >
              <TierIcon className="w-4 h-4 text-white" />
              <span className="text-sm font-medium text-white">
                {currentTier.charAt(0).toUpperCase() + currentTier.slice(1)}
              </span>
            </motion.div>
          </div>
          
          {userTier !== 'longevity' && (
            <motion.button
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-orange-500 to-orange-600 text-white font-medium"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowUpgradeModal(true)}
            >
              Upgrade Plan
            </motion.button>
          )}
        </div>
      </motion.header>
      
      {/* Main content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Module sections */}
        {Object.entries(modulesByTier).map(([tier, modules]) => {
          if (modules.length === 0) return null;
          
          const tierDesign = DASHBOARD_DESIGN.tiers[tier as keyof typeof DASHBOARD_DESIGN.tiers];
          const isAccessible = hasMinimumTier(tier as TierType);
          
          return (
            <motion.section
              key={tier}
              className="mb-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1 }}
            >
              <div className="flex items-center gap-3 mb-6">
                <h2 
                  className="text-xl font-bold"
                  style={{
                    background: tierDesign.gradient,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                  }}
                >
                  {tier.charAt(0).toUpperCase() + tier.slice(1)} Features
                </h2>
                {!isAccessible && (
                  <span className="text-sm text-gray-500">
                    Upgrade to unlock
                  </span>
                )}
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {modules.map((module, idx) => (
                  <motion.div
                    key={module.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                  >
                    <ModuleCard
                      module={module}
                      userTier={currentTier}
                      hasAccess={hasAccess(module)}
                      onSelect={() => handleModuleClick(module)}
                      onUpgrade={() => handleModuleClick(module)}
                    />
                  </motion.div>
                ))}
              </div>
            </motion.section>
          );
        })}
      </div>
      
      {/* Upgrade Modal */}
      <AnimatePresence>
        {showUpgradeModal && (
          <motion.div
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 px-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowUpgradeModal(false)}
          >
            <motion.div
              className="max-w-md w-full rounded-2xl p-8"
              style={DASHBOARD_DESIGN.glass}
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-2xl font-bold text-white mb-4">Unlock Premium Features</h2>
              
              {targetModule && (
                <div className="mb-6">
                  <div className="flex items-center gap-3 mb-2">
                    <targetModule.icon className="w-6 h-6" style={{ color: design.primary }} />
                    <span className="font-semibold text-white">{targetModule.name}</span>
                  </div>
                  <p className="text-gray-400">{targetModule.description}</p>
                  <p className="text-sm mt-2" style={{ color: design.primary }}>
                    Available in {targetModule.requiredTier}+ tier
                  </p>
                </div>
              )}
              
              <div className="space-y-3">
                <motion.button
                  className="w-full py-3 rounded-lg bg-gradient-to-r from-orange-500 to-orange-600 text-white font-medium"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => window.location.href = '/pricing'}
                >
                  View Upgrade Options
                  <ChevronRight className="inline w-4 h-4 ml-2" />
                </motion.button>
                
                <button
                  className="w-full py-3 rounded-lg border border-gray-600 text-gray-400 font-medium hover:text-white transition-colors"
                  onClick={() => setShowUpgradeModal(false)}
                >
                  Maybe Later
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ElegantDashboard;