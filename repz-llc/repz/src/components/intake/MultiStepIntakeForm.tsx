import React, { useState, useEffect, CSSProperties } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronRight,
  ChevronLeft,
  Check,
  X,
  Shield,
  Zap,
  Activity,
  Crown,
  User as UserIcon,
  LogIn
} from 'lucide-react';
import { TierType } from '@/constants/tiers';
import { supabase } from '@/integrations/supabase/client';
import { Button } from '@/ui/atoms/Button';
import { Dialog, DialogContent } from '@/ui/molecules/Dialog';
import type { User } from '@supabase/supabase-js';

// Comprehensive tier styling system
const TIER_STYLING = {
  core: { 
    color: '#0091FF', 
    icon: Shield, 
    name: 'Core Program',
    description: 'Essential coaching for your fitness journey',
    price: '$97/month',
    gradient: 'linear-gradient(135deg, #0091FF 0%, #0066CC 100%)'
  },
  adaptive: { 
    color: '#E87900', 
    icon: Zap, 
    name: 'Adaptive Engine',
    description: 'Smart training that adapts to you',
    price: '$197/month',
    gradient: 'linear-gradient(135deg, #E87900 0%, #CC6600 100%)'
  },
  performance: { 
    color: '#6300A6', 
    icon: Activity, 
    name: 'Performance Suite',
    description: 'Elite coaching for peak performance',
    price: '$297/month',
    gradient: 'linear-gradient(135deg, #6300A6 0%, #4A0080 100%)'
  },
  longevity: { 
    color: '#D4AF37', 
    icon: Crown, 
    name: 'Longevity Concierge',
    description: 'Complete health optimization',
    price: '$447/month',
    gradient: 'linear-gradient(135deg, #D4AF37 0%, #B8941F 100%)'
  }
};

interface MultiStepIntakeFormProps {
  selectedTier?: TierType;
  onComplete?: (data: FormData) => void;
  onTierChange?: (tier: TierType) => void;
  isOpen?: boolean;
  onClose?: () => void;
}

interface FormData {
  // Personal Info
  name: string;
  email: string;
  age: number | '';
  gender: 'male' | 'female' | 'other' | '';
  
  // Goals & Experience
  primaryGoal: string;
  experience: string;
  currentWeight: number | '';
  targetWeight: number | '';
  
  // Lifestyle
  trainingDays: number | '';
  sleepHours: number | '';
  stressLevel: number | '';
  
  // Selected tier
  tier: TierType;
}

const FORM_STEPS = [
  { id: 'personal', title: 'Personal Info', description: 'Tell us about yourself' },
  { id: 'goals', title: 'Goals & Body', description: 'Your transformation goals' },
  { id: 'lifestyle', title: 'Lifestyle', description: 'Training & recovery habits' },
  { id: 'tier', title: 'Choose Plan', description: 'Select your coaching tier' },
  { id: 'complete', title: 'Complete', description: 'Finalize your profile' }
];

export const MultiStepIntakeForm: React.FC<MultiStepIntakeFormProps> = ({
  selectedTier = 'core',
  onComplete,
  onTierChange,
  isOpen = true,
  onClose
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [user, setUser] = useState<User | null>(null);
  
  // Validate selectedTier to prevent undefined errors
  const validTier = (selectedTier && TIER_STYLING[selectedTier]) ? selectedTier : 'core';
  
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    age: '',
    gender: '',
    primaryGoal: '',
    experience: '',
    currentWeight: '',
    targetWeight: '',
    trainingDays: '',
    sleepHours: '',
    stressLevel: '',
    tier: validTier
  });

  // Get current tier styling with safe fallback
  const currentTierStyling = TIER_STYLING[formData.tier as TierType] || TIER_STYLING.core;
  
  // Sync selectedTier prop with internal formData.tier
  useEffect(() => {
    if (selectedTier && TIER_STYLING[selectedTier] && selectedTier !== formData.tier) {
      updateFormData('tier', selectedTier);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTier]);
  
  // Check authentication status
  useEffect(() => {
    const checkAuth = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setUser(user);
        // Pre-fill form with user data
        setFormData(prev => ({
          ...prev,
          name: user.user_metadata?.name || '',
          email: user.email || ''
        }));
      }
    };
    
    checkAuth();
  }, []);

  const updateFormData = (field: keyof FormData, value: FormData[keyof FormData]) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const nextStep = () => {
    if (currentStep < FORM_STEPS.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleComplete = () => {
    if (onComplete) {
      onComplete(formData);
    }
  };

  const handleTierSelect = (tier: TierType) => {
    updateFormData('tier', tier);
    if (onTierChange) {
      onTierChange(tier);
    }
  };

  const renderStepContent = () => {
    switch (FORM_STEPS[currentStep]?.id) {
      case 'personal':
        return (
          <div className="space-y-6">
            {/* Authentication section */}
            {!user && (
              <div className="bg-card/50 rounded-xl p-4 border border-border mb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <UserIcon className="w-5 h-5 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Already have an account?</span>
                  </div>
                  <Button variant="outline" size="sm" className="gap-2" onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                    <LogIn className="w-4 h-4" />
                    Sign In
                  </Button>
                </div>
              </div>
            )}
            
            <div className="space-y-4">
              <label htmlFor="name" className="block text-sm font-medium mb-2">Full Name</label>
              <input
                id="name"
                type="text"
                value={formData.name}
                onChange={(e) => updateFormData('name', e.target.value)}
                className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                style={{
                  borderColor: currentTierStyling.color + '40',
                  '--tw-ring-color': currentTierStyling.color
                } as CSSProperties}
                placeholder="Enter your full name"
              />
            </div>
            
            <div className="space-y-4">
              <label htmlFor="email" className="block text-sm font-medium mb-2">Email</label>
              <input
                id="email"
                type="email"
                value={formData.email}
                onChange={(e) => updateFormData('email', e.target.value)}
                className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                style={{
                  borderColor: currentTierStyling.color + '40',
                  '--tw-ring-color': currentTierStyling.color
                } as CSSProperties}
                placeholder="your.email@example.com"
              />
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <label htmlFor="age" className="block text-sm font-medium mb-2">Age</label>
                <input
                  id="age"
                  type="number"
                  value={formData.age}
                  onChange={(e) => updateFormData('age', parseInt(e.target.value) || '')}
                  className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                  style={{
                    borderColor: currentTierStyling.color + '40',
                    '--tw-ring-color': currentTierStyling.color
                  } as CSSProperties}
                  placeholder="25"
                  min="13"
                  max="100"
                />
              </div>
              
              <div className="space-y-4">
                <label htmlFor="gender" className="block text-sm font-medium mb-2">Gender</label>
                <select
                  id="gender"
                  value={formData.gender}
                  onChange={(e) => updateFormData('gender', e.target.value)}
                  className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                  style={{
                    borderColor: currentTierStyling.color + '40',
                    '--tw-ring-color': currentTierStyling.color
                  } as CSSProperties}
                >
                  <option value="">Select gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>
          </div>
        );

      case 'goals':
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <label htmlFor="primaryGoal" className="block text-sm font-medium mb-2">Primary Goal *</label>
              <select
                id="primaryGoal"
                value={formData.primaryGoal}
                onChange={(e) => updateFormData('primaryGoal', e.target.value)}
                className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                style={{
                  borderColor: currentTierStyling.color + '40',
                  '--tw-ring-color': currentTierStyling.color
                } as CSSProperties}
                required
              >
                <option value="">Select your primary goal</option>
                <option value="muscle_gain">Build Muscle</option>
                <option value="fat_loss">Lose Fat</option>
                <option value="strength">Get Stronger</option>
                <option value="performance">Athletic Performance</option>
                <option value="longevity">Health & Longevity</option>
              </select>
            </div>
            
            <div className="space-y-4">
              <label htmlFor="experience" className="block text-sm font-medium mb-2">Training Experience</label>
              <select
                id="experience"
                value={formData.experience}
                onChange={(e) => updateFormData('experience', e.target.value)}
                className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                style={{
                  borderColor: currentTierStyling.color + '40',
                  '--tw-ring-color': currentTierStyling.color
                } as CSSProperties}
              >
                <option value="">Select experience level</option>
                <option value="beginner">Beginner (0-1 years)</option>
                <option value="intermediate">Intermediate (1-3 years)</option>
                <option value="advanced">Advanced (3+ years)</option>
              </select>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <label htmlFor="currentWeight" className="block text-sm font-medium mb-2">Current Weight (lbs)</label>
                <input
                  id="currentWeight"
                  type="number"
                  value={formData.currentWeight}
                  onChange={(e) => updateFormData('currentWeight', parseInt(e.target.value) || '')}
                  className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                  style={{
                    borderColor: currentTierStyling.color + '40',
                    '--tw-ring-color': currentTierStyling.color
                  } as CSSProperties}
                  placeholder="180"
                />
              </div>
              
              <div className="space-y-4">
                <label className="block text-sm font-medium mb-2">Target Weight (lbs)</label>
                <input
                  type="number"
                  value={formData.targetWeight}
                  onChange={(e) => updateFormData('targetWeight', parseInt(e.target.value) || '')}
                  className="w-full px-4 py-3 rounded-lg border-2 bg-background/50 transition-all duration-200 focus:outline-none focus:border-2"
                  style={{
                    borderColor: currentTierStyling.color + '40',
                    '--tw-ring-color': currentTierStyling.color
                  } as CSSProperties}
                  placeholder="175"
                />
              </div>
            </div>
          </div>
        );

      case 'lifestyle':
        return (
          <div className="space-y-8">
            <div className="space-y-4">
              <label className="block text-sm font-medium mb-3">
                Training Days per Week: <span className="font-bold" style={{ color: currentTierStyling.color }}>{formData.trainingDays || 3}</span>
              </label>
              <div className="relative">
                <input
                  type="range"
                  min="1"
                  max="7"
                  value={formData.trainingDays || 3}
                  onChange={(e) => updateFormData('trainingDays', parseInt(e.target.value))}
                  className="w-full h-3 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, ${currentTierStyling.color} 0%, ${currentTierStyling.color} ${((formData.trainingDays as number || 3) / 7) * 100}%, hsl(var(--muted)) ${((formData.trainingDays as number || 3) / 7) * 100}%, hsl(var(--muted)) 100%)`
                  }}
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>1 day</span>
                  <span>7 days</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <label className="block text-sm font-medium mb-3">
                Sleep Hours per Night: <span className="font-bold" style={{ color: currentTierStyling.color }}>{formData.sleepHours || 7}</span>
              </label>
              <div className="relative">
                <input
                  type="range"
                  min="4"
                  max="12"
                  value={formData.sleepHours || 7}
                  onChange={(e) => updateFormData('sleepHours', parseInt(e.target.value))}
                  className="w-full h-3 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, ${currentTierStyling.color} 0%, ${currentTierStyling.color} ${((formData.sleepHours as number || 7) / 8) * 100}%, hsl(var(--muted)) ${((formData.sleepHours as number || 7) / 8) * 100}%, hsl(var(--muted)) 100%)`
                  }}
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>4 hours</span>
                  <span>12 hours</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <label className="block text-sm font-medium mb-3">
                Stress Level: <span className="font-bold" style={{ color: currentTierStyling.color }}>{formData.stressLevel || 5}/10</span>
              </label>
              <div className="relative">
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={formData.stressLevel || 5}
                  onChange={(e) => updateFormData('stressLevel', parseInt(e.target.value))}
                  className="w-full h-3 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, ${currentTierStyling.color} 0%, ${currentTierStyling.color} ${((formData.stressLevel as number || 5) / 10) * 100}%, hsl(var(--muted)) ${((formData.stressLevel as number || 5) / 10) * 100}%, hsl(var(--muted)) 100%)`
                  }}
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>Low (1)</span>
                  <span>High (10)</span>
                </div>
              </div>
            </div>
          </div>
        );

      case 'tier':
        return (
          <div className="space-y-8">
            {/* Header */}
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">Choose Your Coaching Plan</h2>
              <p className="text-muted-foreground text-lg mb-6">Select the tier that best matches your goals</p>
              
              {/* Billing toggle (visual only for now) */}
              <div className="inline-flex bg-card border rounded-lg p-1 mb-8">
                <button className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium" onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                  Monthly
                </button>
                <button className="px-4 py-2 text-muted-foreground hover:text-foreground rounded-md text-sm font-medium" onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                  Quarterly
                </button>
                <button className="px-4 py-2 text-muted-foreground hover:text-foreground rounded-md text-sm font-medium" onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                  Semi-Annual
                </button>
                <button className="px-4 py-2 text-muted-foreground hover:text-foreground rounded-md text-sm font-medium" onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                  Annual
                </button>
              </div>
            </div>

            {/* Enhanced Tier Comparison Cards */}
            <div className="space-y-8">
              {/* Mobile/Tablet: Enhanced Stack View */}
              <div className="lg:hidden space-y-6">
                {(Object.keys(TIER_STYLING) as TierType[]).map((tier) => {
                  const tierConfig = TIER_STYLING[tier];
                  const isSelected = formData.tier === tier;
                  const Icon = tierConfig.icon;
                  
                  return (
                    <motion.div
                      key={tier}
                      className={`relative bg-card rounded-2xl border-2 overflow-hidden cursor-pointer transition-all duration-300 ${
                        isSelected ? 'ring-4 ring-opacity-20 shadow-2xl scale-105' : 'hover:shadow-lg hover:scale-[1.02]'
                      }`}
                      style={{
                        borderColor: tierConfig.color,
                        backgroundColor: isSelected ? `${tierConfig.color}08` : undefined
                      }}
                      onClick={() => handleTierSelect(tier)}
                      whileTap={{ scale: 0.98 }}
                    >
                      {/* Popular Badge */}
                      {tier === 'performance' && (
                        <div 
                          className="absolute -top-3 left-1/2 transform -translate-x-1/2 px-4 py-1 rounded-full text-white text-xs font-bold z-20"
                          style={{ backgroundColor: tierConfig.color }}
                        >
                          MOST POPULAR
                        </div>
                      )}
                      
                      {/* Header Section */}
                      <div className="p-6 pb-4">
                        <div className="flex items-start gap-4">
                          <div 
                            className="w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0"
                            style={{ backgroundColor: `${tierConfig.color}20` }}
                          >
                            <Icon className="w-8 h-8" style={{ color: tierConfig.color }} />
                          </div>
                          
                          <div className="flex-1">
                            <h3 className="text-xl font-bold mb-1" style={{ color: tierConfig.color }}>
                              {tierConfig.name}
                            </h3>
                            <p className="text-sm text-muted-foreground mb-3">
                              {tier === 'core' && 'Essential foundation for fitness beginners'}
                              {tier === 'adaptive' && 'AI-powered adaptive coaching system'}
                              {tier === 'performance' && 'Elite performance optimization program'}
                              {tier === 'longevity' && 'Comprehensive longevity & wellness protocol'}
                            </p>
                            
                            <div className="flex items-baseline gap-2">
                              <span className="text-3xl font-bold" style={{ color: tierConfig.color }}>
                                {tierConfig.price}
                              </span>
                              <span className="text-sm text-muted-foreground">/month</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Features Grid - Mobile Optimized */}
                      <div className="px-6 pb-6">
                        <div className="grid grid-cols-2 gap-y-3 gap-x-4 text-sm mb-6">
                          <div className="flex items-center gap-2">
                            <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                            <span>Custom workouts</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                            <span>Nutrition plans</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div 
                              className="w-5 h-5 rounded flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                              style={{ backgroundColor: tierConfig.color }}
                            >
                              {tier === 'core' ? '72' : tier === 'adaptive' ? '48' : tier === 'performance' ? '24' : '6'}
                            </div>
                            <span>Hour response time</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {tier === 'core' ? (
                              <X className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                            ) : (
                              <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                            )}
                            <span>Video coaching calls</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {tier === 'core' ? (
                              <X className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                            ) : (
                              <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                            )}
                            <span>AI coaching insights</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {tier === 'longevity' ? (
                              <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                            ) : (
                              <X className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                            )}
                            <span>Priority support</span>
                          </div>
                        </div>
                        
                        {/* CTA Button */}
                        <button
                          className={`w-full py-3 px-4 rounded-lg font-bold transition-all duration-200 ${
                            isSelected ? 'shadow-lg' : 'hover:shadow-md'
                          }`}
                          style={{
                            backgroundColor: isSelected ? tierConfig.color : `${tierConfig.color}20`,
                            color: isSelected ? 'white' : tierConfig.color,
                            border: `2px solid ${tierConfig.color}`
                          }}
                         onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                          {isSelected ? (
                            <div className="flex items-center justify-center gap-2">
                              <Check className="w-4 h-4" />
                              Selected
                            </div>
                          ) : (
                            `Choose ${tierConfig.name}`
                          )}
                        </button>
                      </div>
                    </motion.div>
                  );
                })}
              </div>

              {/* Desktop: Professional Comparison Table */}
              <div className="hidden lg:block">
                <div className="bg-gradient-to-b from-card/80 to-card/40 backdrop-blur-sm rounded-2xl border border-border/50 overflow-hidden shadow-xl">
                  <div className="grid grid-cols-5 gap-0">
                    {/* Feature Categories Column */}
                    <div className="bg-muted/10 p-6 border-r border-border/30">
                      <div className="h-32 flex items-end pb-4">
                        <div>
                          <h3 className="font-bold text-lg mb-2">Compare Plans</h3>
                          <p className="text-sm text-muted-foreground">Find the perfect fit for your goals</p>
                        </div>
                      </div>
                      
                      {/* Core Platform Section */}
                      <div className="mt-8 space-y-4">
                        <h4 className="font-bold text-sm text-foreground border-b border-border/30 pb-3 flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-green-500"></div>
                          Core Platform & Program
                        </h4>
                        <div className="space-y-4 text-sm text-muted-foreground">
                          <div className="py-2 border-b border-border/10">Custom workout plans</div>
                          <div className="py-2 border-b border-border/10">Exercise video library</div>
                          <div className="py-2 border-b border-border/10">Progress tracking dashboard</div>
                          <div className="py-2 border-b border-border/10">Nutrition meal planning</div>
                          <div className="py-2">Mobile app access</div>
                        </div>
                      </div>

                      {/* Coach Access Section */}
                      <div className="mt-10 space-y-4">
                        <h4 className="font-bold text-sm text-foreground border-b border-border/30 pb-3 flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                          Coach Access & Support
                        </h4>
                        <div className="space-y-4 text-sm text-muted-foreground">
                          <div className="py-2 border-b border-border/10">Coach response time</div>
                          <div className="py-2 border-b border-border/10">1-on-1 video coaching</div>
                          <div className="py-2 border-b border-border/10">Form check analysis</div>
                          <div className="py-2 border-b border-border/10">Real-time adjustments</div>
                          <div className="py-2">Priority support access</div>
                        </div>
                      </div>

                      {/* Advanced Features Section */}
                      <div className="mt-10 space-y-4">
                        <h4 className="font-bold text-sm text-foreground border-b border-border/30 pb-3 flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                          Advanced Features
                        </h4>
                        <div className="space-y-4 text-sm text-muted-foreground">
                          <div className="py-2 border-b border-border/10">AI coaching insights</div>
                          <div className="py-2 border-b border-border/10">Biometric integration</div>
                          <div className="py-2 border-b border-border/10">Recovery optimization</div>
                          <div className="py-2 border-b border-border/10">Competition preparation</div>
                          <div className="py-2">Longevity protocols</div>
                        </div>
                      </div>
                    </div>

                    {/* Tier Columns */}
                    {(Object.keys(TIER_STYLING) as TierType[]).map((tier) => {
                      const tierConfig = TIER_STYLING[tier];
                      const isSelected = formData.tier === tier;
                      const Icon = tierConfig.icon;
                      
                      return (
                        <div 
                          key={tier}
                          className={`relative p-6 transition-all duration-300 cursor-pointer hover:bg-muted/5 border-r border-border/20 last:border-r-0 ${
                            isSelected ? 'ring-2 ring-inset shadow-lg' : ''
                          }`}
                          style={{
                            backgroundColor: isSelected ? `${tierConfig.color}05` : undefined
                          }}
                          onClick={() => handleTierSelect(tier)}
                        >
                          {/* Popular Badge */}
                          {tier === 'performance' && (
                            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 z-10">
                              <div 
                                className="px-4 py-1.5 rounded-full text-xs font-bold text-white shadow-lg"
                                style={{ backgroundColor: tierConfig.color }}
                              >
                                MOST POPULAR
                              </div>
                            </div>
                          )}

                          {/* Tier Header */}
                          <div className="text-center mb-8">
                            <div 
                              className="w-20 h-20 rounded-2xl mx-auto mb-4 flex items-center justify-center shadow-lg"
                              style={{ backgroundColor: `${tierConfig.color}20` }}
                            >
                              <Icon className="w-10 h-10" style={{ color: tierConfig.color }} />
                            </div>
                            
                            <h3 className="text-2xl font-bold mb-2" style={{ color: tierConfig.color }}>
                              {tierConfig.name}
                            </h3>
                            <p className="text-sm text-muted-foreground mb-4 h-10 flex items-center justify-center">
                              {tier === 'core' && 'Essential foundation'}
                              {tier === 'adaptive' && 'AI-powered coaching'}
                              {tier === 'performance' && 'Elite optimization'}
                              {tier === 'longevity' && 'Complete wellness'}
                            </p>
                            
                            <div className="space-y-1">
                              <div className="text-4xl font-bold" style={{ color: tierConfig.color }}>
                                {tierConfig.price}
                              </div>
                              <div className="text-sm text-muted-foreground">/month</div>
                            </div>
                          </div>

                          {/* Core Platform Features */}
                          <div className="space-y-4 text-sm">
                            <div className="flex items-center justify-center py-2">
                              <Check className="w-5 h-5 text-green-500" />
                            </div>
                            <div className="flex items-center justify-center py-2">
                              <div className="text-center">
                                <div className="text-xs font-bold" style={{ color: tierConfig.color }}>
                                  {tier === 'core' ? '100+' : tier === 'adaptive' ? '300+' : tier === 'performance' ? '500+' : '1000+'}
                                </div>
                                <div className="text-xs text-muted-foreground">exercises</div>
                              </div>
                            </div>
                            <div className="flex items-center justify-center py-2">
                              <div className="text-center">
                                <div className="text-xs font-bold" style={{ color: tierConfig.color }}>
                                  {tier === 'core' ? 'Basic' : tier === 'adaptive' ? 'Advanced' : 'Premium'}
                                </div>
                                <div className="text-xs text-muted-foreground">analytics</div>
                              </div>
                            </div>
                            <div className="flex items-center justify-center py-2">
                              <div className="text-center">
                                <div className="text-xs font-bold" style={{ color: tierConfig.color }}>
                                  {tier === 'core' ? 'Basic' : tier === 'adaptive' ? 'Custom' : 'Premium'}
                                </div>
                                <div className="text-xs text-muted-foreground">plans</div>
                              </div>
                            </div>
                            <div className="flex items-center justify-center py-2">
                              <Check className="w-5 h-5 text-green-500" />
                            </div>
                          </div>

                          {/* Coach Access Features */}
                          <div className="mt-10 space-y-4 text-sm">
                            <div className="flex items-center justify-center py-2">
                              <div 
                                className="px-3 py-1.5 rounded-lg text-xs font-bold text-white"
                                style={{ backgroundColor: tierConfig.color }}
                              >
                                {tier === 'core' ? '72 hours' : tier === 'adaptive' ? '48 hours' : tier === 'performance' ? '24 hours' : '6 hours'}
                              </div>
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'core' ? (
                                <X className="w-5 h-5 text-muted-foreground" />
                              ) : (
                                <div className="text-center">
                                  <div className="text-xs font-bold" style={{ color: tierConfig.color }}>
                                    {tier === 'adaptive' ? 'Monthly' : tier === 'performance' ? 'Bi-weekly' : 'Weekly'}
                                  </div>
                                  <div className="text-xs text-muted-foreground">sessions</div>
                                </div>
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'core' ? (
                                <div className="text-center">
                                  <div className="text-xs text-muted-foreground">Basic</div>
                                  <div className="text-xs text-muted-foreground">feedback</div>
                                </div>
                              ) : (
                                <Check className="w-5 h-5 text-green-500" />
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'core' ? (
                                <X className="w-5 h-5 text-muted-foreground" />
                              ) : (
                                <Check className="w-5 h-5 text-green-500" />
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'longevity' ? (
                                <Check className="w-5 h-5 text-green-500" />
                              ) : (
                                <X className="w-5 h-5 text-muted-foreground" />
                              )}
                            </div>
                          </div>

                          {/* Advanced Features */}
                          <div className="mt-10 space-y-4 text-sm">
                            <div className="flex items-center justify-center py-2">
                              {tier === 'core' ? (
                                <X className="w-5 h-5 text-muted-foreground" />
                              ) : (
                                <Check className="w-5 h-5 text-green-500" />
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'performance' || tier === 'longevity' ? (
                                <Check className="w-5 h-5 text-green-500" />
                              ) : (
                                <X className="w-5 h-5 text-muted-foreground" />
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'performance' || tier === 'longevity' ? (
                                <Check className="w-5 h-5 text-green-500" />
                              ) : (
                                <X className="w-5 h-5 text-muted-foreground" />
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'performance' ? (
                                <Check className="w-5 h-5 text-green-500" />
                              ) : (
                                <X className="w-5 h-5 text-muted-foreground" />
                              )}
                            </div>
                            <div className="flex items-center justify-center py-2">
                              {tier === 'longevity' ? (
                                <Check className="w-5 h-5 text-green-500" />
                              ) : (
                                <X className="w-5 h-5 text-muted-foreground" />
                              )}
                            </div>
                          </div>

                          {/* CTA Button */}
                          <div className="mt-8">
                            <button
                              className={`w-full py-4 px-4 rounded-lg font-bold transition-all duration-200 ${
                                isSelected 
                                  ? 'transform scale-105 shadow-xl' 
                                  : 'hover:scale-[1.02] hover:shadow-lg'
                              }`}
                              style={{
                                backgroundColor: isSelected ? tierConfig.color : `${tierConfig.color}20`,
                                color: isSelected ? 'white' : tierConfig.color,
                                border: `2px solid ${tierConfig.color}`
                              }}
                             onClick={() => console.log("MultiStepIntakeForm button clicked")}>
                              {isSelected ? (
                                <div className="flex items-center justify-center gap-2">
                                  <Check className="w-5 h-5" />
                                  <span>Selected</span>
                                </div>
                              ) : (
                                <span>Choose {tierConfig.name}</span>
                              )}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Footer */}
            <div className="text-center pt-6">
              <p className="text-sm text-muted-foreground mb-2">
                All plans include a 30-day money-back guarantee
              </p>
              <div className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                <Crown className="w-3 h-3" />
                Join 2,847+ athletes transforming their lives
              </div>
            </div>
          </div>
        );

      case 'complete':
        return (
          <div className="text-center space-y-6">
            <div 
              className="w-20 h-20 rounded-full flex items-center justify-center mx-auto shadow-xl"
              style={{ backgroundColor: currentTierStyling.color }}
            >
              <Check className="w-10 h-10 text-white" />
            </div>
            
            <h2 className="text-2xl font-bold">Profile Complete!</h2>
            <p className="text-muted-foreground max-w-md mx-auto">
              Your intake is complete. Click finish to start your transformation with the {currentTierStyling.name}.
            </p>
            
            <div className="bg-card/50 rounded-xl p-6 text-left border border-border">
              <h3 className="font-bold mb-4" style={{ color: currentTierStyling.color }}>Summary:</h3>
              <div className="space-y-2 text-sm">
                <p><strong>Name:</strong> {formData.name}</p>
                <p><strong>Goal:</strong> {formData.primaryGoal}</p>
                <p><strong>Experience:</strong> {formData.experience}</p>
                <p><strong>Training Days:</strong> {formData.trainingDays}/week</p>
                <p><strong>Selected Plan:</strong> <span style={{ color: currentTierStyling.color }}>{currentTierStyling.name}</span></p>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center">
            <p className="text-muted-foreground">Step not found</p>
          </div>
        );
    }
  };

  // Progress calculation
  const progressPercentage = ((currentStep + 1) / FORM_STEPS.length) * 100;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent 
        className="max-w-4xl max-h-[90vh] p-0 gap-0 border-2 overflow-hidden"
        style={{ borderColor: currentTierStyling.color }}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-50 p-2 hover:bg-accent rounded-full transition-colors"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex flex-col h-full">
          {/* Header with tier branding and progress */}
          <div 
            className="px-8 pt-8 pb-6 border-b"
            style={{ 
              background: `linear-gradient(135deg, ${currentTierStyling.color}15, ${currentTierStyling.color}05)`
            }}
          >
            {/* Tier header */}
            <div className="flex items-center gap-3 mb-6">
              <div 
                className="w-12 h-12 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: currentTierStyling.color + '20' }}
              >
                <currentTierStyling.icon className="w-6 h-6" style={{ color: currentTierStyling.color }} />
              </div>
              <div>
                <h1 className="text-xl font-bold" style={{ color: currentTierStyling.color }}>
                  {currentTierStyling.name} Setup
                </h1>
                <p className="text-sm text-muted-foreground">
                  {Math.round(progressPercentage)}% Complete
                </p>
              </div>
            </div>

            {/* Mobile-friendly progress bar */}
            <div className="space-y-4">
              {/* Desktop progress indicators */}
              <div className="hidden md:flex items-center justify-between">
                {FORM_STEPS.map((step, index) => (
                  <div key={step.id} className="flex items-center">
                    <div className="flex flex-col items-center">
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 border-2 ${
                          index <= currentStep
                            ? 'text-white border-transparent'
                            : 'bg-transparent text-muted-foreground border-muted'
                        }`}
                        style={{
                          backgroundColor: index <= currentStep ? currentTierStyling.color : undefined
                        }}
                      >
                        {index + 1}
                      </div>
                      <div className="text-xs mt-2 text-center max-w-[80px]">
                        <div className={index <= currentStep ? 'text-foreground font-medium' : 'text-muted-foreground'}>
                          {step.title}
                        </div>
                      </div>
                    </div>
                    {index < FORM_STEPS.length - 1 && (
                      <div
                        className={`w-16 h-1 mx-3 rounded-full transition-all duration-300 ${
                          index < currentStep ? '' : 'bg-muted'
                        }`}
                        style={{
                          backgroundColor: index < currentStep ? currentTierStyling.color : undefined
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>

              {/* Mobile progress bar and step info */}
              <div className="md:hidden">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="font-bold text-lg">{FORM_STEPS[currentStep]?.title}</h3>
                    <p className="text-sm text-muted-foreground">{FORM_STEPS[currentStep]?.description}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">
                      {currentStep + 1} of {FORM_STEPS.length}
                    </div>
                  </div>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-300"
                    style={{
                      backgroundColor: currentTierStyling.color,
                      width: `${progressPercentage}%`
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Form content */}
          <div className="flex-1 overflow-y-auto p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
                className="h-full"
              >
                {renderStepContent()}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation */}
          <div className="px-8 py-6 border-t bg-card/50">
            <div className="flex justify-between items-center">
              <Button
                variant="outline"
                onClick={prevStep}
                disabled={currentStep === 0}
                className="gap-2"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </Button>

              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>Step {currentStep + 1} of {FORM_STEPS.length}</span>
              </div>

              {currentStep === FORM_STEPS.length - 1 ? (
                <Button
                  onClick={handleComplete}
                  className="gap-2 min-w-[120px]"
                  style={{ backgroundColor: currentTierStyling.color }}
                >
                  Complete Setup
                  <Check className="w-4 h-4" />
                </Button>
              ) : (
                <Button
                  onClick={nextStep}
                  className="gap-2 min-w-[120px]"
                  style={{ backgroundColor: currentTierStyling.color }}
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default MultiStepIntakeForm;
