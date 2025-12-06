import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { TIER_CONFIGS } from '@/constants/tiers';
import { cn } from '@/lib/utils';
import { useAuth } from '@/contexts/AuthContext';

// Add Calendly type declaration
declare global {
  interface Window {
    Calendly?: {
      initPopupWidget: (options: { url: string }) => void;
    };
  }
}
import { Button } from "@/components/ui/unified-button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { Input } from "@/ui/atoms/Input";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/ui/molecules/Accordion";
import { LoadingButton } from "@/components/ui/loading-button";
import { AnimatedSection } from "@/components/ui/animated-section";
import { TierCard } from "@/components/ui/tier-card";
import { ElegantBackground } from "@/components/ui/elegant-background";
import { SectionBoundary } from "@/components/ui/section-boundary";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { InteractiveCard } from "@/components/ui/interactive-card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { useScrollAnimation, useStaggeredAnimation } from "@/hooks/useScrollAnimation";
import { RepzLogo } from "@/ui/organisms/RepzLogo";
import { AuthButton } from "@/components/auth/AuthButton";
import { AuthModal } from "@/components/auth/AuthModal";
import { Footer } from "@/components/shared/Footer";
import fitnessIcons from "@/assets/fitness-location-icons.png";

import { SubscriptionModal } from "@/components/SubscriptionModal";
import { ProcessFlow } from "@/components/ProcessFlow";
import SEOStructuredData from "@/components/SEOStructuredData";
import AccessibilityEnhancements from "@/components/AccessibilityEnhancements";
import { useAnalyticsContext } from "@/components/AnalyticsProvider";
import { GymTierCard } from "@/components/ui/unified-pricing-card";
import { EnhancedTierDisplay } from "@/components/pricing/EnhancedTierDisplay";
import { useABTest, ABTestContent } from "@/lib/abTesting";
import { 
  Trophy, 
  Users, 
  Target, 
  Brain, 
  Zap, 
  Star, 
  Check, 
  CheckCircle,
  ArrowRight,
  Calendar,
  MessageSquare,
  MessageCircle,
  Activity,
  Award,
  Beaker,
  Heart,
  Clock,
  Phone,
  Mail,
  MapPin,
  Instagram,
  ExternalLink,
  Send,
  User,
  Search,
  Filter,
  X,
  FileText,
  Sheet,
  BarChart3,
  Video,
  UserCheck,
  Settings,
  Crosshair,
  Crown,
  Menu,
  Building,
  Home,
  CreditCard,
  ChevronDown,
  Shield,
  AlertTriangle,
  BookOpen,
  TrendingUp,
  Dumbbell,
  Utensils,
  LayoutDashboard,
  Camera,
  Watch,
  Bot,
  Pill,
  TestTube,
  Dna,
  Rocket,
  Monitor
} from 'lucide-react';
import { ProfessionalCertifications } from '@/components/ui/professional-certifications';
import { PricingToggle, BillingPeriod } from '@/components/ui/pricing-toggle';

const RepzHome = () => {
  const navigate = useNavigate();
  const { user, profile, signOut } = useAuth();
  const [selectedTier, setSelectedTier] = useState('');
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [subscriptionModalOpen, setSubscriptionModalOpen] = useState(false);
  const [faqSearch, setFaqSearch] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  
  // Get user tier for proper booking access
  const userTier = profile?.subscription_tier || 'core';
  const [billingPeriod, setBillingPeriod] = useState<BillingPeriod>('annual'); // Default to annual for psychology
  
  // Analytics and A/B Testing
  const analytics = useAnalyticsContext();
  const heroTest = useABTest('hero_headline');
  const ctaTest = useABTest('cta_button_text');
  const pricingTest = useABTest('pricing_presentation');
  const socialProofTest = useABTest('social_proof_placement');

  // Auth functions
  const handleAuthClick = (mode: 'login' | 'signup') => {
    setAuthMode(mode);
    setShowAuthModal(true);
  };

  const handleCloseAuthModal = () => {
    setShowAuthModal(false);
  };

  // Import pricing tiers from centralized config
  const pricingTiers = TIER_CONFIGS.map(config => ({
    name: config.displayName,
    price: config.price,
    period: config.period,
    description: config.description,
    features: config.features,
    cta: config.cta,
    popular: config.popular,
    variant: config.variant
  }));

  // Testimonials
  const testimonials = [
    {
      name: "Sarah Chen",
      role: "Professional Athlete",
      content: "REPZ transformed my training approach completely. The science-based methods helped me achieve my best season yet.",
      rating: 5,
      image: "/api/placeholder/60/60"
    },
    {
      name: "Marcus Johnson", 
      role: "Fitness Enthusiast",
      content: "The performance suite is incredible. My biomarker improvements and strength gains exceeded all expectations.",
      rating: 5,
      image: "/api/placeholder/60/60"
    },
    {
      name: "Dr. Lisa Rodriguez",
      role: "Medical Professional",
      content: "Finally, a coach who understands the science. The longevity approach has transformed my health metrics.",
      rating: 5,
      image: "/api/placeholder/60/60"
    }
  ];

  // Comprehensive FAQ Data
  const faqData = [
    {
      category: "programs",
      question: "How is this different from other online trainers?",
      answer: "Unlike generic programs, every plan is custom-designed by a PhD engineer with 18+ years of training experience. We use data and biomarkers, not guesswork. Our approach combines academic rigor with practical application for optimal results."
    },
    {
      category: "programs", 
      question: "Do you work with beginners?",
      answer: "Absolutely. Our Core tier is perfect for beginners, and we scale complexity based on your experience level. We meet you where you are and progressively advance your training as you develop."
    },
    {
      category: "programs",
      question: "What's included in the nutrition plans?",
      answer: "Complete meal plans, macro targets, supplement recommendations, and ongoing adjustments based on your progress and preferences. We also provide shopping lists and meal prep guidance."
    },
    {
      category: "programs",
      question: "Is the Bay Area in-person training required?",
      answer: "No, in-person training is optional and only available for Bay Area clients in the Longevity Concierge tier. All coaching tiers work completely remotely with full effectiveness."
    },
    {
      category: "results",
      question: "How quickly will I see results?",
      answer: "Most clients see measurable changes within 4-6 weeks, with significant transformations by 12 weeks. Results depend on consistency and starting point, but our data-driven approach accelerates progress."
    },
    {
      category: "results",
      question: "What if I don't see results?",
      answer: "We track multiple metrics beyond just weight - strength, body composition, energy levels, and biomarkers. If you're not progressing, we immediately adjust your protocol. Our 100% success rate speaks to our commitment to your results."
    },
    {
      category: "results",
      question: "Can you help with specific goals like powerlifting or bodybuilding?",
      answer: "Yes, our programs are fully customized to your specific goals. Whether it's powerlifting, bodybuilding, endurance sports, or general fitness, we design protocols that align with your objectives."
    },
    {
      category: "pricing",
      question: "What if I can't commit to the full program length?",
      answer: "There are no long-term contracts. You can pause or cancel anytime with 30 days notice. We focus on sustainable habits, not short-term fixes, so you retain the knowledge and systems we build together."
    },
    {
      category: "pricing",
      question: "Can I upgrade or downgrade between tiers?",
      answer: "Yes, you can change your coaching tier at any time with 30 days notice. We'll help you find the right level of support for your current goals and situation."
    },
    {
      category: "pricing",
      question: "Are there any hidden fees?",
      answer: "No hidden fees. The monthly price includes everything listed in your tier. The only additional costs might be optional supplements or lab work, which we'll discuss with you first."
    },
    {
      category: "programs",
      question: "What's the difference between PDF and Google Sheet delivery?",
      answer: "Core clients receive a PDF plan via email with text support (72hr response). Adaptive+ clients get access to a shared Google Sheet that your coach continuously updates based on your progress, feedback, and changing needs."
    },
    {
      category: "pricing",
      question: "Do you offer payment plans?",
      answer: "We offer monthly billing with no long-term commitments. For clients wanting to pay quarterly or annually, we provide discounts. Contact us to discuss payment options."
    },
    {
      category: "support",
      question: "How often will we communicate?",
      answer: "Communication frequency depends on your tier: Core (72-hour response), Adaptive (48-hour response), Performance (24-hour response + bi-weekly calls), Concierge (12-hour response + weekly calls)."
    },
    {
      category: "support",
      question: "What if I have questions outside business hours?",
      answer: "You can always send messages anytime. Response times vary by tier, but urgent matters are always prioritized. Higher tiers get faster response times and some include weekend support."
    },
    {
      category: "support",
      question: "Do you provide form checks for exercises?",
      answer: "Yes, Adaptive Engine and above include workout form reviews. You can send videos of your exercises and receive detailed feedback to ensure proper technique and injury prevention."
    },
    {
      category: "technical",
      question: "What technology/apps do I need?",
      answer: "We primarily use this platform for communication and plan delivery. You'll also need a way to track workouts (can be as simple as notes app) and potentially MyFitnessPal for nutrition tracking."
    },
    {
      category: "technical",
      question: "What if I miss a workout or meal?",
      answer: "Life happens, and our programs are designed with flexibility. We provide alternatives and modifications. Missing occasional workouts won't derail your progress - consistency over perfection is key."
    },
    {
      category: "technical",
      question: "Do I need expensive gym equipment?",
      answer: "Programs are designed around your available equipment. We can create effective protocols for home gyms, commercial gyms, or even bodyweight-only setups. Equipment doesn't determine results - programming does."
    }
  ];

  const faqCategories = [
    { id: "all", name: "All Questions", count: faqData.length },
    { id: "programs", name: "Programs", count: faqData.filter(q => q.category === "programs").length },
    { id: "results", name: "Results", count: faqData.filter(q => q.category === "results").length },
    { id: "pricing", name: "Pricing", count: faqData.filter(q => q.category === "pricing").length },
    { id: "support", name: "Support", count: faqData.filter(q => q.category === "support").length },
    { id: "technical", name: "Technical", count: faqData.filter(q => q.category === "technical").length }
  ];

  // Filter FAQ based on search and category
  const filteredFaq = faqData.filter(item => {
    const matchesSearch = item.question.toLowerCase().includes(faqSearch.toLowerCase()) ||
                         item.answer.toLowerCase().includes(faqSearch.toLowerCase());
    const matchesCategory = selectedCategory === "all" || item.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const openCalendly = (link: string, source?: string) => {
    // Track consultation booking
    analytics.trackConversion.consultationBooking();
    analytics.trackFunnel.heroEngagement('consultation_booking', source);
    
    // Track A/B test conversion
    heroTest.trackConversion('consultation_booking');
    ctaTest.trackConversion('consultation_booking');
    
    window.open(link, '_blank');
  };

  const handlePricingClick = (tierName: string) => {
    // Track tier reservation start
    analytics.trackFunnel.reservationFormStart(tierName);
    analytics.trackInteraction.buttonClick(`Reserve ${tierName}`, 'pricing_section');
    
    // Track A/B test conversion
    pricingTest.trackConversion('tier_selection');
    
    setSelectedTier(tierName);
    setSubscriptionModalOpen(true);
  };

  const handleChangeTier = () => {
    setSubscriptionModalOpen(false);
    // Small delay to allow modal to close before scrolling
    setTimeout(() => {
      const element = document.getElementById('pricing');
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 300);
  };

  const getPriceForPeriod = (basePrice: number, period: BillingPeriod) => {
    switch (period) {
      case 'quarterly': return Math.round(basePrice * 0.95); // 5% discount
      case 'semi-annual': return Math.round(basePrice * 0.9); // 10% discount
      case 'annual': return Math.round(basePrice * 0.8); // 20% discount
      default: return basePrice;
    }
  };

  return (
    <div className="min-h-screen bg-primary-elegant">
      <SEOStructuredData page="home" />
      <AccessibilityEnhancements />
      
      {/* Responsive Navigation Header with Stats-Style Background */}
      <nav className="backdrop-blur-2xl bg-gradient-to-br from-black/95 via-gray-950/98 to-black/95 sticky top-0 z-50 animate-fade-in relative overflow-hidden border-b border-orange-500/20"
           role="navigation" aria-label="Main navigation"
           style={{
             backfaceVisibility: "hidden",
             transform: "translateZ(0)",
             willChange: "transform",
             WebkitFontSmoothing: "antialiased",
             textRendering: "optimizeLegibility"
           }}>
        {/* Enhanced animated background elements for header - Stats style */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Geometric patterns similar to stats section */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-500/10 to-transparent transform skew-y-2"></div>
          <div className="absolute inset-0 bg-gradient-to-l from-transparent via-blue-500/8 to-transparent transform -skew-y-1"></div>
          <div className="absolute inset-0 bg-gradient-to-br from-transparent via-purple-500/6 to-transparent transform skew-x-1"></div>
          
          {/* Subtle particle effects */}
          <div className="absolute top-1/2 left-1/4 w-1 h-1 bg-orange-500/30 rounded-full animate-ping" style={{animationDelay: '0.5s'}}></div>
          <div className="absolute top-1/2 right-1/3 w-1 h-1 bg-orange-500/40 rounded-full animate-ping" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/2 left-1/3 w-1 h-1 bg-blue-400/30 rounded-full animate-ping" style={{animationDelay: '3.5s'}}></div>
        </div>
        
        
        {/* Flowing SVG lines in header - Hero style */}
        <svg className="absolute inset-0 w-full h-full opacity-25" viewBox="0 0 1200 80">
          <path
            d="M0,40 Q300,20 600,40 T1200,40"
            stroke="url(#headerGradient)"
            strokeWidth="3"
            fill="none"
            className="animate-pulse"
          />
          <path
            d="M0,50 Q300,60 600,50 T1200,50"
            stroke="url(#headerGradient2)"
            strokeWidth="2"
            fill="none"
            className="animate-pulse"
            style={{animationDelay: '2s'}}
          />
          <defs>
            <linearGradient id="headerGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#F15B23" stopOpacity="0.6"/>
              <stop offset="50%" stopColor="#FB923C" stopOpacity="0.8"/>
              <stop offset="100%" stopColor="#F15B23" stopOpacity="0.4"/>
            </linearGradient>
            <linearGradient id="headerGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.5"/>
              <stop offset="50%" stopColor="#F15B23" stopOpacity="0.7"/>
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.3"/>
            </linearGradient>
          </defs>
        </svg>
        
        <div className="container mx-auto px-2 max-w-full py-3 relative">
          {/* Desktop Layout - CSS Grid with 3 zones */}
          <div className="hidden lg:grid grid-cols-[180px_1fr_320px] items-center gap-2">
            {/* Zone 1: Official REPZ Brand Logo - Moved right slightly */}
            <div className="flex items-center gap-3 justify-self-start ml-2">
              <RepzLogo size="xl" />
            </div>
            
            {/* Zone 2: Navigation Links - Center */}
            <div className="flex items-center justify-center">
              <div className="flex items-center space-x-6">
                  <button 
                   onClick={() => {
                     const element = document.getElementById('pricing');
                     element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                   }}
                   className="relative text-white font-bold transition-all duration-300 hover:scale-105 group"
                   aria-label="View monthly coaching plans"
                 >
                   <span className="relative text-white font-bold"
                         style={{filter: 'drop-shadow(0 0 15px hsl(var(--repz-primary) / 0.7)) drop-shadow(0 0 30px hsl(var(--repz-primary) / 0.5))'}}>
                     Monthly Coaching Plans
                   </span>
                   <div className="absolute inset-0 bg-gradient-to-r from-repz-orange/20 via-amber-500/30 to-repz-orange/20 blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                 </button>
                  <button 
                    onClick={() => {
                      const element = document.getElementById('in-person-training');
                      element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }}
                    className="relative text-white font-bold transition-all duration-300 hover:scale-105 group"
                    aria-label="View in-person training plans"
                  >
                    <span className="relative text-white font-bold"
                          style={{filter: 'drop-shadow(0 0 15px hsl(var(--repz-primary) / 0.7)) drop-shadow(0 0 30px hsl(var(--repz-primary) / 0.5))'}}>
                      In-Person Training Plans
                    </span>
                    <div className="absolute inset-0 bg-gradient-to-r from-repz-orange/20 via-amber-500/30 to-repz-orange/20 blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  </button>
                
                {/* Secondary Navigation */}
                <div className="flex items-center space-x-3 ml-6 pl-6 border-l border-repz-orange/30">
                   <button 
                     onClick={() => {
                       const element = document.getElementById('about');
                       element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                     }}
                      className="nav-elegant"
                     aria-label="About our coaching"
                   >
                     About
                   </button>
                   <span className="text-repz-orange/40">|</span>
                   <button 
                     onClick={() => {
                       const element = document.getElementById('faq');
                       element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                     }}
                      className="nav-elegant"
                     aria-label="Frequently asked questions"
                   >
                     FAQ
                   </button>
                   <span className="text-repz-orange/40">|</span>
                   <button 
                     onClick={() => {
                       const element = document.getElementById('contact');
                       element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                     }}
                     className="nav-elegant"
                     aria-label="Contact information"
                   >
                     Contact
                   </button>
                </div>
              </div>
            </div>
            
            {/* Zone 3: Auth Buttons - EXTREME RIGHT */}
            <div className="flex items-center space-x-2 justify-self-end -mr-2">
              {user ? (
                <>
                  <span className="text-repz-orange font-medium text-sm whitespace-nowrap">
                    Welcome, {user.email?.split('@')[0]}
                  </span>
                  <Button
                    variant="secondary"
                    onClick={() => {
                      if (user?.role === 'coach') {
                        navigate('/coach');
                      } else {
                        navigate('/dashboard');
                      }
                    }}
                     className="btn-elegant border border-repz-orange text-repz-orange hover:bg-repz-orange hover:text-white"
                  >
                    Dashboard
                  </Button>
                  <Button
                    onClick={async () => {
                      await signOut();
                      navigate("/");
                    }}
                    className="bg-repz-orange hover:bg-repz-orange/90 text-white px-4 py-2 rounded-lg font-bold transition-colors text-sm whitespace-nowrap"
                  >
                    Sign Out
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    variant="ghost"
                    onClick={() => navigate('/signup')}
                    className="border-repz-orange text-repz-orange hover:bg-repz-orange hover:text-white px-4 py-2 rounded-lg font-bold transition-colors text-sm whitespace-nowrap"
                  >
                    Sign Up
                  </Button>
                  
                  <Button
                    onClick={() => navigate('/login')}
                    className="bg-repz-orange hover:bg-repz-orange/90 text-white px-4 py-2 rounded-lg font-bold transition-colors text-sm whitespace-nowrap"
                  >
                    Login
                  </Button>
                  
                </>
              )}
            </div>
          </div>

          {/* Mobile/Tablet Layout */}
          <div className="lg:hidden flex items-center justify-between">
            {/* Brand Logo + Name */}
             <div className="flex items-center gap-3">
               <RepzLogo size="md" />
             </div>
            
            {/* Auth Buttons + Menu */}
            <div className="flex items-center space-x-2">
              {user && (
                <span className="text-repz-orange font-medium text-sm max-w-[100px] truncate">
                  {user.email?.split('@')[0]}
                </span>
              )}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-2 rounded-md text-repz-orange hover:text-repz-orange/80 hover:bg-repz-orange/10 transition-colors"
                aria-label="Toggle mobile menu"
              >
                {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
            </div>
          </div>

          {/* Mobile & Tablet Menu Dropdown */}
          {mobileMenuOpen && (
            <div className="lg:hidden mt-4 pb-4 border-t border-white/10">
              <div className="flex flex-col space-y-4 pt-4">
                {/* Mobile & Tablet Navigation Links */}
                <button 
                  onClick={() => {
                    const element = document.getElementById('pricing');
                    element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    setMobileMenuOpen(false);
                  }}
                   className="font-montserrat text-repz-orange hover:text-repz-orange/80 transition-colors font-bold text-lg px-4 py-3 rounded-md hover:bg-repz-orange/10 border-l-4 border-repz-orange text-left"
                  aria-label="View monthly coaching plans"
                >
                  Monthly Coaching Plans
                </button>
                <button 
                  onClick={() => {
                    const element = document.getElementById('in-person-training');
                    element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    setMobileMenuOpen(false);
                  }}
                  className="font-montserrat text-repz-orange hover:text-repz-orange/80 transition-colors font-bold text-lg px-4 py-3 rounded-md hover:bg-repz-orange/10 border-l-4 border-repz-orange text-left"
                  aria-label="View in-person training plans"
                >
                  In-Person Training Plans
                </button>
                
                {/* Mobile Secondary Navigation */}
                <div className="border-t border-white/20 pt-4 mt-4">
                  <button 
                    onClick={() => {
                      const element = document.getElementById('about');
                      element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                      setMobileMenuOpen(false);
                    }}
                     className="font-inter text-white/80 hover:text-repz-orange transition-colors font-medium text-base px-4 py-2 rounded-md hover:bg-repz-orange/5 block text-left w-full"
                    aria-label="About our coaching"
                  >
                     About
                  </button>
                  <button 
                    onClick={() => {
                      const element = document.getElementById('faq');
                      element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                      setMobileMenuOpen(false);
                    }}
                     className="font-inter text-white/80 hover:text-repz-orange transition-colors font-medium text-base px-4 py-2 rounded-md hover:bg-repz-orange/5 block text-left w-full"
                    aria-label="Frequently asked questions"
                  >
                    FAQ
                  </button>
                  <button 
                    onClick={() => {
                      const element = document.getElementById('contact');
                      element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                      setMobileMenuOpen(false);
                    }}
                    className="font-inter text-white/80 hover:text-repz-orange transition-colors font-medium text-base px-4 py-2 rounded-md hover:bg-repz-orange/5 block text-left w-full"
                    aria-label="Contact information"
                  >
                    Contact
                  </button>
                </div>

                {/* Mobile Auth Buttons */}
                <div className="flex flex-col space-y-3 pt-4 border-t border-white/10">
                  {user ? (
                    <>
                      <Button
                        variant="secondary"
                        onClick={() => {
                          if (user?.role === 'coach') {
                            navigate('/coach');
                          } else {
                            navigate('/dashboard');
                          }
                          setMobileMenuOpen(false);
                        }}
                        className="border-repz-orange text-repz-orange hover:bg-repz-orange hover:text-white w-full py-3 rounded-lg font-bold transition-colors"
                      >
                        Dashboard
                      </Button>
                      <Button
                        onClick={async () => {
                          await signOut();
                          navigate("/");
                          setMobileMenuOpen(false);
                        }}
                        className="bg-repz-orange hover:bg-repz-orange/90 text-white w-full py-3 rounded-lg font-bold transition-colors"
                      >
                        Sign Out
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button
                        variant="ghost"
                        onClick={() => {
                          handleAuthClick('signup');
                          setMobileMenuOpen(false);
                        }}
                        className="border-repz-orange text-repz-orange hover:bg-repz-orange hover:text-white w-full py-3 rounded-lg font-bold transition-colors"
                      >
                        Sign Up
                      </Button>
                      
                      <Button
                        onClick={() => {
                          handleAuthClick('login');
                          setMobileMenuOpen(false);
                        }}
                        className="bg-repz-orange hover:bg-repz-orange/90 text-white w-full py-3 rounded-lg font-bold transition-colors"
                      >
                        Login
                      </Button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Enhanced Hero Section with Mobile-Optimized Design */}
      {/* Hero Section Boundary Line */}
      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
      
      <section className="section-padding text-white relative overflow-hidden min-h-[90vh] flex items-center bg-gradient-to-br from-black via-gray-950 to-slate-900"
               role="banner" aria-labelledby="hero-heading"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Enhanced dark overlay with subtle orange glow */}
        <div className="absolute inset-0 bg-gradient-to-r from-black/80 via-gray-950/70 to-black/60"></div>
        
        {/* Beautiful animated background elements */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Large hero gradient orbs */}
          <div className="absolute top-24 right-32 w-96 h-96 bg-gradient-to-br from-repz-orange/25 via-amber-500/18 to-yellow-500/12 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-20 left-20 w-80 h-80 bg-gradient-to-br from-blue-500/20 via-cyan-500/15 to-teal-500/10 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/3 left-1/2 w-72 h-72 bg-gradient-to-br from-purple-500/12 via-pink-500/8 to-repz-orange/6 rounded-full blur-3xl animate-pulse" style={{animationDelay: '4s'}}></div>
          
          {/* Subtle geometric patterns - barely visible dots with themed colors */}
          <div className="absolute top-20 left-28 w-3 h-3 bg-repz-orange/8 rounded-full animate-pulse opacity-30" style={{animationDuration: '6s'}}></div>
          <div className="absolute bottom-28 right-20 w-2 h-2 bg-amber-400/6 rounded-full animate-ping opacity-20" style={{animationDelay: '1.5s', animationDuration: '4s'}}></div>
          <div className="absolute top-40 left-1/3 w-2 h-2 bg-repz-orange/5 rounded-full animate-pulse opacity-25" style={{animationDelay: '3s', animationDuration: '8s'}}></div>
          <div className="absolute bottom-1/3 right-1/3 w-1 h-1 bg-amber-500/4 rounded-full animate-ping opacity-15" style={{animationDelay: '4.5s', animationDuration: '6s'}}></div>
          
          {/* Hero flowing SVG lines */}
          <svg className="absolute inset-0 w-full h-full opacity-20" viewBox="0 0 1200 800">
            <path
              d="M0,400 Q300,200 600,400 T1200,400"
              stroke="url(#heroGradient)"
              strokeWidth="3"
              fill="none"
              className="animate-pulse"
            />
            <path
              d="M0,300 Q300,500 600,300 T1200,300"
              stroke="url(#heroGradient2)"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
              style={{animationDelay: '2s'}}
            />
            <defs>
              <linearGradient id="heroGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.4"/>
              </linearGradient>
              <linearGradient id="heroGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#f97316" stopOpacity="0.7"/>
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.3"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        {/* Elegant floating elements - subtle and colored to blend with background */}
        <div className="hidden md:block absolute top-20 left-20 w-40 h-40 bg-gradient-to-br from-repz-orange/3 to-amber-500/2 rounded-full animate-float backdrop-blur-sm z-0"></div>
        <div className="hidden md:block absolute bottom-32 right-32 w-32 h-32 bg-gradient-to-br from-repz-orange/2 to-red-400/1 rounded-full animate-pulse backdrop-blur-sm z-0" style={{animationDelay: '1s'}}></div>
        <div className="hidden lg:block absolute top-1/2 right-1/3 w-24 h-24 bg-gradient-to-br from-amber-400/2 to-repz-orange/1 rounded-full animate-bounce backdrop-blur-sm z-0" style={{animationDelay: '2s'}}></div>
        
        <style>{`
          @media (min-width: 768px) {
            .hero-bg-image {
              background-position: 85% 53% !important;
              background-size: 92% !important;
            }
          }
        `}</style>
        
        {/* Background image - moved more to the left and better blended */}
        <div 
          className="hero-bg-image absolute inset-0 bg-cover bg-center bg-no-repeat 
                     opacity-35 md:opacity-30 
                     scale-100 -translate-x-4 md:scale-85 md:translate-x-8 md:-translate-y-40"
          style={{
            backgroundImage: "url('/lovable-uploads/da69e7c6-8684-4cd9-9788-86b7274dafd9.png')",
            backgroundPosition: "center center",
            backgroundSize: "100%",
            maskImage: "radial-gradient(ellipse 80% 70% at 40% 50%, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.8) 30%, rgba(0,0,0,0.5) 60%, rgba(0,0,0,0.15) 85%, transparent 100%)",
            WebkitMaskImage: "radial-gradient(ellipse 80% 70% at 40% 50%, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.8) 30%, rgba(0,0,0,0.5) 60%, rgba(0,0,0,0.15) 85%, transparent 100%)",
            filter: "brightness(1.1) contrast(1.05) saturate(1.1)",
            imageRendering: "crisp-edges",
            WebkitFontSmoothing: "antialiased",
            backfaceVisibility: "hidden",
            transform: "translateZ(0)",
            willChange: "transform"
          }}
          // Add desktop styles via CSS custom properties
          data-desktop-bg-pos="90% 30%"
          data-desktop-bg-size="40%"
          role="img"
          aria-label="REPZ fitness training background"
        ></div>
        
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl relative z-10">
          <div className="max-w-4xl h-full flex flex-col justify-between min-h-[80vh]">
            
            {/* Top Section - Main Content (moved higher) */}
            <div className="flex-1 flex flex-col justify-start mt-1 md:-mt-12">
              {/* Badge */}
              <div className="mb-4 md:mb-6 animate-slide-up">
               <span className="inline-block bg-repz-orange/20 text-repz-orange border border-repz-orange/50 rounded-full 
                               px-4 py-2 text-sm md:px-8 md:py-4 md:text-xl 
                               font-bold animate-glow backdrop-blur-sm shadow-lg">
                  Train Smarter. Get Stronger.
                </span>
              </div>
              
              {/* Main heading */}
              <h1 id="hero-heading" className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-montserrat font-bold text-white mb-4 md:mb-6 leading-tight animate-fade-in-up" style={{animationDelay: '0.2s'}}>
                Performance Coaching for
                <span className="block text-repz-orange mt-1 md:mt-2" 
                      style={{filter: 'drop-shadow(0 0 15px hsl(var(--repz-orange) / 0.7)) drop-shadow(0 0 30px hsl(var(--repz-orange) / 0.5))'}}>
                  Elite Results
                </span>
              </h1>
              
              {/* Description */}
              <p className="text-lg sm:text-xl md:text-2xl lg:text-3xl text-white/90 leading-relaxed max-w-2xl animate-slide-up" style={{animationDelay: '0.4s'}}>
                Revolutionary training methods backed by science. Transform your body with personalized nutrition and recovery optimization.
              </p>
            </div>
            
              {/* Premium CTA Panel System */}
              <div className="absolute -bottom-1 left-4 md:-bottom-12 md:left-8 grid grid-cols-1 gap-3 animate-elegant-slide w-[17rem] md:w-80">
                {/* Primary CTA Panel - Mathematical Brand Harmony */}
                <div className="card-premium group cursor-pointer relative overflow-hidden glass-tier-card-adaptive rounded-xl p-6 shadow-elegant hover:shadow-2xl transition-all duration-500 animate-premium-glow"
                     onClick={() => {
                       analytics.trackInteraction.buttonClick("Monthly Coaching Plans", "hero_primary_cta");
                       const element = document.getElementById('pricing');
                       if (element) {
                         element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                         setTimeout(() => {
                           const cards = element.querySelectorAll('.pricing-card');
                           cards.forEach((card, index) => {
                             setTimeout(() => {
                               card.classList.add('animate-sophisticated-pulse');
                               setTimeout(() => card.classList.remove('animate-sophisticated-pulse'), 1000);
                             }, index * 100);
                           });
                         }, 800);
                       }
                     }}>
                    {/* Ultra-Dynamic Orange Visual Effects */}
                    <div className="absolute inset-0 bg-gradient-to-br from-repz-orange/15 via-amber-elegant/10 to-copper-warm/12 opacity-40"></div>
                    <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-repz-orange/8 to-gold-luxe/10 animate-gradient-shift"></div>
                    <div className="absolute inset-0 bg-gradient-to-bl from-rust-deep/6 via-transparent to-repz-orange/4 animate-gradient-shift" style={{animationDelay: '2s', animationDuration: '8s'}}></div>
                    
                    {/* Floating Orb System */}
                    <div className="absolute top-2 right-2 w-16 h-16 bg-repz-orange/15 rounded-full blur-lg animate-sophisticated-pulse"></div>
                    <div className="absolute bottom-2 left-2 w-12 h-12 bg-amber-elegant/12 rounded-full blur-md animate-elegant-float"></div>
                    <div className="absolute top-1/2 left-1/4 w-8 h-8 bg-gold-luxe/8 rounded-full blur-sm animate-elegant-float" style={{animationDelay: '1s'}}></div>
                    <div className="absolute bottom-1/3 right-1/3 w-6 h-6 bg-copper-warm/10 rounded-full blur-sm animate-sophisticated-pulse" style={{animationDelay: '3s'}}></div>
                    
                    {/* Geometric Pattern Overlay */}
                    <div className="absolute inset-0 opacity-5 bg-gradient-to-r from-transparent via-repz-orange to-transparent" style={{
                      backgroundImage: 'radial-gradient(circle at 25% 25%, hsl(var(--repz-orange)) 2px, transparent 2px), radial-gradient(circle at 75% 75%, hsl(var(--amber-elegant)) 1px, transparent 1px)',
                      backgroundSize: '20px 20px'
                    }}></div>
                   
                   <div className="text-center relative z-10">
                     <div className="flex items-center justify-center mb-2">
                       <Dumbbell className="w-5 h-5 text-white mr-2" />
                       <div className="text-white font-bold text-sm md:text-base lg:text-lg">
                        Monthly Coaching Plans
                       </div>
                     </div>
                     <div className="text-white/90 text-sm md:text-base">
                       Personalized programs & nutrition
                     </div>
                   </div>
                </div>
                
                {/* Secondary CTA Panel - Complement Mathematical Harmony */}
                <div className="card-interactive group cursor-pointer relative overflow-hidden glass-tier-card-core rounded-xl p-6 shadow-elegant hover:shadow-2xl transition-all duration-500 animate-premium-glow"
                     onClick={() => {
                       analytics.trackInteraction.buttonClick("Bay Area Training", "hero_secondary_cta");
                       const element = document.getElementById('in-person-training');
                       if (element) {
                         element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                         setTimeout(() => {
                           const cards = element.querySelectorAll('.training-location-card');
                           cards.forEach((card, index) => {
                             setTimeout(() => {
                               card.classList.add('animate-sophisticated-pulse');
                               setTimeout(() => card.classList.remove('animate-sophisticated-pulse'), 1000);
                             }, index * 150);
                           });
                         }, 800);
                       }
                     }}>
                    {/* Ultra-Dynamic Blue-Steel Visual Effects */}
                    <div className="absolute inset-0 bg-gradient-to-br from-tier-core/15 via-blue-steel/12 to-slate-sophisticated/10 opacity-40"></div>
                    <div className="absolute inset-0 bg-gradient-to-tl from-transparent via-tier-core/8 to-teal-elegant/8 animate-gradient-shift"></div>
                    <div className="absolute inset-0 bg-gradient-to-tr from-purple-luxe/4 via-transparent to-tier-core/6 animate-gradient-shift" style={{animationDelay: '1.5s', animationDuration: '10s'}}></div>
                    
                    {/* Floating Orb System */}
                    <div className="absolute top-2 right-2 w-16 h-16 bg-tier-core/15 rounded-full blur-lg animate-sophisticated-pulse"></div>
                    <div className="absolute bottom-2 left-2 w-12 h-12 bg-blue-steel/15 rounded-full blur-md animate-elegant-float"></div>
                    <div className="absolute top-2/3 right-1/4 w-10 h-10 bg-teal-elegant/10 rounded-full blur-md animate-elegant-float" style={{animationDelay: '2s'}}></div>
                    <div className="absolute bottom-1/4 left-1/3 w-7 h-7 bg-slate-sophisticated/12 rounded-full blur-sm animate-sophisticated-pulse" style={{animationDelay: '4s'}}></div>
                    
                    {/* Smooth Gradient Overlay Instead of Dots */}
                    <div className="absolute inset-0 opacity-8 bg-gradient-to-r from-tier-core/3 via-transparent to-blue-steel/4 animate-gradient-shift" style={{animationDelay: '3s', animationDuration: '12s'}}></div>
                   
                   <div className="text-center relative z-10">
                     <div className="flex items-center justify-center mb-2">
                       <MapPin className="w-5 h-5 text-white mr-2" />
                       <div className="text-white font-bold text-sm md:text-base lg:text-lg">
                         Bay Area Training
                       </div>
                     </div>
                     <div className="text-white/90 text-sm md:text-base">
                       In-person sessions & locations
                     </div>
                   </div>
                 </div>
               </div>
            
          </div>
        </div>
        
      </section>

      <SectionBoundary variant="premium" glowEffect={false} animated />

      {/* Stats Banner Section */}
      <section className="section-padding relative overflow-hidden bg-gradient-to-br from-black via-gray-950 to-slate-900"
               style={{ 
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Beautiful animated background */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Large gradient orbs with enhanced glow */}
          <div className="absolute top-16 left-20 w-64 h-64 bg-gradient-to-br from-repz-orange/30 via-amber-500/20 to-yellow-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-20 right-24 w-56 h-56 bg-gradient-to-br from-blue-500/25 via-cyan-500/15 to-teal-500/10 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-gradient-to-br from-purple-500/15 via-pink-500/10 to-repz-orange/5 rounded-full blur-3xl animate-spin" style={{animationDuration: '20s'}}></div>
          
          {/* Animated geometric patterns */}
          <div className="absolute top-12 right-20 w-20 h-20 border-2 border-repz-orange/30 rotate-45 animate-bounce" style={{animationDuration: '3s'}}></div>
          <div className="absolute bottom-16 left-16 w-16 h-16 border border-blue-400/40 rounded-full animate-ping" style={{animationDelay: '1s'}}></div>
          <div className="absolute top-1/3 right-1/4 w-8 h-8 bg-gradient-to-br from-repz-orange/20 to-transparent rotate-12 animate-pulse"></div>
          
          {/* Enhanced flowing SVG lines */}
          <svg className="absolute inset-0 w-full h-full opacity-20" viewBox="0 0 1000 400">
            <path
              d="M0,200 Q250,100 500,200 T1000,200"
              stroke="url(#statsGradient)"
              strokeWidth="3"
              fill="none"
              className="animate-pulse"
            />
            <path
              d="M0,150 Q250,250 500,150 T1000,150"
              stroke="url(#statsGradient2)"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
              style={{animationDelay: '1.5s'}}
            />
            <path
              d="M0,250 Q250,150 500,250 T1000,250"
              stroke="url(#statsGradient3)"
              strokeWidth="1.5"
              fill="none"
              className="animate-pulse"
              style={{animationDelay: '3s'}}
            />
            <defs>
              <linearGradient id="statsGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.4"/>
              </linearGradient>
              <linearGradient id="statsGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#f97316" stopOpacity="0.7"/>
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.3"/>
              </linearGradient>
              <linearGradient id="statsGradient3" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.4"/>
                <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.6"/>
                <stop offset="100%" stopColor="#f97316" stopOpacity="0.5"/>
              </linearGradient>
            </defs>
          </svg>
          
          {/* Sophisticated mesh pattern */}
          <div className="absolute inset-0 opacity-10">
            {/* Enhanced geometric patterns for elegant highlight */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-500/25 to-transparent transform skew-y-6"></div>
            <div className="absolute inset-0 bg-gradient-to-l from-transparent via-blue-500/20 to-transparent transform -skew-y-6"></div>
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-purple-500/15 to-transparent transform skew-x-3"></div>
            
            {/* Elegant highlight border */}
            <div className="absolute inset-0 border border-orange-500/20 rounded-2xl"></div>
            <div className="absolute -inset-2 border border-orange-500/10 rounded-3xl"></div>
          </div>
          
          {/* Enhanced particle effects for elegance */}
          <div className="absolute top-1/4 left-1/3 w-3 h-3 bg-orange-500/20 rounded-full animate-pulse shadow-lg shadow-orange-500/30" style={{animationDelay: '0.5s', animationDuration: '4s'}}></div>
          <div className="absolute bottom-1/3 right-1/4 w-2 h-2 bg-orange-500/30 rounded-full animate-ping shadow-lg shadow-orange-500/40" style={{animationDelay: '2.5s'}}></div>
          <div className="absolute top-2/3 left-1/5 w-2 h-2 bg-blue-400/25 rounded-full animate-pulse shadow-lg shadow-blue-400/30" style={{animationDelay: '4s', animationDuration: '6s'}}></div>
        </div>
        
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl relative z-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <AnimatedSection animation="fade-in" delay={0} className="text-center">
              <AnimatedCounter 
                end={18} 
                suffix="+" 
                className="text-2xl sm:text-3xl md:text-4xl font-montserrat font-black text-white mb-2 drop-shadow-2xl"
              />
              <p className="text-gray-300 font-medium text-sm sm:text-base drop-shadow-lg">Years Experience</p>
            </AnimatedSection>
            <AnimatedSection animation="fade-in" delay={200} className="text-center">
              <AnimatedCounter 
                end={200} 
                suffix="+" 
                className="text-2xl sm:text-3xl md:text-4xl font-montserrat font-black text-white mb-2 drop-shadow-2xl"
              />
              <p className="text-gray-300 font-medium text-sm sm:text-base drop-shadow-lg">Clients Coached</p>
            </AnimatedSection>
            <AnimatedSection animation="fade-in" delay={400} className="text-center">
              <div className="text-xl sm:text-2xl md:text-4xl font-montserrat font-black text-white mb-2 drop-shadow-2xl">ISSA/NASM</div>
              <p className="text-gray-300 font-medium text-sm sm:text-base drop-shadow-lg">Certified</p>
            </AnimatedSection>
            <AnimatedSection animation="fade-in" delay={600} className="text-center">
              <div className="text-xl sm:text-2xl md:text-4xl font-montserrat font-black text-white mb-2 drop-shadow-2xl">Biohacking</div>
              <p className="text-gray-300 font-medium text-sm sm:text-base drop-shadow-lg">Expert</p>
            </AnimatedSection>
          </div>
        </div>
      </section>

      <SectionBoundary variant="top" />

      {/* Monthly Coaching Programs Section */}
      <section id="pricing" className="py-20 bg-gradient-to-br from-black via-gray-950 to-slate-900 relative overflow-hidden"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Enhanced animated background elements - Pricing section */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Large gradient orbs */}
          <div className="absolute top-32 right-40 w-96 h-96 bg-gradient-to-br from-repz-orange/20 via-amber-500/15 to-yellow-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-40 left-32 w-80 h-80 bg-gradient-to-br from-blue-500/18 via-cyan-500/12 to-teal-500/8 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/2 left-1/2 w-72 h-72 bg-gradient-to-br from-purple-500/15 via-pink-500/10 to-repz-orange/8 rounded-full blur-3xl animate-pulse" style={{animationDelay: '4s'}}></div>
          
          {/* Geometric patterns */}
          <div className="absolute top-24 left-32 w-40 h-40 border border-repz-orange/20 rotate-45 animate-pulse" style={{animationDuration: '6s'}}></div>
          <div className="absolute bottom-32 right-24 w-32 h-32 border border-blue-400/15 rounded-full animate-ping" style={{animationDelay: '1.5s', animationDuration: '4s'}}></div>
          
          {/* Flowing SVG lines */}
          <svg className="absolute inset-0 w-full h-full opacity-15" viewBox="0 0 1200 800">
            <path
              d="M0,400 Q300,200 600,400 T1200,400"
              stroke="url(#pricingGradient)"
              strokeWidth="3"
              fill="none"
              className="animate-pulse"
            />
            <path
              d="M0,300 Q300,500 600,300 T1200,300"
              stroke="url(#pricingGradient2)"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
              style={{animationDelay: '2s'}}
            />
            <defs>
              <linearGradient id="pricingGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.4"/>
              </linearGradient>
              <linearGradient id="pricingGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#f97316" stopOpacity="0.7"/>
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.3"/>
              </linearGradient>
            </defs>
          </svg>
          
          {/* Floating elements */}
          <div className="absolute top-20 left-20 w-48 h-48 bg-gradient-to-br from-repz-orange/8 to-amber-500/4 rounded-full animate-float backdrop-blur-sm"></div>
          <div className="absolute bottom-40 right-40 w-40 h-40 bg-gradient-to-br from-blue-500/6 to-cyan-400/3 rounded-full animate-pulse backdrop-blur-sm" style={{animationDelay: '1s'}}></div>
          <div className="absolute top-1/3 right-1/4 w-32 h-32 bg-gradient-to-br from-purple-400/5 to-pink-400/3 rounded-full animate-bounce backdrop-blur-sm" style={{animationDelay: '2s'}}></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 relative z-10">
          {/* Section Header - Matching other sections */}
          <div className="text-center mb-16">
            <div className="relative group mb-8 flex justify-center">
              <div className="relative bg-gradient-to-br from-orange-600/8 via-orange-500/5 to-orange-400/3 backdrop-blur-3xl border-2 border-orange-500/25 hover:border-orange-500/40 rounded-3xl px-8 py-6 shadow-xl w-full max-w-2xl hover:shadow-orange-500/15 transition-all duration-500 text-center bg-black/20">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="bg-gradient-to-br from-orange-600 to-orange-500 rounded-full p-4 shadow-lg">
                    <Calendar className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="pt-6">
                  <h2 className="text-3xl md:text-4xl lg:text-5xl font-montserrat font-bold mb-2 leading-tight text-center" 
                      style={{ 
                        color: '#F15B23',
                        textShadow: '0 2px 4px rgba(241, 91, 35, 0.25)'
                      }}>
                    Monthly Coaching Plans
                  </h2>
                  <p className="text-orange-200 font-semibold text-base md:text-lg">Choose your path to evidence-based transformation</p>
                </div>
              </div>
            </div>
          </div>

          {/* Billing Toggle */}
          <PricingToggle 
            billingPeriod={billingPeriod}
            onToggle={setBillingPeriod}
            className="mb-12"
          />


          {/* Use Enhanced Tier Display Component */}
          <EnhancedTierDisplay 
            billingPeriod={billingPeriod}
            onSelectTier={(tierId) => {
              analytics.trackInteraction.buttonClick(`Choose ${tierId}`, "monthly_coaching_tiers");
              setSelectedTier(tierId);
              setSubscriptionModalOpen(true);
            }}
            currentTier={profile?.subscription_tier}
            className="max-w-7xl mx-auto"
          />
        </div>
      </section>

      <SectionBoundary variant="top" />

      {/* In-Person Training Section - Sales Psychology Optimized */}
      <section className="py-20 bg-gradient-to-br from-black via-gray-950 to-slate-900 relative overflow-hidden"
               id="in-person-training" role="region" aria-labelledby="in-person-heading"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Enhanced animated background elements - In-Person Training section */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Large gradient orbs */}
          <div className="absolute top-40 left-32 w-96 h-96 bg-gradient-to-br from-repz-orange/22 via-amber-500/16 to-yellow-500/12 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-32 right-40 w-80 h-80 bg-gradient-to-br from-blue-500/18 via-cyan-500/14 to-teal-500/10 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/3 right-1/3 w-72 h-72 bg-gradient-to-br from-purple-500/16 via-pink-500/12 to-repz-orange/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '4s'}}></div>
          
          {/* Geometric patterns */}
          <div className="absolute top-32 right-28 w-44 h-44 border border-repz-orange/18 rotate-45 animate-pulse" style={{animationDuration: '6s'}}></div>
          <div className="absolute bottom-40 left-24 w-36 h-36 border border-blue-400/12 rounded-full animate-ping" style={{animationDelay: '1.5s', animationDuration: '4s'}}></div>
          
          {/* Flowing SVG lines */}
          <svg className="absolute inset-0 w-full h-full opacity-12" viewBox="0 0 1200 800">
            <path
              d="M0,350 Q300,150 600,350 T1200,350"
              stroke="url(#inPersonGradient)"
              strokeWidth="3"
              fill="none"
              className="animate-pulse"
            />
            <path
              d="M0,450 Q300,650 600,450 T1200,450"
              stroke="url(#inPersonGradient2)"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
              style={{animationDelay: '2s'}}
            />
            <defs>
              <linearGradient id="inPersonGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.7"/>
                <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.5"/>
              </linearGradient>
              <linearGradient id="inPersonGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#f97316" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4"/>
              </linearGradient>
            </defs>
          </svg>
          
          {/* Floating elements */}
          <div className="absolute top-24 right-16 w-52 h-52 bg-gradient-to-br from-repz-orange/10 to-amber-500/6 rounded-full animate-float backdrop-blur-sm"></div>
          <div className="absolute bottom-28 left-44 w-44 h-44 bg-gradient-to-br from-blue-500/8 to-cyan-400/5 rounded-full animate-pulse backdrop-blur-sm" style={{animationDelay: '1s'}}></div>
          <div className="absolute top-2/3 left-1/4 w-36 h-36 bg-gradient-to-br from-purple-400/7 to-pink-400/4 rounded-full animate-bounce backdrop-blur-sm" style={{animationDelay: '2s'}}></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 relative z-10">
          {/* Hero Header */}
          <div className="text-center mb-16">
            {/* Modern Card Header - Following established pattern */}
            <div className="relative group mb-8 flex justify-center">
              <div className="relative bg-gradient-to-br from-orange-600/8 via-orange-500/5 to-orange-400/3 backdrop-blur-3xl border-2 border-orange-500/25 hover:border-orange-500/40 rounded-3xl px-8 py-6 shadow-xl w-full max-w-2xl hover:shadow-orange-500/15 transition-all duration-500 text-center bg-black/20">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="bg-gradient-to-br from-orange-600 to-orange-500 rounded-full p-4 shadow-lg">
                    <Dumbbell className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="pt-6">
                  <h2 id="in-person-heading" className="text-3xl md:text-4xl lg:text-5xl font-montserrat font-bold mb-2 leading-tight text-center"
                      style={{ 
                        color: '#F26C00',
                        textShadow: '0 2px 4px rgba(242, 108, 0, 0.25)'
                      }}>
                    Premium In-Person Training
                  </h2>
                  <p className="text-orange-200 font-semibold text-base md:text-lg">Elite strength coaching across the Bay Area</p>
                </div>
              </div>
            </div>
          </div>

          {/* Gym Tier Cards - Unified Architecture with Fixed Dimensions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16 max-w-7xl mx-auto place-items-center">
            
            {/* Other Gym - External, Variable, Third-Party Dependent */}
            <GymTierCard
              title="Other Gym"
              subtitle="Train at your preferred facility"
              price="$65/session"
              priceSubtext="+ transportation/membership/guest fees"
              priceSubtextColor="text-red-400"
              tier="core"
              gymType="other"
              features={[
                {
                  text: "Gym membership required",
                  included: false,
                  iconColor: "text-red-400"
                },
                {
                  text: "Subject to gym hours",
                  included: false,
                  iconColor: "text-red-400"
                },
                {
                  text: "Your preferred location only",
                  included: true,
                  iconColor: "text-gray-400"
                }
              ]}
              buttonText="Book"
              buttonVariant="primary"
              onClick={() => {
                console.log('Other Gym Book button clicked');
                // Use correct Calendly URL for gym training
                const calendlyUrl = 'https://calendly.com/repz/gym-training';
                const utmParams = new URLSearchParams({
                  utm_source: 'repz-platform',
                  utm_medium: 'in-app',
                  utm_campaign: 'booking',
                  utm_content: userTier,
                  location: 'other-gym'
                });
                window.open(`${calendlyUrl}?${utmParams.toString()}`, '_blank');
              }}
            />

            {/* City Sports Club - Structured, Urban, High-Access */}
            <GymTierCard
              title="City Sports Club"
              subtitle="Downtown SF Location"
              price="$79/session"
              priceSubtext="Save $400+ annually vs private gyms"
              tier="adaptive" // Architecture reference, not visual
              gymType="city-sports"
              badge="coach-pick"
              features={[
                {
                  text: "Free gym access (no membership required)",
                  included: true,
                  iconColor: "text-green-400"
                },
                {
                  text: "Free validated parking included",
                  included: true,
                  iconColor: "text-green-400"
                },
                {
                  text: "Elite facility access",
                  included: true,
                  iconColor: "text-green-400"
                }
              ]}
              specialSection={
                <div className="bg-blue-900/30 border border-blue-400/40 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-blue-400">
                    <Zap className="w-4 h-4 animate-pulse" />
                    <span className="font-semibold text-xs text-blue-400 animate-pulse"
                          style={{
                            animation: 'shine-blue 3s ease-in-out infinite',
                            textShadow: '0 0 10px rgb(59, 130, 246), 0 0 20px rgb(59, 130, 246, 0.5)',
                          }}>
                      LIMITED SPOTS
                    </span>
                  </div>
                </div>
              }
              buttonText="Book"
              buttonVariant="primary"
              onClick={() => {
                console.log('City Sports Club Book button clicked');
                // Use correct Calendly URL for city sports club
                const calendlyUrl = 'https://calendly.com/repz/city-sports-club';
                const utmParams = new URLSearchParams({
                  utm_source: 'repz-platform',
                  utm_medium: 'in-app',
                  utm_campaign: 'booking',
                  utm_content: userTier,
                  location: 'city-sports-club'
                });
                window.open(`${calendlyUrl}?${utmParams.toString()}`, '_blank');
              }}
            />

            {/* Home Gym - Private, Flexible, Personalized */}
            <GymTierCard
              title="Home Gym"
              subtitle="Your home setup"
              price="$95/session"
              priceSubtext="Ultimate convenience"
              tier="performance" // Architecture reference, not visual
              gymType="home"
              features={[
                {
                  text: "Ultimate privacy",
                  included: true,
                  iconColor: "text-green-400"
                },
                {
                  text: "Your home setup",
                  included: true,
                  iconColor: "text-green-400"
                },
                {
                  text: "No commute needed",
                  included: true,
                  iconColor: "text-green-400"
                },
                {
                  text: "Flexible scheduling",
                  included: true,
                  iconColor: "text-green-400"
                }
              ]}
              buttonText="Book"
              buttonVariant="primary"
              onClick={() => {
                console.log('Home Gym Book button clicked');
                // Use correct Calendly URL for home training
                const calendlyUrl = 'https://calendly.com/repz/home-training';
                const utmParams = new URLSearchParams({
                  utm_source: 'repz-platform',
                  utm_medium: 'in-app',
                  utm_campaign: 'booking',
                  utm_content: userTier,
                  location: 'home-gym'
                });
                window.open(`${calendlyUrl}?${utmParams.toString()}`, '_blank');
              }}
            />
          </div>

          {/* What's Included in Every Session Panel - Elegant Version */}
          <div className="max-w-6xl mx-auto mb-16">
            <div className="relative bg-gradient-to-br from-orange-600/8 via-orange-500/5 to-orange-400/3 backdrop-blur-3xl border-2 border-orange-500/25 hover:border-orange-500/40 rounded-3xl p-10 shadow-xl hover:shadow-orange-500/15 transition-all duration-500 bg-black/20">
              {/* Enhanced floating icon */}
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <div className="bg-gradient-to-br from-orange-600 to-orange-500 rounded-full p-4 shadow-lg">
                  <CheckCircle className="w-10 h-10 text-white" />
                </div>
              </div>
              
              <div className="pt-8">
                <h3 className="text-3xl md:text-4xl font-montserrat font-bold text-center mb-3 text-white">
                  What's Included in Every Session
                </h3>
                <p className="text-gray-300 font-medium text-center text-lg mb-12">
                  Comprehensive training approach for optimal results
                </p>
                
                {/* 2-Column Layout (Desktop), 1-Column (Mobile) */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Technical Excellence */}
                  <div className="group bg-gradient-to-br from-orange-500/10 to-orange-600/5 border border-gray-600/30 rounded-2xl p-6 hover:border-orange-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center gap-4 mb-4">
                      <h4 className="font-bold text-orange-100 text-lg" style={{ color: '#F15B23' }}>Technical Excellence</h4>
                    </div>
                    <p className="text-orange-200 text-sm leading-relaxed">Real-time form analysis and performance optimization with expert coaching feedback</p>
                  </div>

                  {/* Performance Programming */}
                  <div className="group bg-gradient-to-br from-orange-500/10 to-orange-600/5 border border-gray-600/30 rounded-2xl p-6 hover:border-orange-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center gap-4 mb-4">
                      <h4 className="font-bold text-orange-100 text-lg" style={{ color: '#F15B23' }}>Performance Programming</h4>
                    </div>
                    <p className="text-orange-200 text-sm leading-relaxed">Periodized training systems designed for progressive advancement and peak performance</p>
                  </div>

                  {/* Recovery Systems */}
                  <div className="group bg-gradient-to-br from-orange-500/10 to-orange-600/5 border border-gray-600/30 rounded-2xl p-6 hover:border-orange-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center gap-4 mb-4">
                      <h4 className="font-bold text-orange-100 text-lg" style={{ color: '#F15B23' }}>Recovery Systems</h4>
                    </div>
                    <p className="text-orange-200 text-sm leading-relaxed">Mobility protocols and recovery optimization for injury prevention and performance enhancement</p>
                  </div>

                  {/* Competition Readiness */}
                  <div className="group bg-gradient-to-br from-orange-500/10 to-orange-600/5 border border-gray-600/30 rounded-2xl p-6 hover:border-orange-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center gap-4 mb-4">
                      <h4 className="font-bold text-orange-100 text-lg" style={{ color: '#F15B23' }}>Competition Readiness</h4>
                    </div>
                    <p className="text-orange-200 text-sm leading-relaxed">Sport-specific training and performance preparation for competitive excellence</p>
                  </div>

                  {/* Scientific Methodology */}
                  <div className="group bg-gradient-to-br from-orange-500/10 to-orange-600/5 border border-gray-600/30 rounded-2xl p-6 hover:border-orange-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center gap-4 mb-4">
                      <h4 className="font-bold text-orange-100 text-lg" style={{ color: '#F15B23' }}>Scientific Methodology</h4>
                    </div>
                    <p className="text-orange-200 text-sm leading-relaxed">Evidence-backed protocols and research-driven approach to maximize results</p>
                  </div>

                  {/* Immediate Impact */}
                  <div className="group bg-gradient-to-br from-orange-500/10 to-orange-600/5 border border-gray-600/30 rounded-2xl p-6 hover:border-orange-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl">
                    <div className="flex items-center gap-4 mb-4">
                      <h4 className="font-bold text-orange-100 text-lg" style={{ color: '#F15B23' }}>Immediate Impact</h4>
                    </div>
                    <p className="text-orange-200 text-sm leading-relaxed">Measurable results and noticeable improvements from your very first session</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </section>

      <SectionBoundary variant="top" />

      {/* About Me Section - "Where Academic Precision Meets Real-World Results" */}
      <section className="py-20 relative overflow-hidden bg-gradient-to-br from-black via-gray-950 to-slate-900"
               id="about" role="region" aria-labelledby="about-heading"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Beautiful animated background similar to stats */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Elegant gradient orbs */}
          <div className="absolute top-20 right-32 w-72 h-72 bg-gradient-to-br from-amber-500/20 via-amber-400/15 to-yellow-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-24 left-20 w-64 h-64 bg-gradient-to-br from-blue-500/15 via-cyan-500/10 to-teal-500/8 rounded-full blur-2xl animate-pulse" style={{animationDelay: '3s'}}></div>
          
          {/* Sophisticated geometric patterns */}
          <div className="absolute top-16 left-24 w-24 h-24 border border-amber-500/20 rotate-12 animate-pulse" style={{animationDuration: '4s'}}></div>
          <div className="absolute bottom-20 right-16 w-16 h-16 bg-gradient-to-br from-amber-500/10 to-transparent rounded-full animate-ping" style={{animationDelay: '2s'}}></div>
          
          {/* Elegant flowing SVG */}
          <svg className="absolute inset-0 w-full h-full opacity-15" viewBox="0 0 1200 600">
            <path
              d="M0,300 Q300,150 600,300 T1200,300"
              stroke="url(#aboutGradient)"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
            />
            <defs>
              <linearGradient id="aboutGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#F29C00" stopOpacity="0.4"/>
                <stop offset="50%" stopColor="#F2BC00" stopOpacity="0.6"/>
                <stop offset="100%" stopColor="#F29C00" stopOpacity="0.3"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl relative z-10">
          {/* Section Header - Centered above all content */}
          <div className="text-center mb-16">
            {/* Modern Card Header for About Section - Following established pattern */}
            <div className="relative group mb-8 flex justify-center">
              <div className="relative bg-gradient-to-br from-amber-600/8 via-amber-500/5 to-amber-400/3 backdrop-blur-3xl border-2 border-amber-500/25 hover:border-amber-500/40 rounded-3xl px-8 py-6 shadow-xl w-full max-w-2xl hover:shadow-amber-500/15 transition-all duration-500 text-center bg-black/20">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="bg-gradient-to-br from-amber-600 to-amber-500 rounded-full p-4 shadow-lg">
                    <Brain className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="pt-6">
                  <h2 id="about-heading" className="text-3xl md:text-4xl lg:text-5xl font-montserrat font-bold mb-2 leading-tight text-center"
                      style={{ 
                        color: '#F29C00',
                        textShadow: '0 2px 4px rgba(242, 156, 0, 0.25)'
                      }}>
                    About Your Coach
                  </h2>
                  <p className="text-amber-200 font-semibold text-base md:text-lg">Where Science Meets Real-World Results</p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-[400px_1fr] gap-16 items-start">
            {/* Left Panel - Profile Card */}
            <div className="space-y-8">
              {/* Profile Photo with Enhanced Glow and Brightness */}
              <div className="relative">
                <div className="relative w-80 h-80 mx-auto">
                  {/* Enhanced glowing border with physique highlighting */}
                  <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-amber-500/40 via-amber-400/30 to-yellow-500/20 blur-lg animate-pulse"></div>
                  <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-amber-500/60 via-amber-400/40 to-amber-500/30 blur-md animate-pulse" style={{animationDelay: '1s'}}></div>
                  
                  <div className="relative w-full h-full rounded-3xl overflow-hidden border-4 border-amber-500/90 bg-gradient-to-br from-amber-500/30 to-amber-400/40 p-2 transform rotate-1 shadow-2xl shadow-amber-500/30">
                    <img 
                      src="/lovable-uploads/7a3bccc9-ddd1-4a1d-8016-a53346d48693.png"
                      alt="Meshal Alawein, PhD - Lead Coach at REPZ"
                      className="w-full h-full object-cover rounded-2xl transform -rotate-1"
                      style={{
                        filter: "brightness(1.25) contrast(1.15) saturate(1.2) sepia(0.05) hue-rotate(5deg) drop-shadow(0 0 20px rgba(242, 156, 0, 0.6))",
                        boxShadow: "inset 0 0 0 2px rgba(242, 156, 0, 0.3)",
                        imageRendering: "crisp-edges",
                        WebkitFontSmoothing: "antialiased",
                        backfaceVisibility: "hidden",
                        transform: "translateZ(0)",
                        willChange: "transform"
                      }}
                      loading="lazy"
                    />
                  </div>
                  {/* PhD Badge */}
                  <div className="absolute top-2 right-2 bg-amber-500 text-white px-3 py-2 rounded-full font-bold text-sm border-2 border-amber-500/50 shadow-lg">
                    PhD
                  </div>
                </div>
              </div>
              
              {/* Name and Title */}
              <div className="text-center space-y-4">
                <h3 className="text-3xl font-bold text-white">Meshal Alawein</h3>
                <p className="text-lg font-semibold" style={{ color: '#F29C00' }}>PhD, UC Berkeley</p>
                
              </div>
            </div>

            {/* Right Panel - About Text */}
            <div className="space-y-8 text-white">
              {/* Bio Paragraphs */}
              <div className="space-y-6 text-lg leading-relaxed">
                <p>
                  Backed by a PhD from UC Berkeley and over 18 years in bodybuilding and science-based training, 
                  I bring academic excellence and real-world results to every coaching session.
                </p>
                
                <p>
                  I'm certified by{' '}
                  <span className="font-semibold" style={{ color: '#F29C00' }}>NASM and ISSA</span>{' '}
                  in personal training, nutrition, and corrective exercise, with additional credentials 
                  in performance enhancement and CPR/AED.
                </p>
                
                <p>
                  My upcoming book,{' '}
                  <em className="font-medium" style={{ color: '#F29C00' }}>"The Science of Enhancement"</em>, reflects 
                  my commitment to evidence-based coaching grounded in longevity, strength, and optimal health.
                </p>
              </div>
              
              {/* Pull Quote */}
              <div className="border-l-4 border-amber-500 bg-gray-800/50 p-6 rounded-r-xl backdrop-blur-sm">
                <p className="italic text-lg text-gray-200">
                  "Every protocol I design is built on peer-reviewed research, validated through real-world 
                  application, and continuously optimized based on measurable outcomes."
                </p>
              </div>

              {/* Professional Certifications */}
              <ProfessionalCertifications />
            </div>
          </div>

          {/* Science Story Hero Layout - Keep Existing Section */}
          <div className="max-w-7xl mx-auto mt-20">
            <div className="bg-gradient-to-br from-gray-800/50 via-slate-700/50 to-gray-700/50 rounded-2xl p-8 lg:p-12 backdrop-blur-sm border border-gray-600/30">
              <div className="grid lg:grid-cols-2 gap-12 items-center">
                
                {/* Left: Compelling Narrative */}
                <div className="space-y-6">
                  <div className="space-y-4">
                    <h3 className="text-3xl lg:text-4xl font-montserrat font-bold text-white leading-tight">
                      Where Science Meets 
                      <span className="block" style={{ color: '#F29C00' }}>
                        Real-World Results
                      </span>
                    </h3>
                    <div className="w-16 h-1 bg-amber-500 rounded-full"></div>
                  </div>
                  
                  <div className="space-y-4 text-gray-300 leading-relaxed">
                    <p className="text-lg">
                      <strong className="text-white">18 years ago</strong>, I was just an undergrad and an a aspiring bodybuilder. Today, I've transitioned into something different, fusing my PhD-level analytical rigor with two decades of training, nutrition, and supplementation experience to create something unique.
                    </p>
                    <p>
                      <strong style={{ color: '#F29C00' }}>This isn't about trends or quick fixes.</strong> Every protocol I design is built on peer-reviewed research, validated through real-world application, and continuously optimized based on measurable outcomes.
                    </p>
                    <p>
                      Whether you're pursuing elite performance, longevity, or transformative health improvements, my methodology delivers results that last — because it's engineered to work.
                    </p>
                  </div>

                  <div className="pt-4">
                    <div className="inline-flex items-center gap-2 bg-amber-500/20 border border-amber-500/30 rounded-full px-6 py-3">
                      <Award className="w-5 h-5" style={{ color: '#F29C00' }} />
                      <span className="text-sm font-medium text-white">Scientific Engineering Applied to Human Performance</span>
                    </div>
                  </div>
                </div>

                {/* Right: 3 Core Pillars */}
                <div className="space-y-6">
                  <h4 className="text-xl font-montserrat font-bold text-center text-white mb-8">
                    The Three Pillars of My Methodology
                  </h4>
                  
                  <div className="space-y-6">
                    {/* Research Pillar */}
                    <div className="group">
                      <div className="bg-gradient-to-r from-amber-500/10 to-amber-400/5 border border-amber-500/20 rounded-xl p-6 hover:border-amber-500/40 transition-all duration-300 hover:shadow-lg">
                        <div className="flex items-center gap-4 mb-3">
                          <div className="w-12 h-12 bg-gradient-to-br from-amber-500 to-amber-600 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                            <Beaker className="w-6 h-6 text-white" />
                          </div>
                          <h5 className="text-lg font-bold text-white">Evidence-Based</h5>
                        </div>
                        <p className="text-gray-300 text-sm leading-relaxed">
                          Every training and nutrition protocol is grounded in peer-reviewed research and validated through systematic testing.
                        </p>
                      </div>
                    </div>

                    {/* Precision Pillar */}
                    <div className="group">
                      <div className="bg-gradient-to-r from-amber-500/10 to-amber-400/5 border border-amber-500/20 rounded-xl p-6 hover:border-amber-500/50 transition-all duration-300 hover:shadow-lg">
                        <div className="flex items-center gap-4 mb-3">
                          <div className="w-12 h-12 bg-amber-500 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                            <Crosshair className="w-6 h-6 text-white" />
                          </div>
                          <h5 className="text-lg font-bold text-white">Data-Driven</h5>
                        </div>
                        <p className="text-gray-300 text-sm leading-relaxed">
                          Programs tailored to your unique physiology, goals, and lifestyle using data-driven optimization principles.
                        </p>
                      </div>
                    </div>

                    {/* Results Pillar */}
                    <div className="group">
                      <div className="bg-gradient-to-r from-yellow-500/10 to-yellow-300/5 border border-yellow-500/20 rounded-xl p-6 hover:border-yellow-400/50 transition-all duration-300 hover:shadow-lg">
                        <div className="flex items-center gap-4 mb-3">
                          <div className="w-12 h-12 bg-gradient-to-br from-amber-500 to-amber-600 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                            <Trophy className="w-6 h-6 text-white" />
                          </div>
                          <h5 className="text-lg font-bold text-white">Results-Guaranteed</h5>
                        </div>
                        <p className="text-gray-300 text-sm leading-relaxed">
                          Continuous monitoring and systematic adjustments ensure measurable progress toward your transformation goals.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* New Book Release Banner - Exact Match from Screenshot */}
          <div className="max-w-5xl mx-auto mt-20 animate-fade-in">
            <div className="relative bg-gradient-to-br from-amber-600/20 via-amber-500/15 to-amber-400/10 backdrop-blur-xl border-4 border-amber-500/40 hover:border-amber-500/60 rounded-2xl p-8 shadow-2xl hover:shadow-amber-500/20 transition-all duration-500">
              {/* Enhanced floating icon */}
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <div className="bg-gradient-to-br from-amber-600 to-amber-500 rounded-full p-3 shadow-lg">
                  <BookOpen className="w-8 h-8 text-white" />
                </div>
              </div>
              
              <div className="relative pt-6 text-center">
                <div className="flex items-center justify-center space-x-3 mb-4">
                  <span className="font-bold text-lg" style={{ color: '#F29C00' }}>New Book Release</span>
                </div>
                
                <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                  The Science of Enhancement
                </h2>
                
                <p className="text-xl text-amber-200 leading-relaxed max-w-4xl mx-auto mb-8">
                  Get exclusive insights into my evidence-based methodology, complete with research references, 
                  practical protocols, and real client case studies.
                </p>
                
                <Button 
                  className="bg-repz-orange hover:bg-repz-orange-dark text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 hover:scale-105 shadow-lg hover:shadow-orange-500/25"
                >
                  <Star className="w-5 h-5 mr-2" />
                  Pre-Order Now
                </Button>
              </div>
            </div>
          </div>

        </div>
      </section>


      <SectionBoundary variant="top" />

      {/* Enhanced FAQ Section - Gentle Progression */}
      <section id="faq" className="py-20 relative overflow-hidden bg-gradient-to-br from-black via-gray-950 to-slate-900"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Beautiful animated background similar to stats and about */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Elegant gradient orbs */}
          <div className="absolute top-32 left-40 w-80 h-80 bg-gradient-to-br from-purple-500/15 via-pink-500/10 to-repz-orange/8 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-32 right-32 w-60 h-60 bg-gradient-to-br from-repz-orange/18 via-amber-500/12 to-yellow-500/8 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2.5s'}}></div>
          
          {/* Sophisticated geometric patterns */}
          <div className="absolute top-20 right-28 w-28 h-28 border border-purple-400/20 rounded-full animate-ping" style={{animationDelay: '1s', animationDuration: '5s'}}></div>
          <div className="absolute bottom-24 left-24 w-20 h-20 bg-gradient-to-br from-repz-orange/8 to-transparent rotate-45 animate-pulse" style={{animationDelay: '3s'}}></div>
          
          {/* Elegant flowing SVG */}
          <svg className="absolute inset-0 w-full h-full opacity-12" viewBox="0 0 1200 600">
            <path
              d="M0,400 Q300,250 600,400 T1200,400"
              stroke="url(#faqGradient)"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
            />
            <defs>
              <linearGradient id="faqGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.4"/>
                <stop offset="50%" stopColor="#f97316" stopOpacity="0.6"/>
                <stop offset="100%" stopColor="#ec4899" stopOpacity="0.3"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl relative z-10">
          <AnimatedSection className="text-center mb-16">
            {/* Modern Card Header - Following established pattern */}
            <div className="relative group mb-8 flex justify-center">
              <div className="relative bg-gradient-to-br from-yellow-600/8 via-yellow-500/5 to-yellow-400/3 backdrop-blur-3xl border-2 border-yellow-500/25 hover:border-yellow-500/40 rounded-3xl px-8 py-6 shadow-xl w-full max-w-2xl hover:shadow-yellow-500/15 transition-all duration-500 text-center bg-black/20">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="bg-gradient-to-br from-yellow-600 to-yellow-500 rounded-full p-4 shadow-lg">
                    <span className="text-3xl">❓</span>
                  </div>
                </div>
                <div className="pt-6">
                  <h2 className="text-3xl md:text-4xl lg:text-5xl font-montserrat font-bold mb-2 leading-tight text-center"
                      style={{ 
                        color: '#EAB308',
                        textShadow: '0 2px 4px rgba(234, 179, 8, 0.25)'
                      }}>
                    Frequently Asked Questions
                  </h2>
                  <p className="text-yellow-200 font-semibold text-base md:text-lg">Everything you need to know</p>
                </div>
              </div>
            </div>
            
            <p className="text-lg md:text-xl font-inter text-yellow-200 max-w-3xl mx-auto">
              Get detailed answers to the most common questions about our coaching programs.
            </p>
          </AnimatedSection>

          <div className="max-w-6xl mx-auto">
            {/* Search Bar */}
            <AnimatedSection animation="fade-in" className="mb-8">
              <div className="relative max-w-2xl mx-auto">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <Input
                  placeholder="Search questions..."
                  value={faqSearch}
                  onChange={(e) => setFaqSearch(e.target.value)}
                  className="pl-12 h-12 text-base border-muted/40 focus:border-repz-orange bg-background/90 text-foreground backdrop-blur-sm"
                />
              </div>
            </AnimatedSection>

            {/* Enhanced Category Filter Tabs */}
            <AnimatedSection animation="fade-in" className="mb-12">
              <div className="flex flex-wrap gap-3 justify-center">
                <Button
                  onClick={() => setSelectedCategory("all")}
                  className={`h-12 px-6 text-sm font-medium rounded-xl transition-all duration-300 ${
                    selectedCategory === "all"
                      ? "bg-gradient-to-r from-amber-600 to-yellow-600 text-white shadow-lg shadow-amber-600/30"
                      : "bg-surface-glass text-amber-400 border border-amber-400/30 hover:bg-amber-400/10 backdrop-blur-sm hover:shadow-lg hover:shadow-amber-400/20"
                  }`}
                >
                  <span className="mr-2">🔥</span>
                  All Questions ({faqData.length})
                </Button>
                <Button
                  onClick={() => setSelectedCategory("programs")}
                  className={`h-12 px-6 text-sm font-medium rounded-xl transition-all duration-300 ${
                    selectedCategory === "programs"
                      ? "bg-gradient-to-r from-amber-500 to-amber-600 text-white shadow-lg shadow-amber-500/30"
                      : "bg-surface-glass text-amber-400 border border-amber-400/30 hover:bg-amber-400/10 backdrop-blur-sm hover:shadow-lg hover:shadow-amber-400/20"
                  }`}
                >
                  <span className="mr-2">📋</span>
                  Programs ({faqData.filter(q => q.category === "programs").length})
                </Button>
                <Button
                  onClick={() => setSelectedCategory("results")}
                  className={`h-12 px-6 text-sm font-medium rounded-xl transition-all duration-300 ${
                    selectedCategory === "results"
                      ? "bg-gradient-to-r from-amber-600 to-yellow-500 text-white shadow-lg shadow-amber-600/30"
                      : "bg-surface-glass text-amber-400 border border-amber-400/30 hover:bg-amber-400/10 backdrop-blur-sm hover:shadow-lg hover:shadow-amber-400/20"
                  }`}
                >
                  Results ({faqData.filter(q => q.category === "results").length})
                </Button>
                <Button
                  onClick={() => setSelectedCategory("pricing")}
                  className={`h-12 px-6 text-sm font-medium rounded-xl transition-all duration-300 ${
                    selectedCategory === "pricing"
                      ? "bg-gradient-to-r from-amber-500 to-yellow-600 text-white shadow-lg shadow-amber-500/30"
                      : "bg-surface-glass text-amber-400 border border-amber-400/30 hover:bg-amber-400/10 backdrop-blur-sm hover:shadow-lg hover:shadow-amber-400/20"
                  }`}
                >
                  Pricing ({faqData.filter(q => q.category === "pricing").length})
                </Button>
                <Button
                  onClick={() => setSelectedCategory("support")}
                  className={`h-12 px-6 text-sm font-medium rounded-xl transition-all duration-300 ${
                    selectedCategory === "support"
                      ? "bg-gradient-to-r from-amber-500 to-yellow-500 text-white shadow-lg shadow-amber-500/30"
                      : "bg-surface-glass text-amber-400 border border-amber-400/30 hover:bg-amber-400/10 backdrop-blur-sm hover:shadow-lg hover:shadow-amber-400/20"
                  }`}
                >
                  Support ({faqData.filter(q => q.category === "support").length})
                </Button>
                <Button
                  onClick={() => setSelectedCategory("technical")}
                  className={`h-12 px-6 text-sm font-medium rounded-xl transition-all duration-300 ${
                    selectedCategory === "technical"
                      ? "bg-gradient-to-r from-amber-600 to-orange-500 text-white shadow-lg shadow-amber-600/30"
                      : "bg-surface-glass text-amber-400 border border-amber-400/30 hover:bg-amber-400/10 backdrop-blur-sm hover:shadow-lg hover:shadow-amber-400/20"
                  }`}
                >
                  Technical ({faqData.filter(q => q.category === "technical").length})
                </Button>
              </div>
            </AnimatedSection>

            {/* Enhanced FAQ Cards - Grouped by Category */}
            <AnimatedSection animation="slide-up">
              <div className="space-y-8">
                {(() => {
                  // Group FAQs by category
                  const categorizedFAQs = faqData
                    .filter(item =>
                      item.question.toLowerCase().includes(faqSearch.toLowerCase()) ||
                      item.answer.toLowerCase().includes(faqSearch.toLowerCase())
                    )
                    .reduce((acc, faq) => {
                      if (selectedCategory === "all" || faq.category === selectedCategory) {
                        if (!acc[faq.category]) {
                          acc[faq.category] = [];
                        }
                        acc[faq.category].push(faq);
                      }
                      return acc;
                    }, {} as Record<string, typeof faqData>);

                  const categoryInfo = {
                    programs: { title: "Programs", color: "blue" },
                    results: { title: "Results", color: "green" },
                    pricing: { title: "Pricing", color: "purple" },
                    support: { title: "Support", color: "teal" },
                    technical: { title: "Technical", color: "gray" }
                  };

                  // Elegant color variations for each category
                  const colorVariations = {
                    programs: {
                      gradient: "from-amber-900/25 via-yellow-800/15 to-orange-900/10",
                      border: "border-amber-500/40 hover:border-amber-400/60",
                      iconBg: "from-amber-500 to-yellow-600",
                      shadow: "hover:shadow-amber-500/25",
                      textAccent: "text-amber-200"
                    },
                    results: {
                      gradient: "from-yellow-900/25 via-amber-800/15 to-yellow-900/10", 
                      border: "border-yellow-500/40 hover:border-yellow-400/60",
                      iconBg: "from-yellow-500 to-amber-600",
                      shadow: "hover:shadow-yellow-500/25",
                      textAccent: "text-yellow-200"
                    },
                    pricing: {
                      gradient: "from-orange-900/25 via-amber-800/15 to-orange-900/10",
                      border: "border-orange-500/40 hover:border-orange-400/60", 
                      iconBg: "from-orange-500 to-amber-600",
                      shadow: "hover:shadow-orange-500/25",
                      textAccent: "text-orange-200"
                    },
                    support: {
                      gradient: "from-amber-800/25 via-yellow-700/15 to-amber-900/10",
                      border: "border-amber-400/40 hover:border-amber-300/60",
                      iconBg: "from-amber-400 to-yellow-500", 
                      shadow: "hover:shadow-amber-400/25",
                      textAccent: "text-amber-100"
                    },
                    technical: {
                      gradient: "from-yellow-800/25 via-amber-700/15 to-yellow-900/10",
                      border: "border-yellow-400/40 hover:border-yellow-300/60",
                      iconBg: "from-yellow-400 to-amber-500",
                      shadow: "hover:shadow-yellow-400/25", 
                      textAccent: "text-yellow-100"
                    }
                  };

                  return Object.entries(categorizedFAQs).map(([category, faqs]) => {
                    const colors = colorVariations[category as keyof typeof colorVariations];
                    
                    return (
                      <div key={category} className="relative group">
                        {/* Elegant category card with black background and clean boundaries */}
                        <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl transition-all duration-500 hover:shadow-2xl hover:border-white/20">
                          <Accordion type="single" collapsible>
                            <AccordionItem value={category} className="border-none">
                              <AccordionTrigger className="text-left hover:no-underline px-8 py-6 group-hover:bg-white/5 transition-all duration-300">
                                <div className="flex items-center gap-6 text-left w-full">
                                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center text-white text-2xl shadow-lg backdrop-blur-sm transition-all duration-300 group-hover:scale-110 bg-gradient-to-br from-yellow-600 to-yellow-500">
                                    {categoryInfo[category as keyof typeof categoryInfo]?.icon}
                                  </div>
                                  <div className="flex-1">
                                    <h3 className="font-montserrat font-bold text-2xl text-foreground mb-2 group-hover:text-white/90 transition-colors">
                                      {categoryInfo[category as keyof typeof categoryInfo]?.title}
                                    </h3>
                                    <div className="flex items-center gap-4 text-white/80">
                                      <span className="text-sm">
                                        {faqs.length} question{faqs.length !== 1 ? 's' : ''}
                                      </span>
                                      <span className={cn(
                                        "text-xs px-3 py-1 rounded-full border transition-colors",
                                        "bg-amber-500/10 border-amber-400/30 group-hover:bg-amber-400/20"
                                      )}>
                                        Click to expand
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </AccordionTrigger>
                              <AccordionContent className="px-0 pb-0">
                                <div className="space-y-0 bg-black/60 backdrop-blur-sm">
                                  {faqs.map((faq, faqIndex) => (
                                    <div key={faqIndex} className="border-b border-white/10 last:border-b-0 transition-colors hover:bg-white/5 px-8 py-6">
                                      <details className="group/details">
                                        <summary className={cn(
                                          "cursor-pointer list-none flex items-center justify-between",
                                          "text-lg font-semibold text-foreground hover:text-white/90 transition-colors"
                                        )}>
                                          <span className="pr-4">{faq.question}</span>
                                          <ChevronDown className="w-5 h-5 text-yellow-300 group-open/details:rotate-180 transition-transform duration-200" />
                                        </summary>
                                        <div className="mt-4 pt-4 border-t border-white/10">
                                          <p className="leading-relaxed text-white/80">
                                            {faq.answer}
                                          </p>
                                        </div>
                                      </details>
                                    </div>
                                  ))}
                                </div>
                              </AccordionContent>
                            </AccordionItem>
                          </Accordion>
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            </AnimatedSection>

          </div>
        </div>
      </section>


      
      <SectionBoundary variant="top" />
      
      <section id="contact" className="py-12 relative overflow-hidden bg-gradient-to-br from-black via-gray-950 to-slate-900"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "translateZ(0)",
                 willChange: "transform",
                 WebkitFontSmoothing: "antialiased",
                 textRendering: "optimizeLegibility"
               }}>
        {/* Minimalist connection lines */}
        <div className="absolute inset-0 opacity-8">
          <div className="absolute top-1/3 left-1/6 w-8 h-px bg-gradient-to-r from-transparent via-yellow-300 to-transparent"></div>
          <div className="absolute bottom-1/3 right-1/6 w-8 h-px bg-gradient-to-r from-transparent via-yellow-300 to-transparent"></div>
        </div>
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
          <div className="text-center mb-12">
            {/* Modern Card Header - Following established pattern */}
            <div className="relative group mb-8 flex justify-center">
              <div className="relative bg-gradient-to-br from-gray-600/8 via-gray-500/5 to-gray-400/3 backdrop-blur-3xl border-2 border-gray-500/25 hover:border-gray-500/40 rounded-3xl px-8 py-6 shadow-xl w-full max-w-2xl hover:shadow-gray-500/15 transition-all duration-500 text-center bg-black/20">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="bg-gradient-to-br from-gray-600 to-gray-500 rounded-full p-4 shadow-lg">
                    <span className="text-3xl">💬</span>
                  </div>
                </div>
                <div className="pt-6">
                  <h2 className="text-3xl md:text-4xl lg:text-5xl font-montserrat font-bold mb-2 leading-tight text-center"
                      style={{ 
                        color: '#6B7280',
                        textShadow: '0 2px 4px rgba(107, 114, 128, 0.25)'
                      }}>
                    Contact Information
                  </h2>
                  <p className="text-gray-200 font-semibold text-base md:text-lg">Multiple ways to connect</p>
                </div>
              </div>
            </div>
            
            <p className="text-lg md:text-xl font-inter text-gray-200 max-w-3xl mx-auto">
              Multiple ways to connect. Choose what works best for you.
            </p>
          </div>
          
          <div className="max-w-5xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <AnimatedSection animation="slide-up" delay={0}>
                <div className="relative group">
                  <Card className="relative bg-black/40 backdrop-blur-xl border border-white/10 hover:border-white/20 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-500 group-hover:scale-105">
                    <CardContent className="p-8 text-center">
                      <div className="relative mb-6">
                        <div className="relative bg-gradient-to-r from-gray-600 to-gray-500 rounded-2xl p-4 w-fit mx-auto shadow-lg">
                          <Phone className="w-10 h-10 text-white" />
                        </div>
                      </div>
                      <h3 className="font-montserrat font-bold text-xl mb-3 text-white">SMS/Call</h3>
                      <a 
                        href="sms:+14159929792" 
                        className="text-gray-300 hover:text-white transition-colors font-semibold text-lg group-hover:underline"
                      >
                        +1 (415) 992-9792
                      </a>
                    </CardContent>
                  </Card>
                </div>
              </AnimatedSection>
              
              <AnimatedSection animation="slide-up" delay={150}>
                <div className="relative group">
                  <Card className="relative bg-black/40 backdrop-blur-xl border border-white/10 hover:border-white/20 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-500 group-hover:scale-105">
                    <CardContent className="p-8 text-center">
                      <div className="relative mb-6">
                        <div className="relative bg-gradient-to-r from-gray-600 to-gray-500 rounded-2xl p-4 w-fit mx-auto shadow-lg">
                          <Mail className="w-10 h-10 text-white" />
                        </div>
                      </div>
                      <h3 className="font-montserrat font-bold text-xl mb-3 text-white">Email Us</h3>
                      <a 
                        href="mailto:contact@repz.com" 
                        className="text-gray-300 hover:text-white transition-colors font-semibold text-lg group-hover:underline"
                      >
                        contact@repz.com
                      </a>
                    </CardContent>
                  </Card>
                </div>
              </AnimatedSection>
              
              <AnimatedSection animation="slide-up" delay={300}>
                <div className="relative group">
                  <Card className="relative bg-black/40 backdrop-blur-xl border border-white/10 hover:border-white/20 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-500 group-hover:scale-105">
                    <CardContent className="p-8 text-center">
                      <div className="relative mb-6">
                        <div className="relative bg-gradient-to-r from-gray-600 to-gray-500 rounded-2xl p-4 w-fit mx-auto shadow-lg">
                          <MapPin className="w-10 h-10 text-white" />
                        </div>
                      </div>
                      <h3 className="font-montserrat font-bold text-xl mb-3 text-white">Bay Area</h3>
                      <p className="text-gray-300 font-semibold text-lg">
                        In-person training available
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </AnimatedSection>
            </div>
          </div>
        </div>
      </section>

      {/* Section separator before footer */}
      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
      
      <Footer />
      
      {/* Authentication Modal */}
      <AuthModal
        isOpen={showAuthModal}
        mode={authMode}
        onClose={handleCloseAuthModal}
      />


      {/* Subscription Modal */}
      <SubscriptionModal
        isOpen={subscriptionModalOpen}
        onClose={() => setSubscriptionModalOpen(false)}
        selectedTier={selectedTier}
        tierPrice={pricingTiers.find(tier => tier.name === selectedTier)?.price.toString() || '299'}
        onChangeTier={handleChangeTier}
      />
    </div>
  );
};

export default RepzHome;