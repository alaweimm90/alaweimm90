import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PhysicsBackground } from '@/components/PhysicsBackground';
import { OptimizedModuleCard, VirtualizedModuleGrid } from '@/components/PerformanceOptimizedComponents';
import { QuantumFieldVisualization } from '@/components/QuantumFieldVisualization';
import { InteractiveModuleGrid } from '@/components/InteractiveModuleGrid';
import { ProgressiveLoader, ModuleCardSkeleton } from '@/components/AdvancedLoadingStates';
import StartingPointSection from '@/components/StartingPointSection';
import SectionHeader from '@/components/SectionHeader';
import { FeatureShowcasePanel } from '@/components/FeatureShowcasePanel';
import { PhysicsButton, PageTransition } from '@/components/EnhancedUIPolish';
import { useToast } from '@/hooks/use-toast';

import { AdvancedAnalytics } from '@/components/AdvancedAnalytics';
import { PerformanceBenchmark } from '@/components/PerformanceBenchmark';
import { physicsModules } from '@/data/modules';
import { categoryToGroupDomain } from '@/lib/category-domain-map';
import { useDomainTheme } from '@/components/DomainThemeProvider';
import { Search, Atom, Zap, BookOpen, Github, ArrowRight, SkipForward, Mail, ExternalLink, GraduationCap, User, Brain, Cpu, Microscope, FlaskConical, Linkedin, Globe, Monitor } from 'lucide-react';
import { useAnnouncements, useSkipLink, usePrefersReducedMotion } from '@/lib/accessibility-utils';
import BrandLogo from '@/components/ui/BrandLogo';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useInputValidation } from '@/lib/input-validation';
import { handleUserInputError } from '@/lib/error-handling';

import { InteractiveTutorial, platformTutorialSteps } from '@/components/InteractiveTutorial';

import { ParticleSystem } from '@/components/ParticleSystem';
import { PhysicsVisualization } from '@/components/PhysicsVisualization';
import { AnimatedLoading } from '@/components/AnimatedLoading';
import { FloatingActionButton } from '@/components/FloatingActionButton';
import { useSEO } from '@/hooks/use-seo';
import { OptimizedImage } from '@/components/OptimizedImage';

const Index = () => {
  useSEO({ title: 'SimCore – Interactive Scientific Computing', description: 'Explore interactive physics, mathematics, and ML simulations in your browser.' });
  
  const schemaOrg = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'Organization',
    name: 'SimCore',
    url: typeof window !== 'undefined' ? window.location.origin : '',
    logo: '/favicon.ico',
    sameAs: [
      'https://github.com',
      'https://www.linkedin.com',
      'https://twitter.com'
    ]
  }), []);

  const schemaWebSite = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'SimCore',
    url: typeof window !== 'undefined' ? window.location.origin : '',
    potentialAction: {
      '@type': 'SearchAction',
      target: `${typeof window !== 'undefined' ? window.location.origin : ''}?search={search_term_string}`,
      'query-input': 'required name=search_term_string'
    }
  }), []);

  const schemaCollection = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'CreativeWork',
    name: 'Interactive Physics Simulations',
    description: 'Collection of computational physics modules and simulations',
    creator: {
      '@type': 'Person',
      name: 'SimCore Development Team'
    }
  }), []);

  const navigate = useNavigate();
  const { toast } = useToast();
  const { announcement } = useAnnouncements();
  const prefersReducedMotion = usePrefersReducedMotion();
  const { setDomain } = useDomainTheme();

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [isLoading, setIsLoading] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showTutorial, setShowTutorial] = useState(false);

  const { validateSearchInput } = useInputValidation();

  // Check if tutorial should be shown
  useEffect(() => {
    const hasSeenTutorial = localStorage.getItem('simcore-tutorial-completed');
    if (!hasSeenTutorial) {
      setShowTutorial(true);
    }
  }, []);

  // Filter modules based on search and category
  const filteredModules = physicsModules.filter(module => {
    const matchesSearch = module.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         module.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         module.category.toLowerCase().includes(searchTerm.toLowerCase());
    const group = categoryToGroupDomain(module.category);
    const matchesCategory = selectedCategory === 'All' || group === (selectedCategory as any);
    return matchesSearch && matchesCategory;
  });

  const implementedCount = physicsModules.filter(m => m.isImplemented).length;
  const groupDomains = ["Physics", "Mathematics", "Scientific ML"] as const;

  // Map tabs to theme domains
  const getThemeDomain = (category: string) => {
    switch(category) {
      case 'All': 
      case 'Physics': return 'quantum';
      case 'Mathematics': return 'fields';
      case 'Scientific ML': return 'energy';
      default: return 'quantum';
    }
  };

  // Simulate initial loading
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  // Handle module exploration with transition
  const handleModuleExplore = (route?: string) => {
    if (!route) return;
    setIsTransitioning(true);
    setTimeout(() => {
      navigate(route);
    }, 300);
  };

  // Enhanced category change with animation and theme
  const handleCategoryChange = (category: string) => {
    setIsTransitioning(true);
    setDomain(getThemeDomain(category));
    setTimeout(() => {
      setSelectedCategory(category);
      setIsTransitioning(false);
    }, 200);
  };

  const handleTutorialComplete = () => {
    setShowTutorial(false);
    localStorage.setItem('simcore-tutorial-completed', 'true');
    toast({
      title: "Welcome to SimCore!",
      description: "You're all set to explore computational physics. Enjoy your journey!",
    });
  };

  const handleTutorialSkip = () => {
    setShowTutorial(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-cosmic relative overflow-hidden flex items-center justify-center">
        <QuantumFieldVisualization intensity={0.5} />
        <ProgressiveLoader 
          stage={2}
          progress={75}
          stages={['Initializing', 'Loading SimCore', 'Ready']}
        />
      </div>
    );
  }

  return (
    <PageTransition>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaOrg) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaWebSite) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaCollection) }} />
      
      <div className="min-h-screen bg-gradient-cosmic relative overflow-hidden">
        <QuantumFieldVisualization />
      
        {/* Skip Links for Accessibility */}
        <div className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-50">
          <Button
            variant="outline"
            size="sm"
            data-skip-link
            onClick={() => {
              document.getElementById('main-content')?.focus();
              document.getElementById('main-content')?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="bg-background border-2 border-primary"
          >
            <SkipForward className="w-4 h-4 mr-2" />
            Skip to main content
          </Button>
        </div>

        {/* Live region for announcements */}
        <div aria-live="polite" aria-atomic="true" className="sr-only">
          {announcement}
        </div>
        
        {/* Hero Section with Interactive Animation */}
        <header className="relative z-10 pt-20 pb-16 px-4" role="banner">
          <div className="max-w-6xl mx-auto text-center">
            {/* Interactive Quantum Field Animation */}
            <div className="relative mb-8 overflow-hidden rounded-3xl bg-gradient-to-br from-background/80 via-card/50 to-background/80 backdrop-blur-xl border border-border/30 p-8">
              {/* Animated Particles */}
              <div className="absolute inset-0 pointer-events-none complex-animations" aria-hidden="true">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div
                    key={i}
                    className={`absolute w-2 h-2 bg-primary/30 rounded-full ${
                      !prefersReducedMotion ? 'animate-quantum-drift' : ''
                    }`}
                    style={{
                      left: `${Math.random() * 100}%`,
                      top: `${Math.random() * 100}%`,
                      animationDelay: `${i * 0.5}s`,
                      animationDuration: `${8 + Math.random() * 4}s`
                    }}
                  />
                ))}
              </div>

              {/* Main Title with Enhanced Animation */}
                <div className={`relative z-10 ${!prefersReducedMotion ? 'animate-fade-in' : ''}`}>
                  <div className="flex items-center justify-center mb-4">
                    <BrandLogo inline size="xl" withIcon />
                  </div>


                  <p className="text-xl max-w-3xl mx-auto leading-relaxed mb-8 text-muted-foreground text-center">
                  Interactive scientific computing platform for physics, mathematics, and scientific-ML. 
                  Bridging theoretical frontiers across physics, math, and AI-driven discovery.
                </p>

                {/* Scientific Dedication Quote */}
                <div className={`relative mb-6 ${!prefersReducedMotion ? 'animate-quantum-pulse-advanced' : ''}`}>
                  <div className="relative max-w-2xl mx-auto">
                    {/* Background schematic patterns */}
                    <div className="absolute inset-0 opacity-10 pointer-events-none">
                      <svg className="w-full h-full" viewBox="0 0 400 120" fill="none">
                        {/* Orbital paths */}
                        <ellipse cx="50" cy="60" rx="30" ry="15" stroke="currentColor" strokeWidth="0.5" className="text-primary"/>
                        <ellipse cx="50" cy="60" rx="20" ry="30" stroke="currentColor" strokeWidth="0.5" className="text-accent"/>
                        {/* Contour lines */}
                        <path d="M100 20 Q200 40 300 20" stroke="currentColor" strokeWidth="0.3" className="text-primary-glow"/>
                        <path d="M100 40 Q200 60 300 40" stroke="currentColor" strokeWidth="0.3" className="text-primary-glow"/>
                        <path d="M100 60 Q200 80 300 60" stroke="currentColor" strokeWidth="0.3" className="text-primary-glow"/>
                        {/* Grid dots */}
                        {Array.from({ length: 20 }).map((_, i) => (
                          <circle key={i} cx={320 + (i % 5) * 15} cy={30 + Math.floor(i / 5) * 15} r="0.8" fill="currentColor" className="text-accent/60"/>
                        ))}
                      </svg>
                    </div>

                    <div className="relative z-10 text-center p-6 rounded-2xl card-surface-glass border border-border/30">
                      <blockquote className="text-sm italic text-muted-foreground leading-relaxed">
                        "Made with love, and a deep respect for the struggle."
                        <br />
                        "For those still learning, from someone who still is."
                      </blockquote>
                    </div>
                  </div>
                </div>

                {/* Enhanced CTA Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <Button 
                    size="lg" 
                    className={`
                      relative bg-gradient-quantum text-primary-foreground shadow-quantum
                      transition-all duration-300 group overflow-hidden
                      ${!prefersReducedMotion ? 'hover:scale-105 hover:shadow-2xl' : ''}
                    `}
                    onClick={() => document.getElementById('modules-section')?.scrollIntoView({ behavior: 'smooth' })}
                    aria-label="Start exploring interactive physics modules"
                  >
                    <div className="relative z-10 flex items-center gap-2">
                      <Atom className="w-5 h-5" />
                      Start Exploring
                      <ArrowRight className={`w-4 h-4 transition-transform duration-300 ${!prefersReducedMotion ? 'group-hover:translate-x-1' : ''}`} />
                    </div>
                    {!prefersReducedMotion && (
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 transform -skew-x-12" />
                    )}
                  </Button>

                  <Button 
                    variant="outline" 
                    size="lg"
                    className={`
                      bg-card/50 backdrop-blur-sm border-primary/20 hover:bg-primary/10 
                      hover:border-primary/40 transition-all duration-300
                      ${!prefersReducedMotion ? 'hover:scale-105' : ''}
                    `}
                    onClick={() => navigate('/documentation')}
                    aria-label="View comprehensive documentation"
                  >
                    <BookOpen className="w-4 h-4 mr-2" />
                    Documentation
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </header>



        {/* Starting Point Section */}
        <StartingPointSection />

        {/* Feature Showcase Panel */}
        <FeatureShowcasePanel />



        {/* Section divider */}
        <div className="flex items-center justify-center mt-16">
          <div
            className="h-px w-full max-w-3xl bg-gradient-to-r from-transparent via-primary/30 to-transparent"
            role="separator"
            aria-hidden="true"
          />
        </div>

        {/* Main Content */}
        <main id="main-content" tabIndex={-1} className="relative z-10 px-4 py-16" role="main">
          <div className="max-w-7xl mx-auto">
            <section id="modules-section" aria-labelledby="modules-heading">
              <div className="text-center mb-12">
                <SectionHeader
                  id="modules-heading"
                  title="Interactive Physics Laboratory"
                  subtitle={`Explore ${implementedCount} live simulations across multiple domains`}
                  variant="quantum"
                  styleType="panel"
                  eyebrow="Computational Physics Modules"
                />

                {/* Enhanced Search and Filter Section */}
                <div className="max-w-2xl mx-auto mb-8 space-y-6">
                  {/* Search with Enhanced Design */}
                  <div className="relative group">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground transition-colors duration-300 group-focus-within:text-primary" />
                    <Input
                      type="search"
                      placeholder="Search physics modules, theories, or concepts..."
                      value={searchTerm}
                      onChange={(e) => {
                        const validatedInput = validateSearchInput(e.target.value);
                        if (validatedInput.isValid) {
                          setSearchTerm(e.target.value);
                        } else {
                          console.warn('Search input validation failed:', validatedInput.error);
                        }
                      }}
                      className={`
                        pl-12 pr-4 py-4 text-lg rounded-xl 
                        bg-card/50 backdrop-blur-sm border-border/30 
                        focus:border-primary/50 focus:bg-card/70 transition-all duration-300
                        placeholder:text-muted-foreground/70
                        ${!prefersReducedMotion ? 'focus:scale-[1.02]' : ''}
                      `}
                      role="searchbox"
                      aria-label="Search physics modules"
                      autoComplete="off"
                    />
                  </div>
                </div>

                {/* Enhanced Category Filter Tabs with Animations */}
                <Tabs 
                  value={selectedCategory} 
                  onValueChange={handleCategoryChange}
                  className="w-full"
                >
                  <TabsList 
                    className="w-full flex items-center justify-start gap-[var(--semantic-spacing-inline)] rounded-xl bg-card/50 backdrop-blur-sm p-[var(--semantic-spacing-inline)] text-muted-foreground overflow-x-auto scrollbar-hide border border-border/30 min-h-[var(--touch-target-min)]"
                    role="tablist"
                    aria-label="Filter modules by category"
                  >
                    <TabsTrigger 
                      value="All"
                      role="tab"
                      aria-selected={selectedCategory === 'All'}
                      aria-controls="modules-grid"
                      className={`
                        flex-shrink-0 min-w-fit whitespace-nowrap px-4 py-2 rounded-lg
                        transition-all duration-300 ease-out
                        hover:bg-primary/10 hover:text-primary hover:shadow-lg hover:scale-105
                        data-[state=active]:bg-gradient-quantum data-[state=active]:text-primary-foreground 
                        data-[state=active]:shadow-quantum data-[state=active]:scale-105
                        font-medium
                      `}
                    >
                      <div className="flex items-center gap-2 justify-center min-w-0">
                        <span className="truncate">{selectedCategory === 'All' ? 'All' : 'All'}</span>
                        <Badge variant="secondary" className={`border-0 text-xs transition-colors duration-300 flex-shrink-0 ${selectedCategory === 'All' ? 'bg-[hsl(var(--semantic-background-primary))] text-[hsl(var(--semantic-text-primary))]' : 'bg-primary/20 text-primary'}`}>
                          {physicsModules.length}
                        </Badge>
                      </div>
                    </TabsTrigger>
                    {groupDomains.map((domain) => {
                      const count = physicsModules.filter(m => categoryToGroupDomain(m.category) === domain).length;
                      const isActive = selectedCategory === domain;

                      if (count === 0) return null;

                      return (
                        <TabsTrigger 
                          key={domain} 
                          value={domain}
                          role="tab"
                          aria-selected={isActive}
                          aria-controls="modules-grid"
                          className={`
                            flex-shrink-0 min-w-fit whitespace-nowrap px-4 py-2 rounded-lg
                            transition-all duration-300 ease-out
                            hover:bg-primary/10 hover:text-primary hover:shadow-lg hover:scale-105
                            data-[state=active]:bg-gradient-quantum data-[state=active]:text-primary-foreground 
                            data-[state=active]:shadow-quantum data-[state=active]:scale-105
                            font-medium
                          `}
                         >
                           <div className="flex items-center gap-2 justify-center min-w-0">
                             <span className="truncate">{domain}</span>
                             <Badge 
                               variant="secondary" 
                               className={`
                                 border-0 text-xs transition-colors duration-300 flex-shrink-0
                                  ${isActive ? 'bg-[hsl(var(--semantic-background-primary))] text-[hsl(var(--semantic-text-primary))]' : 'bg-primary/20 text-primary'}
                               `}
                             >
                              {count}
                            </Badge>
                          </div>
                        </TabsTrigger>
                      );
                    }).filter(Boolean)}
                  </TabsList>
                </Tabs>
              </div>
            </section>

            {/* Enhanced Module Grid with Advanced Interactions */}
            <div className="relative">
              {/* Grid glow effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5 rounded-3xl blur-3xl" />
              
              <div className="relative">
                {isTransitioning ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 sm:gap-6">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <ModuleCardSkeleton key={i} />
                    ))}
                  </div>
                ) : filteredModules.length > 0 ? (
                  <InteractiveModuleGrid
                    modules={filteredModules}
                    onModuleExplore={handleModuleExplore}
                    onModuleTheory={() => navigate('/documentation')}
                  />
                ) : (
                  <div 
                    className="col-span-full text-center py-16 relative" 
                    role="status"
                    aria-live="polite"
                  >
                    {/* Empty state with animation */}
                    <div className="relative max-w-md mx-auto">
                      <div className={`${!prefersReducedMotion ? 'animate-bounce' : ''} mb-6`}>
                        <div className="w-24 h-24 bg-gradient-quantum rounded-full mx-auto flex items-center justify-center shadow-quantum">
                          <Search className="w-12 h-12 text-primary-foreground" />
                        </div>
                      </div>
                      
                      <h3 className="text-2xl font-bold text-foreground mb-4">
                        No modules found
                      </h3>
                      <p className="text-muted-foreground mb-6 leading-relaxed">
                        Refine your search parameters or explore different research domains to access our computational physics laboratory.
                      </p>
                      
                      <Button
                        variant="outline"
                        size="lg"
                        onClick={() => {
                          setSearchTerm('');
                          setSelectedCategory('All');
                        }}
                        className={`
                          bg-card/50 backdrop-blur-sm border-primary/20 hover:bg-primary/10 
                          hover:border-primary/40 transition-all duration-300
                          ${!prefersReducedMotion ? 'hover:scale-105' : ''}
                        `}
                        aria-label="Clear all filters and show all modules"
                      >
                        <Zap className="w-4 h-4 mr-2" />
                        Reset Filters
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>

        {/* Section Separator */}
        <div className="flex items-center justify-center my-20">
          <div className="h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent w-full max-w-3xl"></div>
        </div>

        {/* About the Creator Section */}
        <section className="relative z-10 py-16 px-4">
          <div className="max-w-7xl mx-auto px-4 md:px-6">
            <SectionHeader
              title="About the Creator"
              subtitle="Meet the computational physicist behind SimCore"
              variant="quantum"
              styleType="panel"
              eyebrow="SimCore Creator & Author"
            />

            {/* Enhanced Profile Card - Hero Style */}
            <div className="relative rounded-3xl overflow-hidden card-surface-glass card-hover-raise">
              {/* Scattered Subtle Physics Background */}

              <div className="relative p-8 md:p-12">
                {/* Profile Header - Centered */}
                <div className="text-center mb-12">
                  {/* Larger Avatar Section */}
                  <div className="flex justify-center mb-6">
                    <div className={`relative ${!prefersReducedMotion ? 'hover:scale-105 transition-all duration-500' : ''}`}>
                      <div className="w-56 h-56 md:w-64 md:h-64 rounded-3xl overflow-hidden shadow-quantum border border-border bg-gradient-to-br from-background/80 to-card/60 backdrop-blur-sm">
                        <OptimizedImage 
                          src="/lovable-uploads/9446dbe7-d528-4ecf-a88c-9ceda59c414a.png"
                          alt="Dr. Meshal Alawein – Computational Physicist"
                          className="w-full h-full object-cover transition-all duration-700"
                          sizes="(max-width: 768px) 224px, 256px"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Profile Info */}
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-indigo-600 via-cyan-400 to-indigo-600 bg-clip-text text-transparent mb-3">
                        Meshal Alawein
                      </h3>
                      <p className="text-xl bg-gradient-to-r from-purple-400 via-purple-600 to-purple-500 bg-clip-text text-transparent mb-4">
                        Computational Physicist | Quantum Materials | Scientific Computing
                      </p>
                      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl card-surface-glass border border-border mb-4">
                        <GraduationCap className="w-4 h-4 text-primary" />
                        <span className="text-sm font-medium text-foreground">PhD, UC Berkeley</span>
                      </div>
                    </div>

                    {/* Playful Bio */}
                    <p className="text-base text-muted-foreground leading-relaxed italic max-w-3xl mx-auto">
                      I build simulation frameworks to control misbehaving electrons, bend new materials to my will, and prototype smart quantum systems — just smart enough to accelerate physics, but not smart enough to start a robot uprising. Yet. 🧠🤖
                    </p>
                  </div>
                </div>

                {/* Research & Projects in compact grid */}
                <div className="relative grid md:grid-cols-2 gap-8">
                  <div className="hidden md:block absolute inset-y-0 left-1/2 w-px bg-gradient-to-b from-transparent via-border to-transparent" />
                  {/* Research Topics with Panel Styling */}
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-foreground flex items-center gap-2">
                      ⚛️ Research Playgrounds
                    </h4>
                    <div className="space-y-4">
                      {[
                        { 
                          area: "Quantum 2D Materials", 
                          desc: "Flat-band hunting in TMD monolayers for Mott insulators and weird correlated stuff we haven't named yet",
                          subItems: [
                            "Electronic topology engineering in twisted bilayer graphene and TMD heterostructures",
                            "Many-body localization phenomena in disordered 2D quantum magnets and spin liquids"
                          ],
                          tools: [
                            { name: "SIESTA", color: "bg-blue-500/20 text-blue-700 dark:text-blue-300" },
                            { name: "VASP", color: "bg-fuchsia-500/20 text-fuchsia-700 dark:text-fuchsia-300" },
                            { name: "Quantum ESPRESSO", color: "bg-sky-500/20 text-sky-700 dark:text-sky-300" },
                            { name: "ASE", color: "bg-purple-500/20 text-purple-700 dark:text-purple-300" }
                          ]
                        },
                        { 
                          area: "Multiscale Simulations", 
                          desc: "DFT + MD hybrids with SIESTA, LAMMPS, ASE, SLURM — glued together with Python and duct tape",
                          subItems: [
                            "Seamless QM/MM coupling for reactive interfaces in catalytic systems and battery materials",
                            "Machine learning force fields trained on DFT data for ns-timescale materials dynamics"
                          ],
                          tools: [
                            { name: "Python", color: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-300" },
                            { name: "SIESTA", color: "bg-blue-500/20 text-blue-700 dark:text-blue-300" },
                            { name: "VASP", color: "bg-fuchsia-500/20 text-fuchsia-700 dark:text-fuchsia-300" },
                            { name: "LAMMPS", color: "bg-green-500/20 text-green-700 dark:text-green-300" },
                            { name: "ASE", color: "bg-purple-500/20 text-purple-700 dark:text-purple-300" },
                            { name: "SLURM", color: "bg-orange-500/20 text-orange-700 dark:text-orange-300" },
                            { name: "Bash", color: "bg-gray-500/20 text-gray-700 dark:text-gray-300" }
                          ]
                        },
                        { 
                          area: "Quantum Device Modeling", 
                          desc: "Circuit-level simulations for spin-orbitronics and magnonic wizardry",
                          subItems: [
                            "Spin-wave propagation modeling in magnonic crystals and metamaterials",
                            "Non-equilibrium transport in quantum dots coupled to superconducting leads"
                          ],
                          tools: [
                            { name: "Python", color: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-300" },
                            { name: "MuMax³", color: "bg-teal-500/20 text-teal-700 dark:text-teal-300" },
                            { name: "OOMMF", color: "bg-lime-500/20 text-lime-700 dark:text-lime-300" },
                            { name: "SIESTA", color: "bg-blue-500/20 text-blue-700 dark:text-blue-300" },
                            { name: "TranSIESTA", color: "bg-blue-600/20 text-blue-800 dark:text-blue-300" },
                            { name: "MATLAB", color: "bg-red-500/20 text-red-700 dark:text-red-300" },
                            { name: "Mathematica", color: "bg-rose-500/20 text-rose-700 dark:text-rose-300" }
                          ]
                        },
                        { 
                          area: "Variational Algorithms", 
                          desc: "Turning Hamiltonians into headaches (and maybe insights) with Qiskit and VQE",
                          subItems: [
                            "Quantum approximate optimization algorithms (QAOA) for combinatorial problems in materials discovery",
                            "Variational quantum eigensolver implementations for strongly correlated electron systems"
                          ],
                          tools: [
                            { name: "Python", color: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-300" },
                            { name: "Qiskit", color: "bg-indigo-500/20 text-indigo-700 dark:text-indigo-300" },
                            { name: "Cirq", color: "bg-amber-500/20 text-amber-700 dark:text-amber-300" }
                          ]
                        },
                        { 
                          area: "Materials Informatics", 
                          desc: "CIFs, JSONs, and regret. Structure curation with pymatgen and data-fueled optimism",
                          subItems: [
                            "High-throughput screening of perovskite solar cell materials using graph neural networks",
                            "Automated crystal structure prediction and stability analysis for energy storage applications"
                          ],
                          tools: [
                            { name: "Python", color: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-300" },
                            { name: "pymatgen", color: "bg-emerald-500/20 text-emerald-700 dark:text-emerald-300" },
                            { name: "ASE", color: "bg-purple-500/20 text-purple-700 dark:text-purple-300" },
                            { name: "Mathematica", color: "bg-rose-500/20 text-rose-700 dark:text-rose-300" }
                          ]
                        },
                        { 
                          area: "Band Structure Modeling", 
                          desc: "Custom tight-binding models and effective mass whispering in quantum zoos",
                          subItems: [
                            "k·p perturbation theory for semiconductor heterostructures and quantum wells",
                            "Wannier function interpolation for accurate band structure calculations in complex materials"
                          ],
                          tools: [
                            { name: "Python", color: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-300" },
                            { name: "VASP", color: "bg-fuchsia-500/20 text-fuchsia-700 dark:text-fuchsia-300" },
                            { name: "Quantum ESPRESSO", color: "bg-sky-500/20 text-sky-700 dark:text-sky-300" }
                          ]
                        }
                      ].map(({ area, desc, subItems, tools }, i) => (
                        <div key={area} className={`group rounded-xl p-4 transition-all duration-300 bg-card/30 backdrop-blur-sm border border-border/40 hover:border-primary/30 hover:bg-card/50 hover:shadow-lg ${!prefersReducedMotion ? 'hover:scale-[1.02]' : ''}`}>
                          <div className="space-y-3">
                            <div className="flex items-start gap-3">
                              <div className={`w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0 ${!prefersReducedMotion ? 'group-hover:animate-pulse' : ''}`} />
                              <div className="flex-1">
                                <div className={`text-sm font-semibold text-foreground mb-1 ${!prefersReducedMotion ? 'group-hover:text-primary transition-colors duration-300' : ''}`}>
                                  {area}
                                </div>
                                <p className="text-xs text-muted-foreground leading-tight mb-3">{desc}</p>
                                
                                {/* Sub-bullet points with square markers */}
                                <div className="space-y-1.5 mb-3">
                                  {subItems.map((item, idx) => (
                                    <div key={idx} className="flex items-start gap-2">
                                      <div className="w-1.5 h-1.5 bg-accent/70 rounded-sm mt-1.5 flex-shrink-0" />
                                      <p className="text-xs text-muted-foreground/90 leading-relaxed">{item}</p>
                                    </div>
                                  ))}
                                </div>
                                
                                <div className="flex flex-wrap gap-1">
                                  {tools.map((tool) => (
                                    <Badge 
                                      key={tool.name} 
                                      variant="secondary" 
                                      className={`text-xs px-2 py-0.5 cursor-pointer ${tool.color} border-0 ${!prefersReducedMotion ? 'hover:scale-105 transition-transform duration-200' : ''}`}
                                      onClick={() => {
                                        const toolUrls = {
                                          'SIESTA': 'https://siesta-project.org/',
                                          'LAMMPS': 'https://www.lammps.org/',
                                          'ASE': 'https://wiki.fysik.dtu.dk/ase/',
                                          'SLURM': 'https://slurm.schedmd.com/',
                                          'VASP': 'https://www.vasp.at/',
                                          'Quantum ESPRESSO': 'https://www.quantum-espresso.org/',
                                          'Qiskit': 'https://qiskit.org/',
                                          'Cirq': 'https://quantumai.google/cirq',
                                          'VQE': 'https://qiskit.org/textbook/ch-algorithms/vqe.html',
                                          'pymatgen': 'https://pymatgen.org/',
                                          'MATLAB': 'https://www.mathworks.com/products/matlab.html',
                                          'Bash': 'https://www.gnu.org/software/bash/',
                                          'Python': 'https://python.org/'
                                        };
                                        if (toolUrls[tool.name]) {
                                          window.open(toolUrls[tool.name], '_blank', 'noopener,noreferrer');
                                        }
                                      }}
                                    >
                                      {tool.name}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Current Projects with personality */}
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-foreground flex items-center gap-2">
                      🧪 Current Playgrounds
                    </h4>
                    
                    {/* Animated Research Visualization - Updated for Current Playgrounds */}
                    <div className="relative my-6 overflow-hidden">
                      <div className={`flex space-x-4 ${!prefersReducedMotion ? 'animate-scroll-horizontal' : ''}`}>
                        <div className="flex space-x-4 min-w-max">
                          {/* Dirac cones */}
                          <div className={`px-3 py-1 rounded-full text-xs ${!prefersReducedMotion ? 'animate-float-slow' : ''} card-surface-glass border border-border/30 text-foreground/80`} aria-label="Dirac cone illustration">
                            <svg width="80" height="22" viewBox="0 0 80 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                              <path d="M5 18 L40 4 L75 18" stroke="currentColor" strokeWidth="1.5"/>
                              <path d="M5 4 L40 18 L75 4" stroke="currentColor" strokeWidth="1.5"/>
                              <circle cx="40" cy="11" r="1.5" fill="currentColor"/>
                            </svg>
                          </div>
                          {/* Bloch sphere */}
                          <div className={`px-3 py-1 rounded-full text-xs ${!prefersReducedMotion ? 'animate-float-slow' : ''} card-surface-glass border border-border/30 text-foreground/80`} aria-label="Bloch sphere illustration">
                            <svg width="80" height="22" viewBox="0 0 80 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                              <g transform="translate(18,2)">
                                <circle cx="22" cy="9" r="8" stroke="currentColor" strokeWidth="1.3"/>
                                <line x1="14" y1="9" x2="30" y2="9" stroke="currentColor" strokeWidth="1"/>
                                <line x1="22" y1="1" x2="22" y2="17" stroke="currentColor" strokeWidth="1" opacity=".7"/>
                                <path d="M22 9 L28 5" stroke="currentColor" strokeWidth="1.5"/>
                                <polygon points="28,5 26,7 30,7" fill="currentColor"/>
                              </g>
                            </svg>
                          </div>
                          {/* Honeycomb lattice */}
                          <div className={`px-3 py-1 rounded-full text-xs ${!prefersReducedMotion ? 'animate-float-slow' : ''} card-surface-glass border border-border/30 text-foreground/80`} aria-label="Honeycomb lattice illustration">
                            <svg width="80" height="22" viewBox="0 0 80 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                              <g transform="translate(8,4)" stroke="currentColor" strokeWidth="1.2">
                                <polygon points="8,0 16,0 20,7 16,14 8,14 4,7" fill="none"/>
                                <circle cx="8" cy="0" r="1" fill="currentColor"/>
                                <circle cx="16" cy="0" r="1" fill="currentColor"/>
                                <circle cx="20" cy="7" r="1" fill="currentColor"/>
                                <circle cx="16" cy="14" r="1" fill="currentColor"/>
                                <circle cx="8" cy="14" r="1" fill="currentColor"/>
                                <circle cx="4" cy="7" r="1" fill="currentColor"/>
                              </g>
                            </svg>
                          </div>
                          {/* Quantum circuit (CNOT motif) */}
                          <div className={`px-3 py-1 rounded-full text-xs ${!prefersReducedMotion ? 'animate-float-slow' : ''} card-surface-glass border border-border/30 text-foreground/80`} aria-label="Quantum circuit illustration">
                            <svg width="110" height="22" viewBox="0 0 110 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                              <g transform="translate(8,6)" stroke="currentColor" strokeWidth="1.5">
                                <line x1="0" y1="4" x2="90" y2="4"/>
                                <line x1="0" y1="12" x2="90" y2="12"/>
                                <circle cx="24" cy="4" r="2.5" fill="currentColor"/>
                                <circle cx="66" cy="12" r="5"/>
                                <line x1="66" y1="7" x2="66" y2="17"/>
                                <line x1="61" y1="12" x2="71" y2="12"/>
                              </g>
                            </svg>
                          </div>
                          {/* Ising spins */}
                          <div className={`px-3 py-1 rounded-full text-xs ${!prefersReducedMotion ? 'animate-float-slow' : ''} card-surface-glass border border-border/30 text-foreground/80`} aria-label="Ising spins illustration">
                            <svg width="100" height="22" viewBox="0 0 100 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                              <g transform="translate(8,3)" stroke="currentColor" strokeWidth="1.5">
                                <path d="M5 14 L5 2"/><polygon points="5,2 2,6 8,6" fill="currentColor"/>
                                <path d="M25 2 L25 14"/><polygon points="25,14 22,10 28,10" fill="currentColor"/>
                                <path d="M45 14 L45 2"/><polygon points="45,2 42,6 48,6" fill="currentColor"/>
                                <path d="M65 2 L65 14"/><polygon points="65,14 62,10 68,10" fill="currentColor"/>
                              </g>
                            </svg>
                          </div>
                          {/* Energy landscape / wave */}
                          <div className={`px-3 py-1 rounded-full text-xs ${!prefersReducedMotion ? 'animate-float-slow' : ''} card-surface-glass border border-border/30 text-foreground/80`} aria-label="Energy landscape illustration">
                            <svg width="100" height="22" viewBox="0 0 100 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                              <path d="M2 11 C 12 3, 22 3, 32 11 S 52 19, 62 11 82 3, 98 11" stroke="currentColor" strokeWidth="1.5" fill="none"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="rounded-xl p-4 bg-card/30 backdrop-blur-sm border border-border/40 hover:border-primary/30 transition-colors">
                        <div className="flex items-start gap-3 mb-3">
                          <span className="text-lg">🧠</span>
                          <div>
                            <div className="text-sm font-medium text-foreground">Graph-Based Quantum Simulation</div>
                            <p className="text-xs text-muted-foreground">Playing with graph topologies as if they were LEGO sets for tight-binding models</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="rounded-xl p-4 bg-card/30 backdrop-blur-sm border border-border/40 hover:border-primary/30 transition-colors">
                        <div className="flex items-start gap-3 mb-3">
                          <span className="text-lg">🤖</span>
                          <div>
                            <div className="text-sm font-medium text-foreground">Scientific ML for Materials</div>
                            <p className="text-xs text-muted-foreground">GNNs, PINNs, and surrogate DFT — for when I want neural nets to approximate physics (and occasionally hallucinate)</p>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-xl p-4 bg-card/30 backdrop-blur-sm border border-border/40 hover:border-primary/30 transition-colors">
                        <div className="flex items-start gap-3 mb-3">
                          <span className="text-lg">🎓</span>
                          <div>
                            <div className="text-sm font-medium text-foreground">QubeML Education</div>
                            <p className="text-xs text-muted-foreground mb-2">Colab notebooks, clean code, and a zero-judgment zone for getting stuck 🤝</p>
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={() => window.open('https://github.com/alaweimm90/QubeML', '_blank', 'noopener,noreferrer')}
                              className="text-xs h-6"
                            >
                              Come play with me
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Quantum Computing & Materials Informatics Visualization */}
                    <div className="mt-8 relative">
                      <div className="group relative rounded-2xl overflow-hidden card-field-surface p-6">
                        {/* Subtle glow overlay */}
                        <div aria-hidden className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity duration-700 glow-overlay-field" />
                        {/* Header */}
                        <div className="text-center mb-6">
                          <h5 className="text-sm font-semibold text-foreground/80 mb-2">Interactive Quantum-Materials Sandbox</h5>
                          <p className="text-xs text-muted-foreground">Real-time visualization of quantum states and material properties</p>
                        </div>

                        {/* Quantum Circuit & Materials Grid */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6">
                          {/* Quantum Circuit Visualization */}
                          <div className="relative">
                            <div className="text-xs font-medium text-foreground/70 mb-3 flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                              Quantum Circuit Evolution
                            </div>
                            <div className="relative h-32 rounded-lg card-surface-glass border border-border/30 overflow-hidden">
                              {/* Quantum Gates */}
                              <div className="absolute inset-0 flex items-center justify-center">
                                <div className="flex items-center space-x-4">
                                  {/* Hadamard Gate */}
                                  <div className={`w-8 h-8 bg-blue-500/20 border border-blue-500/40 rounded flex items-center justify-center ${!prefersReducedMotion ? 'animate-pulse' : ''}`}>
                                    <span className="text-xs font-bold text-blue-400">H</span>
                                  </div>
                                  {/* Connection */}
                                  <div className="w-6 h-px bg-blue-400/60"></div>
                                  {/* CNOT Gate */}
                                  <div className={`w-8 h-8 bg-purple-500/20 border border-purple-500/40 rounded-full flex items-center justify-center ${!prefersReducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: '0.5s' }}>
                                    <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                                  </div>
                                  {/* Connection */}
                                  <div className="w-6 h-px bg-purple-400/60"></div>
                                  {/* Measurement */}
                                  <div className={`w-8 h-8 bg-green-500/20 border border-green-500/40 rounded flex items-center justify-center ${!prefersReducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: '1s' }}>
                                    <span className="text-xs font-bold text-green-400">M</span>
                                  </div>
                                </div>
                              </div>
                              {/* Floating Qubits */}
                              {Array.from({ length: 3 }).map((_, i) => (
                                <div
                                  key={i}
                                  className={`absolute w-2 h-2 bg-blue-400/60 rounded-full ${!prefersReducedMotion ? 'animate-quantum-drift' : ''}`}
                                  style={{
                                    left: `${20 + i * 30}%`,
                                    top: `${20 + i * 15}%`,
                                    animationDelay: `${i * 0.7}s`,
                                    animationDuration: '4s'
                                  }}
                                />
                              ))}
                            </div>
                            {/* Quantum State Display */}
                            <div className="mt-2 text-xs text-center">
                              <InlineMath math="|\\psi\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)" />
                            </div>
                          </div>

                          {/* Materials Property Prediction */}
                          <div className="relative">
                            <div className="text-xs font-medium text-foreground/70 mb-3 flex items-center gap-2">
                              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                              ML-Driven Property Prediction
                            </div>
                            <div className="relative h-32 rounded-lg card-surface-glass border border-border/30 overflow-hidden">
                              {/* Neural Network Visualization */}
                              <div className="absolute inset-0 flex items-center justify-center">
                                <div className="grid grid-cols-3 gap-4 items-center">
                                  {/* Input Layer */}
                                  <div className="flex flex-col space-y-2">
                                    {Array.from({ length: 3 }).map((_, i) => (
                                      <div key={i} className={`w-3 h-3 bg-green-400/40 rounded-full ${!prefersReducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: `${i * 0.3}s` }}></div>
                                    ))}
                                  </div>
                                  {/* Hidden Layer */}
                                  <div className="flex flex-col space-y-1">
                                    {Array.from({ length: 4 }).map((_, i) => (
                                      <div key={i} className={`w-2 h-2 bg-cyan-400/40 rounded-full ${!prefersReducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: `${i * 0.2}s` }}></div>
                                    ))}
                                  </div>
                                  {/* Output Layer */}
                                  <div className="flex flex-col space-y-2">
                                    {Array.from({ length: 2 }).map((_, i) => (
                                      <div key={i} className={`w-3 h-3 bg-orange-400/40 rounded-full ${!prefersReducedMotion ? 'animate-pulse' : ''}`} style={{ animationDelay: `${i * 0.4}s` }}></div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                              {/* Data Flow Lines */}
                              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                                {Array.from({ length: 6 }).map((_, i) => (
                                  <line
                                    key={i}
                                    x1={`${25 + (i % 3) * 10}%`}
                                    y1={`${30 + (i % 3) * 20}%`}
                                    x2={`${65 + (i % 2) * 10}%`}
                                    y2={`${40 + (i % 2) * 20}%`}
                                    stroke="hsl(var(--accent))"
                                    strokeWidth="0.5"
                                    opacity="0.3"
                                    className={!prefersReducedMotion ? 'animate-pulse' : ''}
                                    style={{ animationDelay: `${i * 0.1}s` }}
                                  />
                                ))}
                              </svg>
                            </div>
                            {/* Property Output */}
                            <div className="mt-2 text-xs text-center">
                              <InlineMath math="E_{gap} = \\sigma(W \\cdot \\vec{x} + b)" />
                            </div>
                          </div>

                          {/* Band Structure Preview */}
                          <div className="relative">
                            <div className="text-xs font-medium text-foreground/70 mb-3 flex items-center gap-2">
                              <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                              Band Structure Preview
                            </div>
                            <div className="relative h-32 rounded-lg card-surface-glass border border-border/30 overflow-hidden">
                              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 100" aria-hidden="true">
                                <path d="M0,80 C40,60 60,20 100,30 C140,40 160,80 200,60" stroke="hsl(var(--primary))" strokeWidth="1" fill="none" opacity="0.7" />
                                <path d="M0,50 C40,30 60,70 100,60 C140,50 160,20 200,40" stroke="hsl(var(--accent))" strokeWidth="1" fill="none" opacity="0.5" />
                              </svg>
                            </div>
                          </div>

                          {/* Spin Texture Map */}
                          <div className="relative">
                            <div className="text-xs font-medium text-foreground/70 mb-3 flex items-center gap-2">
                              <div className="w-2 h-2 bg-pink-400 rounded-full"></div>
                              Spin Texture Map
                            </div>
                            <div className="relative h-32 rounded-lg card-surface-glass border border-border/30 overflow-hidden">
                              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 100" aria-hidden="true">
                                {Array.from({ length: 30 }).map((_, i) => (
                                  <circle key={i} cx={(i % 10) * 20 + 10} cy={Math.floor(i / 10) * 20 + 10} r="2" fill="hsl(var(--semantic-domain-quantum))" opacity="0.6" />
                                ))}
                              </svg>
                            </div>
                          </div>
                        </div>

                        {/* Live Data Stream */}
                        <div className="mt-4 pt-4 border-t border-border/20">
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2">
                              <div className={`w-2 h-2 bg-green-400 rounded-full ${!prefersReducedMotion ? 'animate-pulse' : ''}`}></div>
                              <span className="text-muted-foreground">Live simulation active</span>
                            </div>
                            <div className="flex space-x-4 text-muted-foreground">
                              <span>Qubits: 2</span>
                              <span>•</span>
                              <span>Materials: 1,247</span>
                              <span>•</span>
                              <span>Accuracy: 94.2%</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>


                {/* Highlighted Projects */}
                <div className="mt-8 pt-6 border-t border-border/30">
                  <h4 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                    📂 Tools I Build When I Should Be Sleeping
                    <Badge variant="outline" className="text-xs">Online & Maintained</Badge>
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {[
                      { name: "📘 QubeML", desc: "Educational quantum computing + materials informatics framework", url: "https://github.com/alaweimm90/QubeML" },
                      { name: "🔷 QMatSim", desc: "DFT + MD pipeline for flat-band treasure hunting", url: "https://github.com/alaweimm90/QMatSim" },
                      { name: "🧮 SciComp", desc: "Physics in multiple dialects (multilingual simulations)", url: "https://github.com/alaweimm90/SciComp" },
                      { name: "🌀 SpinCirc", desc: "SPICE-compatible simulator for not-yet-obsolete device concepts", url: "https://github.com/alaweimm90/SpinCirc" },
                      { name: "🧲 MagLogic", desc: "Modeling toolkit for spin-based mischief", url: "https://github.com/alaweimm90/MagLogic" },
                      { name: "🌐 SimCore", desc: "This platform: pretty plots and buttons that still don't work", url: "https://simcore.dev" }
                    ].map(({ name, desc, url }) => (
                      <button
                        key={name} 
                        className={`group flex items-center justify-between gap-3 w-full text-left rounded-lg p-3 transition-all duration-300 hover:bg-gradient-to-r hover:from-primary/5 hover:to-accent/5 hover:shadow-sm border border-transparent hover:border-primary/20 ${!prefersReducedMotion ? 'hover:scale-[1.02]' : ''}`}
                        onClick={() => window.open(url, '_blank', 'noopener,noreferrer')}
                      >
                        <div className="min-w-0 flex-1">
                          <div className={`text-sm font-medium text-foreground ${!prefersReducedMotion ? 'group-hover:text-primary transition-colors duration-300' : ''}`}>
                            {name}
                          </div>
                          <p className="text-xs text-muted-foreground line-clamp-1">{desc}</p>
                        </div>
                        <ExternalLink className={`w-3 h-3 text-muted-foreground flex-shrink-0 ${!prefersReducedMotion ? 'group-hover:text-primary group-hover:scale-110 transition-all duration-300' : ''}`} />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Benchmark */}
        <div className="max-w-7xl mx-auto px-4 md:px-6 mb-12">
          <PerformanceBenchmark />
        </div>

        {/* Compact Analytics at Footer */}
        <div className="max-w-7xl mx-auto px-4 md:px-6 mb-12">
          <AdvancedAnalytics variant="compact" />
        </div>

        {/* Poetic Footer */}
        <footer className="relative z-10 py-16 px-4 animate-fade-in">
          <div className="max-w-4xl mx-auto">
            <div className="text-center border-t border-border/20 pt-8">
              <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground/60">
                 <span>© 2025 Meshal Alawein</span>
                 <span>•</span>
                 <BrandLogo inline size="sm" withIcon={false} />
              </div>
            </div>
          </div>
         </footer>

         {/* Floating Action Button */}
         <FloatingActionButton />
      </div>
    </PageTransition>
  );
};

export default Index;