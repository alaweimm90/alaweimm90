import React from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useIsMobile } from "@/hooks/use-mobile";
import { AppSidebar } from "@/components/AppSidebar";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { EnhancedErrorBoundary } from "@/components/EnhancedErrorBoundary";
import { DomainThemeProvider } from "@/components/DomainThemeProvider";
import { FocusManager, SkipToContent } from "@/components/AccessibilityEnhancements";
import { PerformanceMonitor } from "@/components/PerformanceOptimizations";
import { PWAInstallBanner } from "@/components/PWAInstallBanner";
import { ThemeProvider } from "@/components/EnhancedDarkMode";
import { ProductionErrorBoundary, usePerformanceMonitor, ProductionStatusIndicator } from '@/components/ProductionReadiness';
import { GradientBackground } from '@/components/EnhancedUIPolish';
import { Button } from "@/components/ui/button";
import { Mail, Linkedin, Github, Globe, GraduationCap, Atom } from "lucide-react";
import BrandLogo from "@/components/ui/BrandLogo";
import Index from "./pages/Index";
import {
  LazyGrapheneBandStructure,
  LazyMoS2ValleyPhysics,
  LazyTDSESolver,
  LazyCrystalVisualizer,
  LazyLLGDynamics,
  LazyBlochSphereDynamics,
  LazyQuantumTunneling,
  LazyBZFolding,
  LazyPhononBandStructure,
  LazyQuantumFieldTheory,
  LazyLaplaceEigenmodes,
  LazyPINNSchrodinger,
  LazyMLShowcase,
  LazyAdvancedSimulation,
  LazyErrorReporting,
  LazyAccessibilityFeatures,
  LazyPWAFeatures,
  LazyIsingModel,
  LazyBoltzmannDistribution,
  LazyMicrostatesEntropy,
  LazyCanonicalEnsemble,
  LazyBrownianMotion,
  LazyDocumentation,
  LazyNotFound,
  LazySimulationDashboard,
  LazyScientificComputing,
  LazyAboutPlatform,
  LazyInteractiveLearning,
  LazyEnhancedVisualization,
  LazyDataExport,
  LazyWebWorkerDemo,
} from "./components/LazyRoutes";
import { LazyTestingDashboard } from "./components/LazyRoutes";

const queryClient = new QueryClient();

const App = () => {
  const isMobile = useIsMobile();
  usePerformanceMonitor();
  
  return (
    <ProductionErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <ThemeProvider>
            <FocusManager>
              <PerformanceMonitor>
                <EnhancedErrorBoundary 
                  enableRecovery={true}
                  showErrorDetails={import.meta.env.DEV}
                >
                  <ErrorBoundary>
                    <Toaster />
                    <Sonner />
                  <BrowserRouter>
                    <DomainThemeProvider defaultDomain="quantum">
                <SidebarProvider>
                  <GradientBackground variant="subtle" className="min-h-screen">
                    <div className="min-h-screen flex w-full bg-background/80 backdrop-blur-sm">
                  <SkipToContent />
                  <AppSidebar />
                  
                  <div className="flex-1 flex flex-col">
                     <header className="h-auto sm:h-16 border-b border-border/50 bg-card/50 backdrop-blur-sm px-4 py-2 sm:py-0">
                      <div className="flex items-center justify-between gap-3">
                        {/* Left: global sidebar trigger */}
                        <div className="flex items-center">
                          <SidebarTrigger className="hover-scale touch-manipulation" />
                        </div>
                        {/* Center: brand */}
                        <div className="flex-1 flex justify-center">
                          <div className="flex flex-col items-center text-center">
                            <div className="flex items-center">
                              <BrandLogo inline size="md" withIcon />
                              <span className="sr-only">Simulation Core</span>
                            </div>
                            <span className="text-xs sm:text-sm text-muted-foreground text-center px-2 leading-tight">
                              Interactive Scientific Computing Platform
                            </span>
                          </div>
                        </div>
                        {/* Right: contact icons */}
                        <div className="hidden sm:flex items-center gap-1">
                          <Button variant="ghost" size="icon" aria-label="Email" title="Email" onClick={() => window.open('mailto:meshal@berkeley.edu')}
                            className="h-9 w-9">
                            <Mail className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" aria-label="LinkedIn" title="LinkedIn" onClick={() => window.open('https://linkedin.com/in/meshal-alawein', '_blank', 'noopener,noreferrer')}
                            className="h-9 w-9">
                            <Linkedin className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" aria-label="GitHub" title="GitHub" onClick={() => window.open('https://github.com/alaweimm90', '_blank', 'noopener,noreferrer')}
                            className="h-9 w-9">
                            <Github className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" aria-label="Website" title="Website" onClick={() => window.open('https://malawein.com', '_blank', 'noopener,noreferrer')}
                            className="h-9 w-9">
                            <Globe className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" aria-label="Google Scholar" title="Google Scholar" onClick={() => window.open('https://scholar.google.com/citations?user=IB_E6GQAAAAJ', '_blank', 'noopener,noreferrer')}
                            className="h-9 w-9">
                            <GraduationCap className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </header>

                     {/* Main content area */}
                     <main id="main-content" className="flex-1 relative">
                        <Routes>
                         <Route path="/" element={<Index />} />
                         
                         {/* Physics Module Routes */}
                         <Route path="/modules/graphene-band-structure" element={<LazyGrapheneBandStructure />} />
                         <Route path="/modules/mos2-valley-physics" element={<LazyMoS2ValleyPhysics />} />
                         <Route path="/modules/tdse-solver" element={<LazyTDSESolver />} />
                         <Route path="/modules/crystal-visualizer" element={<LazyCrystalVisualizer />} />
                         <Route path="/modules/llg-dynamics" element={<LazyLLGDynamics />} />
                         <Route path="/modules/bloch-sphere" element={<LazyBlochSphereDynamics />} />
                         <Route path="/modules/quantum-tunneling" element={<LazyQuantumTunneling />} />
                         <Route path="/modules/bz-folding" element={<LazyBZFolding />} />
                         <Route path="/modules/phonon-band-structure" element={<LazyPhononBandStructure />} />
                         <Route path="/modules/quantum-field-theory" element={<LazyQuantumFieldTheory />} />
                          <Route path="/modules/laplace-eigenmodes" element={<LazyLaplaceEigenmodes />} />
                          <Route path="/modules/pinn-schrodinger" element={<LazyPINNSchrodinger />} />
                          <Route path="/modules/ml-showcase" element={<LazyMLShowcase />} />
                          <Route path="/modules/ising-model" element={<LazyIsingModel />} />
                         <Route path="/modules/boltzmann-distribution" element={<LazyBoltzmannDistribution />} />
                         <Route path="/modules/microstates-entropy" element={<LazyMicrostatesEntropy />} />
                         <Route path="/modules/canonical-ensemble" element={<LazyCanonicalEnsemble />} />
                         <Route path="/modules/brownian-motion" element={<LazyBrownianMotion />} />
                         
                         {/* Platform Feature Routes */}
                         <Route path="/advanced-simulation" element={<LazyAdvancedSimulation />} />
                         <Route path="/simulation-dashboard" element={<LazySimulationDashboard />} />
                         <Route path="/interactive-learning" element={<LazyInteractiveLearning />} />
                         <Route path="/enhanced-visualization" element={<LazyEnhancedVisualization />} />
                         <Route path="/data-export" element={<LazyDataExport />} />
                         <Route path="/scientific-computing" element={<LazyScientificComputing />} />
                         <Route path="/documentation" element={<LazyDocumentation />} />
                         <Route path="/error-reporting" element={<LazyErrorReporting />} />
                         <Route path="/ml-showcase" element={<LazyMLShowcase />} />
                         <Route path="/about-platform" element={<LazyAboutPlatform />} />
                          <Route path="/accessibility-features" element={<LazyAccessibilityFeatures />} />
                           <Route path="/pwa-features" element={<LazyPWAFeatures />} />
                           <Route path="/webworker-demo" element={<LazyWebWorkerDemo />} />
                           <Route path="/testing-dashboard" element={<LazyTestingDashboard />} />
                           
                           {/* Catch-all route */}
                         <Route path="*" element={<LazyNotFound />} />
                       </Routes>
                    </main>
                  </div>
                    </div>
                  </GradientBackground>
                  <PWAInstallBanner />
                  <ProductionStatusIndicator />
                </SidebarProvider>
              </DomainThemeProvider>
            </BrowserRouter>
                  </ErrorBoundary>
                </EnhancedErrorBoundary>
        </PerformanceMonitor>
      </FocusManager>
    </ThemeProvider>
  </TooltipProvider>
</QueryClientProvider>
</ProductionErrorBoundary>
);
};

export default App;