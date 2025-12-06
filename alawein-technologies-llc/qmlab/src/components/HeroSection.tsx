import { Button } from "@/components/ui/button";
import { EnhancedButton } from "@/components/ui/enhanced-button";
import { Card } from "@/components/ui/card";
import { StatusChip } from "@/components/ui/status-chip";
import { MiniBlochSphere } from "@/components/MiniBlochSphere";
import { QuantumParticleField } from "@/components/QuantumParticleField";
import { ArrowRight, Zap, Eye, Play, Activity, Cpu, Layers, BookOpen, Lightbulb } from "lucide-react";
import { trackQuantumEvents } from "@/lib/analytics";

export const HeroSection = () => {
  const scrollToPlayground = () => {
    const mainContent = document.getElementById('main-content');
    if (mainContent) {
      mainContent.scrollIntoView({ behavior: 'smooth' });
    }
    trackQuantumEvents.featureDiscovery('hero_start_building');
  };

  const scrollToLearning = () => {
    const learningSection = document.getElementById('learning-resources');
    if (learningSection) {
      learningSection.scrollIntoView({ behavior: 'smooth' });
    }
    trackQuantumEvents.featureDiscovery('hero_learn_quantum');
  };
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
      {/* Subtle gradient background */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 opacity-90" />

      <div className="container mx-auto px-6 py-16 relative z-10">
        <div className="grid lg:grid-cols-12 gap-12 items-center max-w-7xl mx-auto">
          
          {/* Left Column: Content */}
          <div className="lg:col-span-7 space-y-8">
            {/* Main headline */}
            <div>
              <div className="flex items-center gap-2 mb-6">
                <div className="px-3 py-1 rounded-sm bg-slate-800/50 border border-slate-700 text-slate-400 text-xs uppercase tracking-wider font-medium">
                  Quantum Computing Lab
                </div>
              </div>
              <h1 className="font-display text-6xl md:text-7xl font-bold mb-4 text-white">
                QMLab
              </h1>
              <p className="text-2xl md:text-3xl text-slate-300 leading-tight mb-6 font-light">
                Quantum Machine Learning Research Platform
              </p>
              <p className="text-base text-slate-500 max-w-lg leading-relaxed">
                Design and simulate quantum circuits. Visualize state evolution on Bloch spheres. 
                Train variational quantum algorithms for classification tasks.
              </p>
            </div>

            {/* Enhanced Educational CTAs */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button 
                variant="primary" 
                size="lg" 
                onClick={scrollToPlayground}
                className="group h-12 px-6 text-sm font-medium bg-white text-slate-900 hover:bg-slate-100 transition-colors"
              >
                Launch Playground
                <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </Button>
              <Button 
                variant="outline" 
                size="lg" 
                onClick={scrollToLearning}
                className="h-12 px-6 text-sm font-medium border-slate-700 text-slate-300 hover:bg-slate-800/50 transition-colors"
              >
                Documentation
              </Button>
            </div>
            
            {/* Status indicator */}
            <div className="flex items-center gap-4 pt-4">
              <div className="flex items-center gap-2 text-xs text-slate-600">
                <div className="w-1.5 h-1.5 bg-green-500/80 rounded-full"></div>
                <span>Environment ready</span>
              </div>
              <EnhancedButton
                variant="cta-interactive"
                size="sm"
                onClick={() => {
                  const searchEvent = new KeyboardEvent('keydown', {
                    key: 'k',
                    metaKey: true,
                    bubbles: true
                  });
                  document.dispatchEvent(searchEvent);
                }}
                className="text-xs"
              >
                ⚡ Quick Search (⌘K)
              </EnhancedButton>
            </div>

            {/* Educational Status */}
            <div className="flex flex-wrap gap-3">
              <StatusChip variant="idle" icon={<Activity className="w-3 h-3" />}>
                System: Ready
              </StatusChip>
              <StatusChip variant="pure" icon={<Zap className="w-3 h-3" />}>
                Quantum: Superposition
              </StatusChip>
              <StatusChip variant="success" icon={<Lightbulb className="w-3 h-3" />}>
                Learning Mode
              </StatusChip>
            </div>

            {/* Quick Learning Modules */}
            <div className="space-y-3">
              <span className="text-sm text-muted-foreground font-medium">Start with quantum concepts:</span>
              <div className="flex flex-wrap gap-3">
                <button className="group px-4 py-2 text-sm bg-gradient-to-r from-blue-500/20 to-blue-400/10 hover:from-blue-500/30 hover:to-blue-400/20 border border-blue-400/30 hover:border-blue-400/50 text-blue-300 hover:text-blue-200 rounded-xl transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10 quantum-hover-glow">
                  <span className="relative z-10">VQE Algorithm</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></div>
                </button>
                <button className="group px-4 py-2 text-sm bg-gradient-to-r from-purple-500/20 to-purple-400/10 hover:from-purple-500/30 hover:to-purple-400/20 border border-purple-400/30 hover:border-purple-400/50 text-purple-300 hover:text-purple-200 rounded-xl transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/10 quantum-hover-glow">
                  <span className="relative z-10">Bell States</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></div>
                </button>
                <button className="group px-4 py-2 text-sm bg-gradient-to-r from-indigo-500/20 to-indigo-400/10 hover:from-indigo-500/30 hover:to-indigo-400/20 border border-indigo-400/30 hover:border-indigo-400/50 text-indigo-300 hover:text-indigo-200 rounded-xl transition-all duration-300 hover:shadow-lg hover:shadow-indigo-500/10 quantum-hover-glow">
                  <span className="relative z-10">Quantum ML</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></div>
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Live Mini Bloch Sphere */}
          <div className="lg:col-span-5 flex justify-center">
            <div className="relative w-80 h-80 rounded-3xl bg-slate-900/80 border border-slate-600/40 p-8 shadow-xl">
              <div className="absolute top-4 left-4 right-4 flex justify-between items-center">
                <span className="text-small font-mono text-muted">Live Preview</span>
                <StatusChip size="sm" variant="running">
                  H → Rz(θ) → CNOT
                </StatusChip>
              </div>
              
              <div className="mt-8 mb-4">
                <MiniBlochSphere 
                  className="w-full h-48" 
                  animated={true}
                  preset="hadamard"
                />
              </div>
              
              <div className="space-y-2 text-center">
                <div className="grid grid-cols-2 gap-4 text-small font-mono">
                  <div>
                    <span className="text-muted">|0⟩:</span>
                    <span className="ml-1 text-text">0.71</span>
                  </div>
                  <div>
                    <span className="text-muted">|1⟩:</span>
                    <span className="ml-1 text-text">0.71</span>
                  </div>
                </div>
                <div className="text-small text-muted font-mono">
                  ψ = (|0⟩ + |1⟩)/√2
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Cards - Redesigned with hierarchy */}
        <div className="grid md:grid-cols-3 gap-6 mt-24 max-w-6xl mx-auto">
          {/* Primary Lab Card */}
          <Card className="relative rounded-2xl border border-blue-400/30 bg-slate-900/80 pl-4 pr-6 py-6 shadow-xl hover:border-blue-400/50 transition-all duration-300">
            <div className="absolute left-0 top-6 bottom-6 w-1 bg-blue-400 rounded-r-full"></div>
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-blue-500/20 border border-blue-400/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <Zap className="w-5 h-5 text-blue-400" />
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold mb-2 text-slate-200">Interactive Circuits</h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Drag, drop, and configure quantum gates with real-time visualization
                </p>
                <ArrowRight className="w-4 h-4 text-blue-400 mt-3" />
              </div>
            </div>
          </Card>

          <Card className="relative rounded-2xl border border-purple-400/30 bg-slate-900/80 pl-4 pr-6 py-6 shadow-xl hover:border-purple-400/50 transition-all duration-300">
            <div className="absolute left-0 top-6 bottom-6 w-1 bg-purple-400 rounded-r-full"></div>
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-purple-500/20 border border-purple-400/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <Eye className="w-5 h-5 text-purple-400" />
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold mb-2 text-slate-200">Live Visualization</h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Watch quantum states evolve on the Bloch sphere in real-time
                </p>
                <ArrowRight className="w-4 h-4 text-purple-400 mt-3" />
              </div>
            </div>
          </Card>

          <Card className="relative rounded-2xl border border-indigo-400/30 bg-slate-900/80 pl-4 pr-6 py-6 shadow-xl hover:border-indigo-400/50 transition-all duration-300">
            <div className="absolute left-0 top-6 bottom-6 w-1 bg-indigo-400 rounded-r-full"></div>
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-indigo-500/20 border border-indigo-400/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <Layers className="w-5 h-5 text-indigo-400" />
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold mb-2 text-slate-200">Zero Boilerplate</h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Focus on learning - no setup or complex code required
                </p>
                <ArrowRight className="w-4 h-4 text-indigo-400 mt-3" />
              </div>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
};