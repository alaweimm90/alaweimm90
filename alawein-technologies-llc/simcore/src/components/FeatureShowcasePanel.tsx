import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { ElegantPhysicsVisualization } from '@/components/ElegantPhysicsVisualization';
import { useDomainTheme } from '@/components/DomainThemeProvider';
import SectionHeader from '@/components/SectionHeader';
import { Zap, BookOpen, Atom, Cpu, Layers, BarChart3, Plus, Minus, RotateCcw, Maximize2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { Dialog, DialogContent, DialogTrigger } from '@/components/ui/dialog';
import BrandLogo from '@/components/ui/BrandLogo';

export const FeatureShowcasePanel: React.FC = () => {
  const prefersReducedMotion = usePrefersReducedMotion();
  const { currentDomain } = useDomainTheme();

  const container = {
    hidden: { opacity: 0, y: 16 },
    show: { opacity: 1, y: 0, transition: { staggerChildren: 0.08, duration: 0.4, ease: 'easeOut' } }
  } as const;

  const item = {
    hidden: { opacity: 0, y: 12, scale: 0.98 },
    show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.35 } }
  } as const;

  const features = [
    {
      icon: Zap,
      title: 'High Performance',
      description:
        'WebGPU acceleration and optimized algorithms for real-time scientific simulations.',
      tags: ['WebGPU', 'Real-time', 'Optimized'],
    },
    {
      icon: BookOpen,
      title: 'Theory & Practice',
      description:
        'Complete derivations alongside interactive visualizations for deep understanding.',
      tags: ['LaTeX', 'Interactive', 'Educational'],
    },
    {
      icon: Atom,
      title: 'Research-Grade',
      description:
        'Export data, reproduce results, and integrate with your workflow with confidence.',
      tags: ['Export', 'Reproducible', 'Integration'],
    },
    {
      icon: Cpu,
      title: 'Advanced Computing',
      description:
        'Finite difference, Monte Carlo, and ML techniques for state-of-the-art simulations.',
      tags: ['Numerical', 'Monte Carlo', 'ML'],
    },
    {
      icon: Layers,
      title: 'Multi-Physics',
      description:
        'Coupled simulations across quantum, statistical, and materials domains.',
      tags: ['Quantum', 'Statistical', 'Materials'],
    },
    {
      icon: BarChart3,
      title: 'Data Analytics',
      description:
        'Built-in visualization and statistics for comprehensive result interpretation.',
      tags: ['Visualization', 'Statistics', 'Analysis'],
    },
  ];

  return (
    <section
      id="feature-showcase"
      aria-labelledby="feature-showcase-heading"
      className="my-16 md:my-24"
    >
      <div className="max-w-7xl mx-auto px-4 md:px-6">
        <SectionHeader
          id="feature-showcase-heading"
          title="Explore Powerful Capabilities"
          subtitle="Experience cutting-edge physics simulations and explore our powerful platform capabilities in real-time!"
          variant="quantum"
          styleType="panel"
          eyebrow="Platform Features Showcase"
        />

        <div className="space-y-16">
          
          {/* Interactive Scientific Demonstrations - Enhanced */}
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h3 className="text-[clamp(20px,2.5vw,28px)] font-bold mb-2 flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10 border border-primary/20 backdrop-blur-sm">
                ⚛️ Interactive Scientific Demonstrations
              </h3>
              <p className="text-base text-muted-foreground font-medium">
                Real-time physics simulations with full interactive controls
              </p>
            </div>
            
            <div className="rounded-3xl bg-gradient-to-b from-background/40 via-card/30 to-background/40 border border-border/30 p-6 md:p-8">
              <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 ${!prefersReducedMotion ? 'animate-fade-in' : ''}`}>
                <motion.div
                  initial={prefersReducedMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
                  viewport={{ once: true, amount: 0.2 }}
                  whileHover={prefersReducedMotion ? undefined : { scale: 1.02 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                >
                  <div className={`group relative rounded-2xl overflow-hidden min-h-[280px] lg:min-h-[320px] card-field-surface`}>
                    <div aria-hidden className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity duration-700 glow-overlay-field" />
                    <TransformWrapper>
                      {({ zoomIn, zoomOut, resetTransform }) => (
                        <>
                          <div className="absolute top-3 right-3 z-10 flex gap-1">
                            <button aria-label="Zoom in" title="Zoom in" onClick={(e) => { e.stopPropagation(); zoomIn(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Plus className="w-4 h-4" /></button>
                            <button aria-label="Zoom out" title="Zoom out" onClick={(e) => { e.stopPropagation(); zoomOut(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Minus className="w-4 h-4" /></button>
                            <button aria-label="Reset view" title="Reset view" onClick={(e) => { e.stopPropagation(); resetTransform(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><RotateCcw className="w-4 h-4" /></button>
                            <Dialog>
                              <DialogTrigger asChild>
                                <button aria-label="Open fullscreen" title="Open fullscreen" className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80">
                                  <Maximize2 className="w-4 h-4" />
                                </button>
                              </DialogTrigger>
                              <DialogContent className="w-[min(96vw,1200px)] h-[min(90vh,800px)] p-0 bg-background/95">
                                <div className="relative h-full rounded-lg overflow-hidden">
                                  <TransformWrapper>
                                    {({ zoomIn, zoomOut, resetTransform }) => (
                                      <>
                                        <div className="absolute top-3 right-3 z-10 flex gap-1">
                                          <button aria-label="Zoom in" title="Zoom in" onClick={(e) => { e.stopPropagation(); zoomIn(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Plus className="w-4 h-4" /></button>
                                          <button aria-label="Zoom out" title="Zoom out" onClick={(e) => { e.stopPropagation(); zoomOut(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Minus className="w-4 h-4" /></button>
                                          <button aria-label="Reset view" title="Reset view" onClick={(e) => { e.stopPropagation(); resetTransform(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><RotateCcw className="w-4 h-4" /></button>
                                        </div>
                                        <TransformComponent wrapperClass="h-full" contentClass="h-full">
                                          <div className="h-full">
                                            <ElegantPhysicsVisualization type="pendulum" className="h-full" />
                                          </div>
                                        </TransformComponent>
                                      </>
                                    )}
                                  </TransformWrapper>
                                </div>
                              </DialogContent>
                            </Dialog>
                          </div>
                          <TransformComponent wrapperClass="h-full" contentClass="h-full">
                            <div className="h-full">
                              <ElegantPhysicsVisualization type="pendulum" className="h-full" />
                            </div>
                          </TransformComponent>
                          <div className="absolute inset-x-3 bottom-3 pointer-events-none">
                            <div className="rounded-md bg-background/70 backdrop-blur-sm border border-border/40 px-3 py-2 flex items-center gap-2">
                              <Badge variant="outline" className="badge-outline-quantum text-[10px] px-1.5 py-0">Live</Badge>
                              <div className="min-w-0">
                                <p className="text-sm font-medium leading-tight">Pendulum Dynamics</p>
                                <p className="text-xs text-muted-foreground leading-tight line-clamp-1">Energy exchange and phase space motion</p>
                              </div>
                            </div>
                          </div>
                        </>
                      )}
                    </TransformWrapper>
                  </div>
                </motion.div>
                <motion.div
                  initial={prefersReducedMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
                  viewport={{ once: true, amount: 0.2 }}
                  whileHover={prefersReducedMotion ? undefined : { scale: 1.02 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                >
                  <div className={`group relative rounded-2xl overflow-hidden min-h-[280px] lg:min-h-[320px] card-field-surface`}>
                    <div aria-hidden className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity duration-700 glow-overlay-field" />
                    <TransformWrapper>
                      {({ zoomIn, zoomOut, resetTransform }) => (
                        <>
                          <div className="absolute top-3 right-3 z-10 flex gap-1">
                            <button aria-label="Zoom in" title="Zoom in" onClick={(e) => { e.stopPropagation(); zoomIn(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Plus className="w-4 h-4" /></button>
                            <button aria-label="Zoom out" title="Zoom out" onClick={(e) => { e.stopPropagation(); zoomOut(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Minus className="w-4 h-4" /></button>
                            <button aria-label="Reset view" title="Reset view" onClick={(e) => { e.stopPropagation(); resetTransform(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><RotateCcw className="w-4 h-4" /></button>
                            <Dialog>
                              <DialogTrigger asChild>
                                <button aria-label="Open fullscreen" title="Open fullscreen" className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80">
                                  <Maximize2 className="w-4 h-4" />
                                </button>
                              </DialogTrigger>
                              <DialogContent className="w-[min(96vw,1200px)] h-[min(90vh,800px)] p-0 bg-background/95">
                                <div className="relative h-full rounded-lg overflow-hidden">
                                  <TransformWrapper>
                                    {({ zoomIn, zoomOut, resetTransform }) => (
                                      <>
                                        <div className="absolute top-3 right-3 z-10 flex gap-1">
                                          <button aria-label="Zoom in" onClick={(e) => { e.stopPropagation(); zoomIn(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Plus className="w-4 h-4" /></button>
                                          <button aria-label="Zoom out" onClick={(e) => { e.stopPropagation(); zoomOut(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Minus className="w-4 h-4" /></button>
                                          <button aria-label="Reset view" onClick={(e) => { e.stopPropagation(); resetTransform(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><RotateCcw className="w-4 h-4" /></button>
                                        </div>
                                        <TransformComponent wrapperClass="h-full" contentClass="h-full">
                                          <div className="h-full">
                                            <ElegantPhysicsVisualization type="wave" className="h-full" />
                                          </div>
                                        </TransformComponent>
                                      </>
                                    )}
                                  </TransformWrapper>
                                </div>
                              </DialogContent>
                            </Dialog>
                          </div>
                          <TransformComponent wrapperClass="h-full" contentClass="h-full">
                            <div className="h-full">
                              <ElegantPhysicsVisualization type="wave" className="h-full" />
                            </div>
                          </TransformComponent>
                          <div className="absolute inset-x-3 bottom-3 pointer-events-none">
                            <div className="rounded-md bg-background/70 backdrop-blur-sm border border-border/40 px-3 py-2 flex items-center gap-2">
                              <Badge variant="outline" className="badge-outline-quantum text-[10px] px-1.5 py-0">Live</Badge>
                              <div className="min-w-0">
                                <p className="text-sm font-medium leading-tight">Wave Interference</p>
                                <p className="text-xs text-muted-foreground leading-tight line-clamp-1">Superposition and phase evolution</p>
                              </div>
                            </div>
                          </div>
                        </>
                      )}
                    </TransformWrapper>
                  </div>
                </motion.div>
                <motion.div
                  initial={prefersReducedMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
                  viewport={{ once: true, amount: 0.2 }}
                  whileHover={prefersReducedMotion ? undefined : { scale: 1.02 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                >
                  <div className={`group relative rounded-2xl overflow-hidden min-h-[280px] lg:min-h-[320px] card-field-surface`}>
                    <div aria-hidden className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity duration-700 glow-overlay-field" />
                    <TransformWrapper>
                      {({ zoomIn, zoomOut, resetTransform }) => (
                        <>
                          <div className="absolute top-3 right-3 z-10 flex gap-1">
                            <button aria-label="Zoom in" title="Zoom in" onClick={(e) => { e.stopPropagation(); zoomIn(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Plus className="w-4 h-4" /></button>
                            <button aria-label="Zoom out" title="Zoom out" onClick={(e) => { e.stopPropagation(); zoomOut(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Minus className="w-4 h-4" /></button>
                            <button aria-label="Reset view" title="Reset view" onClick={(e) => { e.stopPropagation(); resetTransform(); }} className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><RotateCcw className="w-4 h-4" /></button>
                            <Dialog>
                              <DialogTrigger asChild>
                                <button aria-label="Open fullscreen" title="Open fullscreen" className="h-8 w-8 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80">
                                  <Maximize2 className="w-4 h-4" />
                                </button>
                              </DialogTrigger>
                              <DialogContent className="w-[min(96vw,1200px)] h-[min(90vh,800px)] p-0 bg-background/95">
                                <div className="relative h-full rounded-lg overflow-hidden">
                                  <TransformWrapper>
                                    {({ zoomIn, zoomOut, resetTransform }) => (
                                      <>
                                        <div className="absolute top-3 right-3 z-10 flex gap-1">
                                          <button aria-label="Zoom in" onClick={(e) => { e.stopPropagation(); zoomIn(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Plus className="w-4 h-4" /></button>
                                          <button aria-label="Zoom out" onClick={(e) => { e.stopPropagation(); zoomOut(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><Minus className="w-4 h-4" /></button>
                                          <button aria-label="Reset view" onClick={(e) => { e.stopPropagation(); resetTransform(); }} className="h-9 w-9 grid place-items-center rounded-md bg-background/60 border border-border/40 text-foreground hover:bg-background/80"><RotateCcw className="w-4 h-4" /></button>
                                        </div>
                                        <TransformComponent wrapperClass="h-full" contentClass="h-full">
                                          <div className="h-full">
                                            <ElegantPhysicsVisualization type="orbit" className="h-full" />
                                          </div>
                                        </TransformComponent>
                                      </>
                                    )}
                                  </TransformWrapper>
                                </div>
                              </DialogContent>
                            </Dialog>
                          </div>
                          <TransformComponent wrapperClass="h-full" contentClass="h-full">
                            <div className="h-full">
                              <ElegantPhysicsVisualization type="orbit" className="h-full" />
                            </div>
                          </TransformComponent>
                          <div className="absolute inset-x-3 bottom-3 pointer-events-none">
                            <div className="rounded-md bg-background/70 backdrop-blur-sm border border-border/40 px-3 py-2 flex items-center gap-2">
                              <Badge variant="outline" className="badge-outline-quantum text-[10px] px-1.5 py-0">Beta</Badge>
                              <div className="min-w-0">
                                <p className="text-sm font-medium leading-tight">Orbital Mechanics</p>
                                <p className="text-xs text-muted-foreground leading-tight line-clamp-1">Keplerian motion and stability</p>
                              </div>
                            </div>
                          </div>
                        </>
                      )}
                    </TransformWrapper>
                  </div>
                </motion.div>
              </div>
            </div>
          </div>

          {/* Platform Capabilities - Enhanced with subtle theming */}
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h3 className="text-[clamp(20px,2.5vw,28px)] font-bold mb-2 flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-accent/10 via-primary/10 to-accent/10 border border-accent/20 backdrop-blur-sm">
                💎 Platform Capabilities
              </h3>
              <p className="text-base text-muted-foreground font-medium">
                Powerful features that make <BrandLogo inline size="sm" withIcon={false} /> extraordinary
              </p>
            </div>
            
            <div className={`grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 ${!prefersReducedMotion ? 'animate-fade-in' : ''}`}>
              {features.map(({ icon: Icon, title, description, tags }, index) => (
                <motion.div
                  key={title}
                  initial={prefersReducedMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
                  viewport={{ once: true, amount: 0.2 }}
                  whileHover={prefersReducedMotion ? undefined : { scale: 1.02 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                >
                  <Card
                    className={`group relative rounded-2xl overflow-hidden min-h-[180px] h-full card-surface-glass card-hover-raise`}
                    aria-label={title}
                  >
                    {/* Subtle glow overlay matching platform theme */}
                    <div
                      aria-hidden
                      className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity duration-700 glow-overlay-quantum"
                    />
                    
                    <div className="relative p-6 h-full flex flex-col">
                      <div className="flex items-start gap-4 flex-1">
                        <div className="shrink-0 icon-chip-quantum">
                          <Icon className="w-7 h-7 text-[hsl(var(--semantic-domain-quantum))] transition-all duration-300" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <h4 className="text-xl font-bold leading-7 mb-3 gradient-text-quantum">
                            {title}
                          </h4>
                          <p className="text-base text-muted-foreground line-clamp-3 mb-4 leading-relaxed">
                            {description}
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {tags.map((tag) => (
                              <Badge
                                key={tag}
                                variant="outline"
                                className="text-xs badge-outline-quantum"
                              >
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
