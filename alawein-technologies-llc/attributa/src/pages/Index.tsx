import React from 'react';
import { Link } from 'react-router-dom';
import { BarChart3, Brain, Shield, BookOpen, FileCode, Target, Copy, Zap, Code, Search, HelpCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useSEO } from '@/hooks/useSEO';
import Typewriter from '@/components/dev/Typewriter';
import LazyBackground from '@/components/dev/LazyBackground';
import MethodCard from '@/components/dev/MethodCard';
import CodeBlock from '@/components/dev/CodeBlock';
import AISecurityIcon from '@/components/dev/AISecurityIcon';
import PixelAccents from '@/components/dev/PixelAccents';
import { MetricPill } from '@/components/ui/metric-pill';
import { PerformanceMonitor } from '@/components/ui/performance-monitor';
import { HoverLift, ScrollReveal } from '@/components/ui/micro-interactions';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { QuickActions } from '@/components/ui/quick-actions';
import { StatsDashboard } from '@/components/ui/stats-dashboard';
import { MethodComparison } from '@/components/ui/method-comparison';

export default function Index() {
  useSEO({
    title: "Attributa.dev — Open‑source AI Attribution",
    description: "Community-built, local-first attribution analysis. No account required. Optional sign in to sync settings.",
    canonical: "https://attributa.dev/"
  });

  // Add JSON-LD structured data
  React.useEffect(() => {
    const script = document.createElement('script');
    script.type = 'application/ld+json';
    script.textContent = JSON.stringify({
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      "name": "Attributa",
      "description": "Open-source AI attribution intelligence for transparent human-AI collaboration",
      "url": "https://attributa.dev",
      "applicationCategory": "DeveloperApplication",
      "operatingSystem": "Web Browser",
      "offers": {
        "@type": "Offer",
        "price": "0",
        "priceCurrency": "USD"
      },
      "creator": {
        "@type": "Organization",
        "name": "Attributa Community"
      }
    });
    document.head.appendChild(script);
    
    return () => {
      document.head.removeChild(script);
    };
  }, []);

  return (
    <TooltipProvider>
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary">
      {/* Skip links for keyboard navigation */}
      <a 
        href="#main-content" 
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-50 bg-primary text-primary-foreground px-4 py-2 rounded-md focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
      >
        Skip to main content
      </a>
      <a 
        href="#methods" 
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-40 z-50 bg-primary text-primary-foreground px-4 py-2 rounded-md focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
      >
        Skip to features
      </a>
      
      <LazyBackground />

      <div className="relative z-10 container mx-auto px-4 py-16 space-y-32">
        {/* Hero Section */}
        <section id="main-content" className="relative grid lg:grid-cols-2 gap-12 items-center min-h-[80vh] engineering-pattern" aria-labelledby="hero-heading">
          <PixelAccents variant="hero" />

          {/* Left Column - Content */}
          <div className="space-y-8 lg:pr-8">
            <div className="flex items-center gap-4 mb-6">
              <AISecurityIcon size={48} className="text-primary" />
              <Badge variant="outline" className="text-xs font-mono border-accent/30 text-accent">
                v2.1.0 • Ethical AI Partnership
              </Badge>
            </div>
            
            <h1 id="hero-heading" className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.1] font-mono">
              Understand{" "}
              <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent bg-[length:200%_100%] animate-gradient-shift">
                AI Attribution
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground leading-relaxed max-w-2xl font-sans">
              <span className="text-primary font-medium">Analyze text for AI patterns.</span> 
              Local processing. No false confidence.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Button 
                asChild 
                variant="hero"
                size="lg" 
                className="text-lg px-8 py-6 min-h-[48px] focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
              >
                <Link to="/scan" aria-label="Try attribution analysis in your browser">Analyze</Link>
              </Button>
              <Button 
                variant="secondary-quiet" 
                size="lg" 
                className="text-lg px-8 py-6 min-h-[48px] focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2"
                onClick={() => {
                  const methodsSection = document.getElementById('methods');
                  if (methodsSection) {
                    methodsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }
                }}
              >
                Learn Methods
              </Button>
            </div>

            <div className="pt-4">
              <Badge variant="outline" className="px-4 py-2 text-sm border-accent/20 text-accent">
                🔓 Open Source • Local Processing • No Account Required
              </Badge>
            </div>
          </div>

          {/* Right Column - Interactive Code Demo */}
          <div className="relative">
            <div className="mb-3 font-mono text-sm text-muted-foreground">
              <Typewriter lines={['$ attributa local-analyze sample.txt', '$ open /documentation']} />
            </div>
            <div className="glass-card p-6 rounded-lg hover-lift">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-3 h-3 rounded-full bg-destructive"></div>
                <div className="w-3 h-3 rounded-full bg-warning"></div>
                <div className="w-3 h-3 rounded-full bg-success"></div>
                <span className="ml-2 text-sm text-muted-foreground font-mono">local-analyzer.ts</span>
              </div>
              <div className="code-block p-4 text-sm font-mono">
                <pre className="whitespace-pre">{`import { analyzeText } from "@/lib/nlp/analyzer"

const { gltr, detectgpt } = await analyzeText("Your content...")

console.log("GLTR tail token share:", gltr.tailTokenShare.toFixed(3))
console.log("DetectGPT curvature:", detectgpt.curvature.toFixed(3))`}</pre>
              </div>
            </div>
            
            {/* Floating attribution visualization */}
            <div className="absolute -top-4 -right-4 w-24 h-24 rounded-full bg-primary/10 animate-pulse"></div>
            <div className="absolute -bottom-4 -left-4 w-16 h-16 rounded-full bg-accent/10 animate-pulse" style={{ animationDelay: '1s' }}></div>
          </div>
        </section>

        
        {/* Methods Section */}
        <section id="methods" className="space-y-16 animate-fade-in" aria-labelledby="methods-heading">
          <div className="text-center space-y-6">
            <Badge variant="outline" className="mb-4 text-xs font-mono border-primary/30 text-primary">
              Statistical Analysis • Known Limitations
            </Badge>
            <h2 id="methods-heading" className="text-4xl md:text-5xl font-bold tracking-tight">
              Detection Methods
            </h2>
            <p className="text-xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
              Four analysis techniques for AI pattern detection. 
              <span className="text-primary font-medium">Results include confidence bounds</span> and method limitations.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="glass-card p-6 rounded-xl hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-6 w-6 text-primary" />
                  <Badge variant="secondary" className="text-xs font-mono">Research Validated</Badge>
                </div>
              </div>
              <h3 className="text-xl font-semibold font-mono mb-2">Statistical Analysis</h3>
              <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                Token probability patterns using GPT-2 likelihood distributions (GLTR method)
              </p>
              <div className="flex gap-2 mb-4">
                <MetricPill label="Confidence" value={0.78} format="percentage" confidence="medium" />
                <MetricPill label="Uncertainty" value={0.15} format="percentage" confidence="low" />
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <div className="p-2 bg-muted/50 rounded text-xs">
                  <strong>Limitations:</strong> Works best with older models • Limited for GPT-4/Claude
                </div>
              </div>
            </div>

            <div className="glass-card p-6 rounded-xl hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Zap className="h-6 w-6 text-accent" />
                  <Badge variant="outline" className="text-xs font-mono border-warning/30 text-warning">Experimental</Badge>
                </div>
              </div>
              <h3 className="text-xl font-semibold font-mono mb-2">Perturbation Analysis</h3>
              <div className="text-muted-foreground mb-4 text-sm leading-relaxed flex items-center gap-1">
                <span>Probability curvature patterns via text perturbations (DetectGPT method)</span>
                <Tooltip>
                  <TooltipTrigger>
                    <HelpCircle className="h-3 w-3 text-muted-foreground/60 hover:text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p>Analyzes how text probability changes when slightly modified to detect AI patterns</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="flex gap-2 mb-4">
                <MetricPill label="Signal" value={0.65} format="percentage" confidence="medium" />
                <MetricPill label="Variance" value={0.23} format="percentage" confidence="low" />
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <div className="p-2 bg-muted/50 rounded text-xs">
                  <strong>Limitations:</strong> High computational cost • Model-dependent performance
                </div>
              </div>
            </div>

            <div className="glass-card p-6 rounded-xl hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Shield className="h-6 w-6 text-info" />
                  <Badge variant="outline" className="text-xs font-mono border-destructive/30 text-destructive">Proof of Concept</Badge>
                </div>
              </div>
              <h3 className="text-xl font-semibold font-mono mb-2">Watermark Signals</h3>
              <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                Cryptographic markers in text (when models support watermarking)
              </p>
              <div className="flex gap-2 mb-4">
                <MetricPill label="Coverage" value={0.05} format="percentage" confidence="low" />
                <MetricPill label="Precision" value={0.99} format="percentage" confidence="high" />
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <div className="p-2 bg-muted/50 rounded text-xs">
                  <strong>Limitations:</strong> Rare in practice • Requires model cooperation
                </div>
              </div>
            </div>

            <div className="glass-card p-6 rounded-xl hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <BookOpen className="h-6 w-6 text-success" />
                  <Badge variant="secondary" className="text-xs font-mono">Production Ready</Badge>
                </div>
              </div>
              <h3 className="text-xl font-semibold font-mono mb-2">Citation Validation</h3>
              <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                Cross-reference verification and source attribution
              </p>
              <div className="flex gap-2 mb-4">
                <MetricPill label="Valid Refs" value={0.876} format="percentage" confidence="medium" />
                <MetricPill label="DOI Match" value={0.934} format="percentage" confidence="high" />
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <p className="text-xs text-muted-foreground">Academic reference validation</p>
              </div>
            </div>

            <div className="glass-card p-6 rounded-xl hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Code className="h-6 w-6 text-destructive" />
                  <Badge variant="outline" className="text-xs font-mono border-info/30 text-info">Research Tool</Badge>
                </div>
              </div>
              <h3 className="text-xl font-semibold font-mono mb-2">Code Security</h3>
              <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                Static analysis for AI-generated code patterns
              </p>
              <div className="flex gap-2 mb-4">
                <MetricPill label="CVE Detect" value={0.823} format="percentage" confidence="medium" />
                <MetricPill label="Patterns" value={47} format="integer" confidence="high" />
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <p className="text-xs text-muted-foreground">Security vulnerability detection</p>
              </div>
            </div>

            <div className="glass-card p-6 rounded-xl hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Search className="h-6 w-6 text-accent" />
                  <Badge variant="secondary" className="text-xs font-mono">Core Feature</Badge>
                </div>
              </div>
              <h3 className="text-xl font-semibold font-mono mb-2">Explainable Results</h3>
              <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                Interpretable confidence intervals and model agreement
              </p>
              <div className="flex gap-2 mb-4">
                <MetricPill label="Agreement" value={0.901} format="percentage" confidence="high" />
                <MetricPill label="CI Width" value={0.034} format="decimal" confidence="medium" />
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                <p className="text-xs text-muted-foreground">Model agreement analysis</p>
              </div>
            </div>
          </div>
        </section>

        {/* Attribution Process */}
        <section className="space-y-8" aria-labelledby="process-heading">
          <div className="text-center space-y-4">
            <h2 id="process-heading" className="text-3xl font-bold tracking-tight">How It Works</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Five-step attribution process with complete transparency
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            {/* Process Steps */}
            <div className="space-y-6">
              {/* Step 1 */}
              <div className="flex items-start gap-4 p-6 rounded-lg border bg-card/50">
                <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-primary font-semibold text-sm">1</span>
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Upload Content</h3>
                  <p className="text-sm text-muted-foreground">Paste text or upload PDF. All processing happens locally.</p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex items-start gap-4 p-6 rounded-lg border bg-card/50">
                <div className="w-8 h-8 bg-accent/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-accent font-semibold text-sm">2</span>
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Segment Text</h3>
                  <p className="text-sm text-muted-foreground">Split document into sentences and paragraphs for analysis.</p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex items-start gap-4 p-6 rounded-lg border bg-card/50 border-primary/20">
                <div className="w-8 h-8 bg-info/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-info font-semibold text-sm">3</span>
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Run Attribution Tests</h3>
                  <p className="text-sm text-muted-foreground mb-3">Apply multiple detection methods in parallel:</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-primary rounded-full"></div>
                      <span>Token likelihood patterns</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-accent rounded-full"></div>
                      <span>Text perturbation analysis</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-success rounded-full"></div>
                      <span>Citation verification</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-warning rounded-full"></div>
                      <span>Watermark detection</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 4 */}
              <div className="flex items-start gap-4 p-6 rounded-lg border bg-card/50">
                <div className="w-8 h-8 bg-success/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-success font-semibold text-sm">4</span>
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Calculate Confidence</h3>
                  <p className="text-sm text-muted-foreground">Generate uncertainty bounds and measure method agreement.</p>
                </div>
              </div>

              {/* Step 5 */}
              <div className="flex items-start gap-4 p-6 rounded-lg border bg-card/50 border-accent/20">
                <div className="w-8 h-8 bg-warning/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-warning font-semibold text-sm">5</span>
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Generate Report</h3>
                  <p className="text-sm text-muted-foreground">Export detailed attribution analysis with confidence intervals.</p>
                </div>
              </div>
            </div>

          </div>
        </section>

        {/* Quick Actions & Stats */}
        <section className="space-y-8">
          <StatsDashboard />
          <div className="grid lg:grid-cols-3 gap-8">
            <div className="lg:col-span-1">
              <QuickActions />
            </div>
            <div className="lg:col-span-2">
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Method Comparison</h3>
                <MethodComparison />
              </div>
            </div>
          </div>
        </section>

      </div>
      
      {/* Performance Monitor */}
      <PerformanceMonitor />
    </main>
    </TooltipProvider>
  );
}