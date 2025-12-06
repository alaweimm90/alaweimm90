/**
 * Aegis AI Style Specification & Interactive Demo
 * Transforms generic template into evaluation-focused platform
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Shield, Zap, TrendingUp, Eye, Activity, Users, Code, BarChart3 } from 'lucide-react';

// Aegis Domain Tokens
const aegisTokens = {
  // Domain-specific color palette
  trust: 'hsl(160, 84%, 39%)', // Emerald for verified/passing
  risk: 'hsl(43, 96%, 56%)',   // Amber for warning/attention  
  fail: 'hsl(0, 84%, 60%)',    // Rose for failures/critical
  neutral: 'hsl(215, 16%, 47%)', // Slate for baseline
  
  // Evaluation-specific gradients
  trustGlow: 'linear-gradient(135deg, hsl(160, 84%, 39%) 0%, hsl(160, 84%, 49%) 100%)',
  shieldPattern: 'radial-gradient(circle at 30% 70%, hsl(160, 84%, 39%, 0.1) 0%, transparent 50%)',
  
  // Typography hierarchy
  displayFont: '"Inter", system-ui, sans-serif',
  monoFont: '"JetBrains Mono", "Fira Code", monospace',
  
  // Shadows with domain colors
  trustShadow: '0 4px 20px hsl(160, 84%, 39%, 0.25)',
  elevatedShadow: '0 8px 32px hsl(215, 16%, 47%, 0.15)',
};

// Mock data for interactive demo
const mockBenchmarkData = [
  { model: 'GPT-4', score: 94.2, trend: '+2.1%', status: 'trust' },
  { model: 'Claude-3', score: 91.8, trend: '+0.8%', status: 'trust' },
  { model: 'Gemini Pro', score: 87.3, trend: '-1.2%', status: 'risk' },
  { model: 'LLaMA-2', score: 82.1, trend: '-3.4%', status: 'fail' },
];

// Component variants following the spec
const EvaluationCard = ({ title, metric, status, icon: Icon, trend }: any) => (
  <div 
    className="group relative overflow-hidden rounded-xl border border-border/50 bg-card/60 p-4 backdrop-blur-sm transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
    style={{
      boxShadow: status === 'trust' ? aegisTokens.trustShadow : aegisTokens.elevatedShadow
    }}
  >
    <div className="flex items-center justify-between mb-2">
      <Icon 
        className="h-5 w-5" 
        style={{ color: aegisTokens[status as keyof typeof aegisTokens] }}
      />
      <Badge 
        variant="secondary" 
        className="text-xs"
        style={{ 
          backgroundColor: `${aegisTokens[status as keyof typeof aegisTokens]}20`,
          color: aegisTokens[status as keyof typeof aegisTokens]
        }}
      >
        {trend}
      </Badge>
    </div>
    <div className="space-y-1">
      <p className="text-sm font-medium text-muted-foreground">{title}</p>
      <p className="text-2xl font-bold" style={{ fontFamily: aegisTokens.monoFont }}>
        {metric}
      </p>
    </div>
  </div>
);

// Hero Mock Component
const AegisHeroMock = () => {
  const [activeTab, setActiveTab] = useState('arena');
  
  return (
    <div className="relative min-h-screen bg-gradient-to-br from-background via-background to-muted/20 overflow-hidden">
      {/* Background pattern */}
      <div 
        className="absolute inset-0 opacity-30"
        style={{ 
          backgroundImage: aegisTokens.shieldPattern,
          backgroundSize: '400px 400px'
        }}
      />
      
      {/* Header */}
      <header className="relative z-10 flex items-center justify-between p-6 border-b border-border/50 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Shield className="h-8 w-8" style={{ color: aegisTokens.trust }} />
          <span className="text-xl font-bold" style={{ fontFamily: aegisTokens.displayFont }}>
            Aegis AI
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant="secondary" className="text-xs">
            Local-only • No data transmission
          </Badge>
          <Button size="sm" style={{ background: aegisTokens.trustGlow }}>
            Get Started
          </Button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 grid lg:grid-cols-2 gap-12 p-6 lg:p-12 max-w-7xl mx-auto">
        {/* Left: Copy */}
        <div className="space-y-6 lg:pt-12">
          <div className="space-y-4">
            <Badge 
              variant="secondary" 
              className="text-xs px-3 py-1"
              style={{ 
                backgroundColor: `${aegisTokens.trust}20`,
                color: aegisTokens.trust
              }}
            >
              🛡️ The Shield of AI Evaluation
            </Badge>
            <h1 
              className="text-4xl lg:text-6xl font-bold leading-tight"
              style={{ fontFamily: aegisTokens.displayFont }}
            >
              Evaluate AI models
              <br />
              <span style={{ color: aegisTokens.trust }}>with confidence</span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-lg leading-relaxed">
              Unite interactive testing and rigorous benchmarking to understand, compare, and trust model behavior through verifiable evaluation.
            </p>
          </div>
          
          <div className="flex gap-3">
            <Button 
              size="lg" 
              className="text-white"
              style={{ background: aegisTokens.trustGlow }}
            >
              <Zap className="h-4 w-4 mr-2" />
              Open The Arena
            </Button>
            <Button variant="outline" size="lg">
              <BarChart3 className="h-4 w-4 mr-2" />
              Run Benchmarks
            </Button>
          </div>
        </div>

        {/* Right: Interactive Panel */}
        <div className="space-y-6">
          <Card className="border-border/50 bg-card/60 backdrop-blur-sm">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">Live Benchmark Panel</CardTitle>
                <Badge variant="secondary" className="text-xs">
                  <Activity className="h-3 w-3 mr-1" />
                  Running
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {mockBenchmarkData.map((item, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: aegisTokens[item.status as keyof typeof aegisTokens] }}
                    />
                    <span className="font-medium text-sm">{item.model}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-lg font-bold">{item.score}</div>
                    <div 
                      className="text-xs font-medium"
                      style={{ color: aegisTokens[item.status as keyof typeof aegisTokens] }}
                    >
                      {item.trend}
                    </div>
                  </div>
                </div>
              ))}
              
              <div className="grid grid-cols-3 gap-3 pt-2">
                <EvaluationCard 
                  title="Confidence"
                  metric="94.2%"
                  status="trust"
                  icon={Shield}
                  trend="+2.1%"
                />
                <EvaluationCard 
                  title="Reliability"
                  metric="87.8%"
                  status="trust"
                  icon={TrendingUp}
                  trend="+0.9%"
                />
                <EvaluationCard 
                  title="Bias Score"
                  metric="Low"
                  status="trust"
                  icon={Eye}
                  trend="Stable"
                />
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

// Style Specification Document
const StyleSpecDocument = () => (
  <div className="max-w-4xl mx-auto p-6 space-y-8">
    <div className="space-y-4">
      <h1 className="text-3xl font-bold">Aegis AI Style Specification</h1>
      <p className="text-muted-foreground">
        Transform from generic template to evaluation-focused platform with domain-specific design language.
      </p>
    </div>

    <Tabs defaultValue="tokens" className="space-y-6">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="tokens">Design Tokens</TabsTrigger>
        <TabsTrigger value="components">Components</TabsTrigger>
        <TabsTrigger value="layout">Layout</TabsTrigger>
        <TabsTrigger value="motion">Motion</TabsTrigger>
      </TabsList>
      
      <TabsContent value="tokens" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Color System</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(aegisTokens).slice(0, 4).map(([name, value]) => (
                <div key={name} className="space-y-2">
                  <div 
                    className="h-16 rounded-lg border"
                    style={{ backgroundColor: value }}
                  />
                  <div>
                    <p className="font-medium text-sm">{name}</p>
                    <p className="text-xs text-muted-foreground font-mono">{value}</p>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="space-y-2">
              <h4 className="font-semibold">Usage Guidelines</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li><strong>Trust (Emerald):</strong> Passing tests, verified results, high confidence</li>
                <li><strong>Risk (Amber):</strong> Warnings, attention needed, moderate confidence</li>
                <li><strong>Fail (Rose):</strong> Failed tests, critical issues, low confidence</li>
                <li><strong>Neutral (Slate):</strong> Baseline, inactive states, secondary info</li>
              </ul>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Typography System</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div>
                <p className="text-sm text-muted-foreground">Display Font</p>
                <p className="text-2xl font-bold" style={{ fontFamily: aegisTokens.displayFont }}>
                  Inter - Headers & UI
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Monospace Font</p>
                <p className="text-lg" style={{ fontFamily: aegisTokens.monoFont }}>
                  JetBrains Mono - Metrics & Code
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="components" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Component Variants</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <h4 className="font-semibold">Evaluation Cards</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <EvaluationCard 
                  title="Pass Rate"
                  metric="94.2%"
                  status="trust"
                  icon={Shield}
                  trend="+2.1%"
                />
                <EvaluationCard 
                  title="Warning"
                  metric="12"
                  status="risk"
                  icon={Users}
                  trend="±0%"
                />
                <EvaluationCard 
                  title="Critical"
                  metric="3"
                  status="fail"
                  icon={Code}
                  trend="-1"
                />
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold">Badge System</h4>
              <div className="flex gap-2 flex-wrap">
                <Badge style={{ backgroundColor: `${aegisTokens.trust}20`, color: aegisTokens.trust }}>
                  Verified
                </Badge>
                <Badge style={{ backgroundColor: `${aegisTokens.risk}20`, color: aegisTokens.risk }}>
                  Attention
                </Badge>
                <Badge style={{ backgroundColor: `${aegisTokens.fail}20`, color: aegisTokens.fail }}>
                  Critical
                </Badge>
                <Badge variant="secondary">Local-only</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="layout" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Hero Layout Principles</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <h4 className="font-semibold">Asymmetric Grid</h4>
              <ul className="text-sm space-y-1">
                <li>• Left: Strong verbal hierarchy (badge → H1 → description → CTAs)</li>
                <li>• Right: Live benchmark panel with real artifacts</li>
                <li>• 60/40 split on desktop, stacked on mobile</li>
              </ul>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold">Background Treatment</h4>
              <ul className="text-sm space-y-1">
                <li>• Subtle radial gradient with shield motif</li>
                <li>• Low-opacity dot matrix suggesting measurement</li>
                <li>• Backdrop blur on elevated elements</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="motion" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Micro-interactions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <h4 className="font-semibold">Hover States</h4>
              <ul className="text-sm space-y-1">
                <li>• Cards: lift + shadow increase + border glow</li>
                <li>• Buttons: gradient shift + slight scale</li>
                <li>• Metrics: pulse effect on status change</li>
              </ul>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold">Loading States</h4>
              <ul className="text-sm space-y-1">
                <li>• Progress bars with domain colors</li>
                <li>• Skeleton loading for benchmark data</li>
                <li>• Breathing effect for live indicators</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  </div>
);

export default function AegisStyleSpec() {
  const [view, setView] = useState<'spec' | 'demo'>('demo');
  
  return (
    <div className="min-h-screen">
      <div className="fixed top-4 right-4 z-50 flex gap-2">
        <Button 
          variant={view === 'demo' ? 'default' : 'outline'} 
          size="sm"
          onClick={() => setView('demo')}
        >
          Hero Demo
        </Button>
        <Button 
          variant={view === 'spec' ? 'default' : 'outline'} 
          size="sm"
          onClick={() => setView('spec')}
        >
          Style Spec
        </Button>
      </div>
      
      {view === 'demo' ? <AegisHeroMock /> : <StyleSpecDocument />}
    </div>
  );
}