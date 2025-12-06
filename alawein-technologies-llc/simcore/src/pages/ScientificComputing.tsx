import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { ScientificPlot } from '@/components/ScientificPlot';
import { 
  BookOpen, 
  Calculator, 
  Code, 
  Cpu, 
  GitBranch, 
  Layers, 
  Zap,
  Monitor,
  Brain,
  FileText,
  Play,
  Settings,
  ChevronRight
} from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useSEO } from '@/hooks/use-seo';

const ScientificComputing = () => {
  useSEO({ title: 'Scientific Computing Guide – SimCore', description: 'Tutorials and best practices for computational physics and programming.' });

  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Scientific Computing Guide',
    description: 'Tutorials and best practices for computational physics and programming.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'scientific computing, tutorials, computational physics',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);

  const [activeSection, setActiveSection] = useState('fundamentals');

  const tutorialSections = [
    {
      id: 'fundamentals',
      title: 'Fundamentals',
      icon: BookOpen,
      description: 'Core concepts in computational physics',
      topics: [
        'Numerical Methods',
        'Error Analysis',
        'Algorithm Complexity',
        'Data Structures'
      ]
    },
    {
      id: 'algorithms',
      title: 'Algorithms',
      icon: Calculator,
      description: 'Essential computational algorithms',
      topics: [
        'Finite Difference Methods',
        'Monte Carlo Simulations',
        'Optimization Techniques',
        'Linear Algebra'
      ]
    },
    {
      id: 'implementation',
      title: 'Implementation',
      icon: Code,
      description: 'Best practices for scientific code',
      topics: [
        'Performance Optimization',
        'Memory Management',
        'Testing Strategies',
        'Documentation'
      ]
    },
    {
      id: 'advanced',
      title: 'Advanced Topics',
      icon: Brain,
      description: 'Cutting-edge computational methods',
      topics: [
        'Machine Learning',
        'GPU Computing',
        'Parallel Processing',
        'WebGL/WebGPU'
      ]
    }
  ];

  const codeExamples = {
    fundamentals: {
      title: 'Finite Difference Example',
      description: 'Solving the 1D heat equation using finite differences',
      code: `// 1D Heat Equation: ∂u/∂t = α ∂²u/∂x²
function heatEquation1D(u, alpha, dx, dt, steps) {
  const n = u.length;
  const r = alpha * dt / (dx * dx);
  
  for (let step = 0; step < steps; step++) {
    const u_new = new Array(n).fill(0);
    
    // Interior points
    for (let i = 1; i < n - 1; i++) {
      u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1]);
    }
    
    // Boundary conditions (Dirichlet)
    u_new[0] = 0;
    u_new[n-1] = 0;
    
    u = u_new;
  }
  
  return u;
}`
    },
    algorithms: {
      title: 'Monte Carlo Integration',
      description: 'Estimating π using random sampling',
      code: `// Monte Carlo estimation of π
function estimatePi(samples) {
  let insideCircle = 0;
  
  for (let i = 0; i < samples; i++) {
    const x = Math.random() * 2 - 1; // [-1, 1]
    const y = Math.random() * 2 - 1; // [-1, 1]
    
    if (x*x + y*y <= 1) {
      insideCircle++;
    }
  }
  
  return 4 * insideCircle / samples;
}

// Performance optimization with typed arrays
function estimatePiFast(samples) {
  const coords = new Float64Array(samples * 2);
  crypto.getRandomValues(coords);
  
  let insideCircle = 0;
  for (let i = 0; i < samples; i++) {
    const x = coords[i*2] * 2 - 1;
    const y = coords[i*2+1] * 2 - 1;
    if (x*x + y*y <= 1) insideCircle++;
  }
  
  return 4 * insideCircle / samples;
}`
    }
  };

  const performanceData = [
    {
      x: [1000, 10000, 100000, 1000000],
      y: [0.1, 0.8, 8.2, 82.5],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Standard Implementation',
      line: { color: 'hsl(var(--primary))' }
    },
    {
      x: [1000, 10000, 100000, 1000000],
      y: [0.05, 0.3, 2.8, 28.1],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Optimized Implementation',
      line: { color: 'hsl(var(--accent))' }
    }
  ];

  const plotLayout = {
    title: 'Performance Comparison',
    xaxis: { title: 'Problem Size', type: 'log' },
    yaxis: { title: 'Execution Time (ms)', type: 'log' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'hsl(var(--foreground))' }
  };

return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Scientific Computing Guide"
        description="Comprehensive guide to computational physics and scientific programming"
        difficulty="Intermediate"
        category="Tutorial"
        
      />

      <div className="container mx-auto px-4 py-6">
        {/* Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {tutorialSections.map((section) => {
            const Icon = section.icon;
            const isActive = activeSection === section.id;
            
            return (
              <Card 
                key={section.id}
                className={`cursor-pointer transition-all duration-300 hover:shadow-lg border-2 ${
                  isActive 
                    ? 'border-primary bg-primary/5 shadow-lg' 
                    : 'border-border hover:border-primary/50'
                }`}
                onClick={() => setActiveSection(section.id)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-3 mb-2">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      isActive ? 'bg-primary text-primary-foreground' : 'bg-muted'
                    }`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <CardTitle className="text-lg">{section.title}</CardTitle>
                  </div>
                  <CardDescription className="text-sm">
                    {section.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex flex-wrap gap-1">
                    {section.topics.slice(0, 2).map((topic) => (
                      <Badge key={topic} variant="secondary" className="text-xs">
                        {topic}
                      </Badge>
                    ))}
                    {section.topics.length > 2 && (
                      <Badge variant="outline" className="text-xs">
                        +{section.topics.length - 2} more
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <Tabs value={activeSection} onValueChange={setActiveSection} className="space-y-6">
          {/* Fundamentals */}
          <TabsContent value="fundamentals" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BookOpen className="w-5 h-5" />
                    Numerical Methods
                  </CardTitle>
                  <CardDescription>
                    Core techniques for solving mathematical problems computationally
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <h4 className="font-semibold">Finite Difference Methods</h4>
                    <p className="text-sm text-muted-foreground">
                      Approximate derivatives using discrete differences:
                    </p>
                    <div className="bg-muted p-3 rounded-lg">
                      <BlockMath math="f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}" />
                    </div>
                    <p className="text-sm">
                      Forward, backward, and central difference schemes with different accuracy orders.
                    </p>
                  </div>
                  
                  <div className="space-y-3">
                    <h4 className="font-semibold">Error Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      Understanding and quantifying computational errors:
                    </p>
                    <ul className="text-sm space-y-1 list-disc list-inside ml-4">
                      <li>Truncation errors from approximations</li>
                      <li>Round-off errors from finite precision</li>
                      <li>Stability and convergence analysis</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Code Example</CardTitle>
                  <CardDescription>
                    {codeExamples.fundamentals.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto">
                    <code>{codeExamples.fundamentals.code}</code>
                  </pre>
                  <Button className="mt-4 w-full" variant="outline">
                    <Play className="w-4 h-4 mr-2" />
                    Run Interactive Demo
                  </Button>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Performance Considerations</CardTitle>
                <CardDescription>
                  Understanding computational complexity and optimization
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScientificPlot
                  data={performanceData}
                  title="Performance Comparison"
                  plotType="2d"
                  className="h-64"
                />
                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-primary">O(n)</div>
                    <div className="text-sm text-muted-foreground">Time Complexity</div>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-accent">O(1)</div>
                    <div className="text-sm text-muted-foreground">Space Complexity</div>
                  </div>
                  <div className="text-center p-3 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-primary-glow">2.5x</div>
                    <div className="text-sm text-muted-foreground">Speedup Factor</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Algorithms */}
          <TabsContent value="algorithms" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Calculator className="w-5 h-5" />
                    Monte Carlo Methods
                  </CardTitle>
                  <CardDescription>
                    Stochastic algorithms for numerical integration and simulation
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <h4 className="font-semibold">Basic Principle</h4>
                    <p className="text-sm text-muted-foreground">
                      Use random sampling to solve deterministic problems:
                    </p>
                    <div className="bg-muted p-3 rounded-lg">
                      <BlockMath math="\int_a^b f(x)dx \approx \frac{b-a}{N}\sum_{i=1}^N f(x_i)" />
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <h4 className="font-semibold">Applications</h4>
                    <ul className="text-sm space-y-1 list-disc list-inside ml-4">
                      <li>Numerical integration in high dimensions</li>
                      <li>Statistical physics simulations</li>
                      <li>Financial modeling and risk assessment</li>
                      <li>Quantum Monte Carlo methods</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Implementation</CardTitle>
                  <CardDescription>
                    {codeExamples.algorithms.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto">
                    <code>{codeExamples.algorithms.code}</code>
                  </pre>
                  <div className="mt-4 grid grid-cols-2 gap-2">
                    <Button variant="outline" size="sm">
                      <Play className="w-4 h-4 mr-2" />
                      Run Demo
                    </Button>
                    <Button variant="outline" size="sm">
                      <Settings className="w-4 h-4 mr-2" />
                      Benchmark
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Implementation */}
          <TabsContent value="implementation" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[
                {
                  title: 'Performance Optimization',
                  icon: Zap,
                  points: [
                    'Use typed arrays for numerical data',
                    'Minimize garbage collection',
                    'Vectorize operations when possible',
                    'Profile and benchmark code'
                  ]
                },
                {
                  title: 'Memory Management',
                  icon: Cpu,
                  points: [
                    'Pre-allocate arrays',
                    'Reuse objects and buffers',
                    'Monitor memory usage',
                    'Use WebAssembly for intensive tasks'
                  ]
                },
                {
                  title: 'Testing Strategies',
                  icon: GitBranch,
                  points: [
                    'Unit tests for algorithms',
                    'Numerical accuracy tests',
                    'Performance regression tests',
                    'Integration tests for workflows'
                  ]
                }
              ].map((section) => {
                const Icon = section.icon;
                return (
                  <Card key={section.title}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Icon className="w-5 h-5" />
                        {section.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-2">
                        {section.points.map((point, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm">
                            <ChevronRight className="w-4 h-4 mt-0.5 text-primary flex-shrink-0" />
                            {point}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          {/* Advanced Topics */}
          <TabsContent value="advanced" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Modern Computational Physics
                </CardTitle>
                <CardDescription>
                  Cutting-edge techniques and technologies
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-semibold">Machine Learning Integration</h4>
                    <ul className="text-sm space-y-2 list-disc list-inside ml-4">
                      <li>Physics-informed neural networks (PINNs)</li>
                      <li>Neural differential equations</li>
                      <li>Reinforcement learning for optimization</li>
                      <li>Generative models for sampling</li>
                    </ul>
                  </div>
                  
                  <div className="space-y-4">
                    <h4 className="font-semibold">GPU Computing</h4>
                    <ul className="text-sm space-y-2 list-disc list-inside ml-4">
                      <li>WebGPU compute shaders</li>
                      <li>Parallel reduction algorithms</li>
                      <li>Memory coalescing patterns</li>
                      <li>Mixed precision arithmetic</li>
                    </ul>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-2 pt-4">
                  <Button variant="outline" size="sm">
                    <Monitor className="w-4 h-4 mr-2" />
                    View GPU Examples
                  </Button>
                  <Button variant="outline" size="sm">
                    <Brain className="w-4 h-4 mr-2" />
                    ML Showcase
                  </Button>
                  <Button variant="outline" size="sm">
                    <FileText className="w-4 h-4 mr-2" />
                    Documentation
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PhysicsModuleLayout>
  );
};

export default ScientificComputing;