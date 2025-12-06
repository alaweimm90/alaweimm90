import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { 
  Atom, 
  Brain, 
  Code, 
  Globe, 
  Heart, 
  Lightbulb, 
   
  Shield, 
  Smartphone, 
  Zap,
  Github,
  ExternalLink,
  Mail,
  Linkedin,
  GraduationCap,
  Book
} from 'lucide-react';
import { InlineMath } from '@/components/ui/Math';
import { useSEO } from '@/hooks/use-seo';
import BrandLogo from '@/components/ui/BrandLogo';

const AboutPlatform = () => {
  useSEO({ title: 'About SimCore (Simulation Core) – Research‑grade Physics, Math, and Scientific‑ML', description: 'Browser‑native platform for advanced simulation in physics, mathematics, and scientific machine learning.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'AboutPage',
    name: 'About SimCore',
    description: 'Overview of SimCore: interactive physics, math, and ML simulations.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const features = [
    {
      icon: Atom,
      title: "Interactive Scientific Computing",
      description: "Real-time simulations across physics, mathematics, and ML with adjustable parameters"
    },
    {
      icon: Brain,
      title: "Scientific Machine Learning",
      description: "ML models to predict, analyze, and enhance simulations across physics and mathematics"
    },
    {
      icon: Code,
      title: "Open Source",
      description: "Transparent, extensible codebase built with modern web technologies"
    },
    {
      icon: Smartphone,
      title: "Mobile-First",
      description: "Responsive design with touch-optimized controls for all devices"
    },
    {
      icon: Zap,
      title: "High Performance",
      description: "WebGL/WebGPU acceleration with Web Workers for complex calculations"
    },
    {
      icon: Shield,
      title: "Accessible",
      description: "WCAG 2.1 compliant with screen reader support and keyboard navigation"
    }
  ];

  const technologies = [
    { name: "React 18", category: "Frontend", description: "Modern reactive UI framework" },
    { name: "TypeScript", category: "Language", description: "Type-safe JavaScript development" },
    { name: "Tailwind CSS", category: "Styling", description: "Utility-first CSS framework" },
    { name: "WebGL/WebGPU", category: "Graphics", description: "Hardware-accelerated computing" },
    { name: "Web Workers", category: "Performance", description: "Background computation threads" },
    { name: "Plotly.js", category: "Visualization", description: "Interactive scientific plots" },
    { name: "Three.js", category: "3D Graphics", description: "3D visualization and animation" },
    { name: "Vite", category: "Build", description: "Fast development and build tool" }
  ];

  const researchAreas = [
    {
      title: "Condensed Matter Physics",
      topics: ["Graphene band structure", "MoS₂ valley physics", "Ising model", "Crystal visualization"],
      equation: "H = -J\\sum_{\\langle i,j \\rangle} S_i S_j"
    },
    {
      title: "Quantum Mechanics",
      topics: ["Schrödinger equation", "Quantum tunneling", "Bloch sphere dynamics", "Field theory"],
      equation: "i\\hbar\\frac{\\partial\\psi}{\\partial t} = \\hat{H}\\psi"
    },
    {
      title: "Statistical Physics",
      topics: ["Boltzmann distribution", "Brownian motion", "Canonical ensemble", "Phase transitions"],
      equation: "P_i = \\frac{e^{-E_i/k_B T}}{Z}"
    },
    {
      title: "Computational Methods",
      topics: ["PINN solvers", "Monte Carlo", "Finite differences", "ML optimization"],
      equation: "\\nabla \\cdot \\mathcal{L} = \\nabla \\cdot \\frac{\\partial \\mathcal{L}}{\\partial(\\nabla \\phi)} - \\frac{\\partial \\mathcal{L}}{\\partial \\phi}"
    }
  ];

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="About SimCore"
        description="An open-source scientific computing platform for physics, mathematics, and scientific machine learning"
        difficulty="Beginner"
        category="Platform"
        
      />

      <div className="container mx-auto px-4 py-6 space-y-8">
        {/* Mission Statement */}
        <Card className="border-primary/20 bg-gradient-to-br from-background via-primary/5 to-background">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-3">
              <Heart className="w-6 h-6 text-red-500" />
              Our Mission
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-lg leading-relaxed">
              SimCore was born from a simple belief: <strong>scientific computing should be accessible, interactive, and inspiring</strong>. 
              We bridge the gap between theoretical knowledge and hands-on understanding through real-time simulations 
              across physics, mathematics, and scientific machine learning that run directly in your browser.
            </p>
            <blockquote className="border-l-4 border-primary pl-4 italic text-muted-foreground">
              "SimCore refers to the platform's core computational engine—a modular, high-performance system designed to enable 
              research-grade, interactive simulations across physics, mathematics, and scientific machine learning."
            </blockquote>
          </CardContent>
        </Card>

        {/* Platform Features */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Lightbulb className="w-6 h-6 text-yellow-500" />
            Platform Features
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card key={index} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Icon className="w-5 h-5 text-primary" />
                      </div>
                      {feature.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            Technology Stack
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {technologies.map((tech, index) => (
              <Card key={index} className="hover:border-primary/50 transition-colors">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">{tech.name}</CardTitle>
                    <Badge variant="secondary" className="text-xs">{tech.category}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="text-xs text-muted-foreground">{tech.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Research Areas */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <GraduationCap className="w-6 h-6 text-green-500" />
            Research Areas
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {researchAreas.map((area, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <CardTitle className="text-lg">{area.title}</CardTitle>
                  <div className="bg-muted/30 p-2 rounded-md text-muted-foreground/70 text-sm">
                    <InlineMath math={area.equation} />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {area.topics.map((topic, topicIndex) => (
                      <Badge key={topicIndex} variant="outline" className="text-xs">
                        {topic}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Development Philosophy */}
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Book className="w-6 h-6 text-purple-500" />
              Development Philosophy
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-3">
                <h3 className="font-semibold text-lg text-primary">Education First</h3>
                <p className="text-sm text-muted-foreground">
                  Every feature is designed with learning in mind. Complex concepts in physics, mathematics, 
                  and ML become intuitive through interactive visualization and real-time parameter adjustment.
                </p>
              </div>
              
              <div className="space-y-3">
                <h3 className="font-semibold text-lg text-accent">Open Science</h3>
                <p className="text-sm text-muted-foreground">
                  Full transparency in methods, algorithms, and implementations. All code is open-source, 
                  well-documented, and follows scientific computing best practices.
                </p>
              </div>
              
              <div className="space-y-3">
                <h3 className="font-semibold text-lg text-primary-glow">Accessibility</h3>
                <p className="text-sm text-muted-foreground">
                  Education in physics, mathematics, and scientific ML should be available to everyone. 
                  We prioritize accessibility, mobile compatibility, and low-resource requirements.
                </p>
              </div>
            </div>
            
            <div className="border-t pt-6">
              <h3 className="font-semibold text-lg mb-3">Technical Principles</h3>
              <ul className="space-y-2 text-sm list-disc list-inside ml-4">
                <li><strong>Performance:</strong> WebGL/WebGPU acceleration with Web Workers for intensive calculations</li>
                <li><strong>Reliability:</strong> Comprehensive testing, error boundaries, and graceful degradation</li>
                <li><strong>Maintainability:</strong> Clean architecture, TypeScript safety, and extensive documentation</li>
                <li><strong>Extensibility:</strong> Modular design for physics, mathematics, and scientific ML modules</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Get Involved */}
        <Card className="border-primary/30 bg-gradient-to-r from-primary/5 to-accent/5">
          <CardHeader>
            <CardTitle className="text-2xl">Get Involved</CardTitle>
            <CardDescription>
              Join our community of educators, researchers, and developers in physics, mathematics, and scientific ML
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                <Github className="w-6 h-6" />
                <span className="font-medium">Contribute Code</span>
                <span className="text-xs text-muted-foreground">GitHub Repository</span>
              </Button>
              
              <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                <Book className="w-6 h-6" />
                <span className="font-medium">Documentation</span>
                <span className="text-xs text-muted-foreground">Help & Tutorials</span>
              </Button>
              
              <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                <Mail className="w-6 h-6" />
                <span className="font-medium">Contact Us</span>
                <span className="text-xs text-muted-foreground">Feedback & Ideas</span>
              </Button>
              
              <Button variant="outline" className="h-auto p-4 flex flex-col items-center gap-2">
                <ExternalLink className="w-6 h-6" />
                <span className="font-medium">Share</span>
                <span className="text-xs text-muted-foreground">Spread the Word</span>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Footer Attribution */}
        <Card className="bg-muted/50">
          <CardContent className="py-6">
            <div className="text-center space-y-2">
              <p className="text-sm text-muted-foreground">
                Built with ❤️ for the physics, mathematics, and scientific ML community
              </p>
              <p className="text-xs text-muted-foreground flex items-center justify-center gap-2">
                © 2024 <BrandLogo inline size="sm" withIcon={false} /> • Open Source • Educational Use
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </PhysicsModuleLayout>
  );
};

export default AboutPlatform;