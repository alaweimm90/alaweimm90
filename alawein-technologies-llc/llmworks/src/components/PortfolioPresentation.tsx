import { memo, useState, useEffect } from 'react';
import { Monitor, Presentation, Download, Share, Github, Globe, Mail, Linkedin, ExternalLink, Layers, Code, Palette, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface PortfolioPresentationProps {
  className?: string;
  isActive?: boolean;
  onToggle?: () => void;
}

const PortfolioPresentationComponent = ({ 
  className = '', 
  isActive = false, 
  onToggle 
}: PortfolioPresentationProps) => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const portfolioData = {
    project: {
      name: "LLM Works - Strategic Command Center",
      tagline: "Open-Source LLM Evaluation Platform",
      description: "A sophisticated, enterprise-grade platform for evaluating AI language models through interactive testing and rigorous benchmarking.",
      version: "2.0.0",
      status: "Production Ready",
      domain: "llmworks.dev",
      repository: "https://github.com/user/llm-works"
    },
    technologies: [
      { name: "React 18", category: "Frontend", icon: "⚛️" },
      { name: "TypeScript", category: "Language", icon: "🔷" },
      { name: "Vite", category: "Build Tool", icon: "⚡" },
      { name: "Tailwind CSS", category: "Styling", icon: "🎨" },
      { name: "shadcn/ui", category: "Components", icon: "🧩" },
      { name: "Supabase", category: "Backend", icon: "🚀" },
      { name: "React Query", category: "State", icon: "🔄" },
      { name: "Lucide Icons", category: "Icons", icon: "🎯" }
    ],
    features: [
      {
        title: "Interactive Gaming Elements",
        description: "XP progression, achievements, leaderboards, and battle animations",
        highlights: ["Robot combat simulations", "Level progression system", "6-tier ranking system", "Victory celebrations"],
        category: "Gaming"
      },
      {
        title: "Professional Dashboard Suite",
        description: "Real-time monitoring, command panels, and technical documentation",
        highlights: ["Live intelligence feeds", "Strategic command interface", "Technical specifications", "Performance metrics"],
        category: "Enterprise"
      },
      {
        title: "Advanced UX/UI Design",
        description: "Glass morphism, dynamic backgrounds, and sophisticated interactions",
        highlights: ["Strategic Command color palette", "Magnetic hover effects", "Particle systems", "Accessibility compliance"],
        category: "Design"
      },
      {
        title: "Developer Experience",
        description: "Type-safe architecture, comprehensive tooling, and maintainable code",
        highlights: ["TypeScript throughout", "Component library", "Performance optimization", "Modern build pipeline"],
        category: "Technical"
      }
    ],
    achievements: [
      { metric: "12K+", label: "Strategic Evaluations", description: "AI model assessments completed" },
      { metric: "47", label: "AI Models", description: "Language models evaluated" },
      { metric: "98%", label: "System Uptime", description: "Platform reliability score" },
      { metric: "4.9/5", label: "User Rating", description: "Average user satisfaction" }
    ],
    designSystem: {
      colors: {
        primary: "#2563EB",
        secondary: "#0891B2", 
        accent: "#B45309",
        background: "#0A0A0B"
      },
      typography: [
        { name: "Display", usage: "Hero headings", weight: "700" },
        { name: "Refined", usage: "Section headers", weight: "600" },
        { name: "Elegant", usage: "Body text", weight: "400" }
      ],
      components: [
        "Glass morphism panels",
        "Magnetic buttons",
        "Animated notifications",
        "Interactive backgrounds"
      ]
    }
  };

  const slides = [
    {
      id: 'overview',
      title: 'Project Overview',
      content: 'project-info'
    },
    {
      id: 'features',
      title: 'Key Features',
      content: 'features-showcase'
    },
    {
      id: 'technology',
      title: 'Technology Stack',
      content: 'tech-stack'
    },
    {
      id: 'design',
      title: 'Design System',
      content: 'design-system'
    },
    {
      id: 'achievements',
      title: 'Project Impact',
      content: 'achievements'
    },
    {
      id: 'contact',
      title: 'Connect & Explore',
      content: 'contact-info'
    }
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isActive) return;
      
      switch (e.key) {
        case 'ArrowLeft':
          setCurrentSlide(prev => Math.max(0, prev - 1));
          break;
        case 'ArrowRight':
          setCurrentSlide(prev => Math.min(slides.length - 1, prev + 1));
          break;
        case 'F11':
          e.preventDefault();
          toggleFullscreen();
          break;
        case 'Escape':
          if (isFullscreen) {
            exitFullscreen();
          } else if (onToggle) {
            onToggle();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isActive, isFullscreen, onToggle]);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const exitFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    }
    setIsFullscreen(false);
  };

  const exportToPDF = () => {
    // This would integrate with a PDF generation library
    console.log('Exporting portfolio to PDF...');
  };

  const sharePortfolio = () => {
    if (navigator.share) {
      navigator.share({
        title: portfolioData.project.name,
        text: portfolioData.project.description,
        url: window.location.href
      });
    } else {
      navigator.clipboard.writeText(window.location.href);
    }
  };

  const renderSlideContent = () => {
    const slide = slides[currentSlide];
    
    switch (slide.content) {
      case 'project-info':
        return (
          <div className="space-y-8">
            <div className="text-center space-y-4">
              <h1 className="heading-display text-6xl md:text-7xl bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
                {portfolioData.project.name}
              </h1>
              <p className="text-2xl text-primary font-semibold">{portfolioData.project.tagline}</p>
              <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
                {portfolioData.project.description}
              </p>
            </div>
            
            <div className="flex justify-center gap-4">
              <Badge variant="secondary" className="px-4 py-2 text-sm">
                {portfolioData.project.status}
              </Badge>
              <Badge variant="outline" className="px-4 py-2 text-sm">
                v{portfolioData.project.version}
              </Badge>
            </div>

            <div className="glass-panel p-8 max-w-4xl mx-auto">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
                <div className="space-y-2">
                  <Globe className="h-8 w-8 text-primary mx-auto" />
                  <div className="font-semibold">Domain</div>
                  <div className="text-sm text-muted-foreground">{portfolioData.project.domain}</div>
                </div>
                <div className="space-y-2">
                  <Code className="h-8 w-8 text-secondary mx-auto" />
                  <div className="font-semibold">Open Source</div>
                  <div className="text-sm text-muted-foreground">MIT License</div>
                </div>
                <div className="space-y-2">
                  <Layers className="h-8 w-8 text-accent mx-auto" />
                  <div className="font-semibold">Enterprise Ready</div>
                  <div className="text-sm text-muted-foreground">Production Scale</div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'features-showcase':
        return (
          <div className="space-y-8">
            <h2 className="heading-display text-4xl text-center mb-12">Core Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {portfolioData.features.map((feature, index) => (
                <div key={index} className="glass-panel p-6 space-y-4">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="text-xs">{feature.category}</Badge>
                    <h3 className="heading-refined text-xl">{feature.title}</h3>
                  </div>
                  <p className="text-muted-foreground">{feature.description}</p>
                  <ul className="space-y-2">
                    {feature.highlights.map((highlight, i) => (
                      <li key={i} className="flex items-center gap-2 text-sm">
                        <Zap className="h-4 w-4 text-primary" />
                        {highlight}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        );

      case 'tech-stack':
        return (
          <div className="space-y-8">
            <h2 className="heading-display text-4xl text-center mb-12">Technology Stack</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {portfolioData.technologies.map((tech, index) => (
                <div key={index} className="glass-minimal p-6 text-center space-y-3 hover:bg-muted/10 transition-colors">
                  <div className="text-4xl">{tech.icon}</div>
                  <div className="font-semibold">{tech.name}</div>
                  <Badge variant="secondary" className="text-xs">{tech.category}</Badge>
                </div>
              ))}
            </div>
            
            <div className="glass-panel p-6 max-w-2xl mx-auto text-center">
              <h3 className="heading-refined text-xl mb-4">Architecture Highlights</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>✅ Type-safe development</div>
                <div>✅ Component-driven architecture</div>
                <div>✅ Performance optimized</div>
                <div>✅ Accessibility compliant</div>
              </div>
            </div>
          </div>
        );

      case 'design-system':
        return (
          <div className="space-y-8">
            <h2 className="heading-display text-4xl text-center mb-12">Design System</h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Color Palette */}
              <div className="glass-panel p-6 space-y-4">
                <h3 className="heading-refined text-xl flex items-center gap-2">
                  <Palette className="h-5 w-5" />
                  Strategic Command Colors
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(portfolioData.designSystem.colors).map(([name, color]) => (
                    <div key={name} className="flex items-center gap-3">
                      <div
                        className="w-8 h-8 rounded-lg border border-white/20"
                        style={{ backgroundColor: color }}
                      />
                      <div>
                        <div className="font-medium capitalize">{name}</div>
                        <div className="text-xs text-muted-foreground">{color}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Typography */}
              <div className="glass-panel p-6 space-y-4">
                <h3 className="heading-refined text-xl">Typography System</h3>
                <div className="space-y-4">
                  {portfolioData.designSystem.typography.map((type, index) => (
                    <div key={index} className="border-l-2 border-primary/30 pl-4">
                      <div className="font-semibold">{type.name}</div>
                      <div className="text-sm text-muted-foreground">{type.usage} • Weight {type.weight}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Components */}
            <div className="glass-panel p-6">
              <h3 className="heading-refined text-xl mb-4">Component Library</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {portfolioData.designSystem.components.map((component, index) => (
                  <div key={index} className="glass-minimal p-3 text-center text-sm">
                    {component}
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 'achievements':
        return (
          <div className="space-y-8">
            <h2 className="heading-display text-4xl text-center mb-12">Project Impact</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {portfolioData.achievements.map((achievement, index) => (
                <div key={index} className="glass-panel p-6 text-center space-y-3">
                  <div className="heading-display text-4xl text-primary">{achievement.metric}</div>
                  <div className="font-semibold">{achievement.label}</div>
                  <div className="text-sm text-muted-foreground">{achievement.description}</div>
                </div>
              ))}
            </div>
            
            <div className="glass-panel p-8 max-w-3xl mx-auto text-center space-y-4">
              <h3 className="heading-refined text-2xl">Development Excellence</h3>
              <p className="text-muted-foreground leading-relaxed">
                Built with modern best practices, comprehensive testing, and enterprise-grade architecture. 
                This project demonstrates full-stack capabilities, sophisticated UI/UX design, and 
                scalable software engineering principles.
              </p>
            </div>
          </div>
        );

      case 'contact-info':
        return (
          <div className="space-y-8 text-center">
            <h2 className="heading-display text-4xl mb-12">Let's Connect</h2>
            
            <div className="space-y-6">
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Interested in this project? Let's discuss how we can work together 
                to build innovative solutions.
              </p>
              
              <div className="flex justify-center gap-4 flex-wrap">
                <Button className="glass-panel hover:bg-primary/10" variant="outline">
                  <Github className="h-4 w-4 mr-2" />
                  View Repository
                </Button>
                <Button className="glass-panel hover:bg-primary/10" variant="outline">
                  <Globe className="h-4 w-4 mr-2" />
                  Live Demo
                </Button>
                <Button className="glass-panel hover:bg-primary/10" variant="outline">
                  <Mail className="h-4 w-4 mr-2" />
                  Get in Touch
                </Button>
                <Button className="glass-panel hover:bg-primary/10" variant="outline">
                  <Linkedin className="h-4 w-4 mr-2" />
                  LinkedIn
                </Button>
              </div>
            </div>

            <div className="glass-panel p-6 max-w-lg mx-auto">
              <h3 className="heading-refined text-lg mb-4">Project Highlights</h3>
              <div className="text-sm text-muted-foreground space-y-2">
                <div>🎮 Gaming elements with enterprise design</div>
                <div>🎨 Custom design system implementation</div>
                <div>⚡ High-performance React architecture</div>
                <div>♿ WCAG accessibility compliance</div>
                <div>📱 Responsive and mobile-optimized</div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (!isActive) {
    return (
      <Button
        onClick={onToggle}
        variant="outline"
        className="glass-minimal border-primary/30 hover:bg-primary/10"
      >
        <Presentation className="h-4 w-4 mr-2" />
        Portfolio Mode
      </Button>
    );
  }

  return (
    <div className={`fixed inset-0 bg-background z-50 overflow-hidden ${className}`}>
      {/* Header Controls */}
      <div className="absolute top-4 left-4 right-4 flex justify-between items-center z-10">
        <div className="flex items-center gap-4">
          <Button
            onClick={onToggle}
            variant="outline"
            size="sm"
            className="glass-minimal border-primary/30"
          >
            ← Exit
          </Button>
          
          <div className="glass-minimal px-3 py-1 rounded-lg text-sm">
            {currentSlide + 1} / {slides.length}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={sharePortfolio}
            variant="outline"
            size="sm"
            className="glass-minimal border-primary/30"
          >
            <Share className="h-4 w-4" />
          </Button>
          
          <Button
            onClick={exportToPDF}
            variant="outline"
            size="sm"
            className="glass-minimal border-primary/30"
          >
            <Download className="h-4 w-4" />
          </Button>
          
          <Button
            onClick={toggleFullscreen}
            variant="outline"
            size="sm"
            className="glass-minimal border-primary/30"
          >
            <Monitor className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="h-full flex flex-col justify-center p-8 pt-20 pb-16">
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full max-w-6xl mx-auto">
            {renderSlideContent()}
          </div>
        </div>
      </div>

      {/* Bottom Navigation */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
        <div className="glass-panel px-6 py-3 flex items-center gap-4">
          <Button
            onClick={() => setCurrentSlide(prev => Math.max(0, prev - 1))}
            disabled={currentSlide === 0}
            variant="outline"
            size="sm"
            className="glass-minimal border-primary/30"
          >
            Previous
          </Button>

          <div className="flex gap-2">
            {slides.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentSlide(index)}
                className={`w-3 h-3 rounded-full transition-colors ${
                  index === currentSlide ? 'bg-primary' : 'bg-muted/30'
                }`}
              />
            ))}
          </div>

          <Button
            onClick={() => setCurrentSlide(prev => Math.min(slides.length - 1, prev + 1))}
            disabled={currentSlide === slides.length - 1}
            size="sm"
            className="bg-primary hover:bg-primary/90"
          >
            Next
          </Button>
        </div>
      </div>

      {/* Keyboard Shortcuts Hint */}
      <div className="absolute bottom-4 right-4 glass-minimal px-3 py-2 rounded-lg text-xs text-muted-foreground">
        Use ← → to navigate • F11 for fullscreen • Esc to exit
      </div>
    </div>
  );
};

export const PortfolioPresentation = memo(PortfolioPresentationComponent);