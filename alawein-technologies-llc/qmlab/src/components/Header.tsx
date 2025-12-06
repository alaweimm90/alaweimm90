import React from 'react';
import { Button } from '@/components/ui/button';
import { Atom, Waves, Zap, BookOpen, Brain, Calculator, Mail, Github, Linkedin, ExternalLink, GraduationCap, HelpCircle, Search } from 'lucide-react';
import { useTutorial } from '@/contexts/TutorialContext';
import { QuantumSearch } from '@/components/QuantumSearch';

// Quantum Logo Component with unified branding
const QuantumLogo: React.FC = () => (
  <div className="flex items-center gap-4">
    {/* Quantum Symbol */}
    <div className="relative">
      <div className="w-12 h-12 rounded-full border-2 border-blue-400 flex items-center justify-center bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-indigo-600/20 backdrop-blur-sm">
        <div className="relative">
          {/* Electron orbits */}
          <div className="absolute inset-0 w-8 h-8 border border-blue-300/50 rounded-full animate-spin" style={{ animationDuration: '4s' }}></div>
          <div className="absolute inset-1 w-6 h-6 border border-purple-300/50 rounded-full animate-spin" style={{ animationDuration: '3s', animationDirection: 'reverse' }}></div>
          {/* Nucleus */}
          <div className="w-2 h-2 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full mx-auto mt-3 animate-pulse"></div>
        </div>
      </div>
      {/* Quantum glow effect */}
      <div className="absolute inset-0 w-12 h-12 bg-gradient-to-br from-blue-400/20 to-purple-500/20 rounded-full animate-pulse blur-sm -z-10"></div>
    </div>
    
    {/* Brand Text */}
    <div className="flex flex-col">
      <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-indigo-600 bg-clip-text text-transparent leading-tight">
        QMLab
      </h1>
      <div className="text-xs text-blue-300/80 font-medium tracking-wide">
        Quantum Machine Learning
      </div>
    </div>
  </div>
);

// Contact Information Component
const ContactInfo: React.FC = () => {
  const contactLinks = [
    {
      label: 'Email',
      url: 'mailto:meshal@berkeley.edu',
      icon: <Mail className="w-4 h-4" />,
      color: 'text-blue-400 hover:text-blue-300'
    },
    {
      label: 'LinkedIn',
      url: 'https://www.linkedin.com/in/meshal-alawein',
      icon: <Linkedin className="w-4 h-4" />,
      color: 'text-blue-500 hover:text-blue-400'
    },
    {
      label: 'GitHub',
      url: 'https://github.com/alaweimm90',
      icon: <Github className="w-4 h-4" />,
      color: 'text-slate-400 hover:text-slate-300'
    },
    {
      label: 'Portfolio',
      url: 'https://www.malawein.com',
      icon: <ExternalLink className="w-4 h-4" />,
      color: 'text-purple-400 hover:text-purple-300'
    },
    {
      label: 'Google Scholar',
      url: 'https://scholar.google.com/citations?user=IB_E6GQAAAAJ',
      icon: <GraduationCap className="w-4 h-4" />,
      color: 'text-green-400 hover:text-green-300'
    }
  ];

  return (
    <div className="hidden md:flex items-center gap-3">
      <span className="text-sm text-slate-400 font-medium">Contact:</span>
      <div className="flex items-center gap-2">
        {contactLinks.map((link) => (
          <a
            key={link.label}
            href={link.url}
            target="_blank"
            rel="noopener noreferrer"
            className={`p-2 rounded-lg transition-all duration-200 ${link.color} hover:bg-slate-800/50 hover:scale-105`}
            title={link.label}
            aria-label={`Contact via ${link.label}`}
          >
            {link.icon}
          </a>
        ))}
      </div>
    </div>
  );
};

// Educational Navigation Component
const QuantumNavigation: React.FC = () => {
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const navItems = [
    {
      id: 'lab-section',
      label: 'Interactive Laboratory',
      icon: <Calculator className="w-4 h-4" />,
      description: 'Interactive quantum computing tools'
    },
    {
      id: 'learning-resources', 
      label: 'Learning Resources',
      icon: <BookOpen className="w-4 h-4" />,
      description: 'Educational materials and concepts'
    },
    {
      id: 'advanced-features',
      label: 'Advanced Features',
      icon: <Brain className="w-4 h-4" />,
      description: 'Coming soon features and tools'
    }
  ];

  return (
    <nav className="hidden lg:flex items-center gap-6" role="navigation" aria-label="Quantum learning modules">
      {navItems.map((item) => (
        <button
          key={item.id}
          onClick={() => scrollToSection(item.id)}
          className="group flex items-center gap-2 px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/50 transition-all duration-300 border border-transparent hover:border-blue-500/30"
          aria-label={`Navigate to ${item.description}`}
        >
          <div className="text-blue-400 group-hover:text-blue-300 transition-colors">
            {item.icon}
          </div>
          <span className="text-sm font-medium">{item.label}</span>
        </button>
      ))}
    </nav>
  );
};

// Mobile Menu Component
const MobileMenu: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="lg:hidden absolute top-full left-0 right-0 bg-slate-900/95 backdrop-blur-md border-t border-slate-700/50 z-50">
      <div className="p-6 space-y-4">
        <button
          onClick={() => scrollToSection('lab-section')}
          className="flex items-center gap-3 w-full p-3 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/50 transition-all text-left"
        >
          <Calculator className="w-5 h-5 text-blue-400" />
          <div>
            <div className="font-medium">Interactive Laboratory</div>
            <div className="text-xs text-slate-400">Interactive quantum computing tools</div>
          </div>
        </button>
        
        <button
          onClick={() => scrollToSection('learning-resources')}
          className="flex items-center gap-3 w-full p-3 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/50 transition-all text-left"
        >
          <BookOpen className="w-5 h-5 text-purple-400" />
          <div>
            <div className="font-medium">Learning Resources</div>
            <div className="text-xs text-slate-400">Educational materials and concepts</div>
          </div>
        </button>
        
        <button
          onClick={() => scrollToSection('advanced-features')}
          className="flex items-center gap-3 w-full p-3 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/50 transition-all text-left"
        >
          <Brain className="w-5 h-5 text-indigo-400" />
          <div>
            <div className="font-medium">Advanced Features</div>
            <div className="text-xs text-slate-400">Coming soon features and tools</div>
          </div>
        </button>
        
        {/* Tutorial Button for Mobile */}
        <MobileTutorialButton onClose={onClose} />
        
        {/* Contact info for mobile */}
        <div className="pt-4 border-t border-slate-700/50">
          <div className="text-xs text-slate-400 font-medium mb-3">Contact:</div>
          <div className="grid grid-cols-3 gap-2">
            <a href="mailto:meshal@berkeley.edu" className="flex items-center gap-2 p-2 rounded-lg text-blue-400 hover:bg-slate-800/50 transition-colors text-xs">
              <Mail className="w-3 h-3" />
              Email
            </a>
            <a href="https://www.linkedin.com/in/meshal-alawein" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 p-2 rounded-lg text-blue-500 hover:bg-slate-800/50 transition-colors text-xs">
              <Linkedin className="w-3 h-3" />
              LinkedIn
            </a>
            <a href="https://github.com/alaweimm90" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 p-2 rounded-lg text-slate-400 hover:bg-slate-800/50 transition-colors text-xs">
              <Github className="w-3 h-3" />
              GitHub
            </a>
            <a href="https://www.malawein.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 p-2 rounded-lg text-purple-400 hover:bg-slate-800/50 transition-colors text-xs">
              <ExternalLink className="w-3 h-3" />
              Portfolio
            </a>
            <a href="https://scholar.google.com/citations?user=IB_E6GQAAAAJ" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 p-2 rounded-lg text-green-400 hover:bg-slate-800/50 transition-colors text-xs">
              <GraduationCap className="w-3 h-3" />
              Scholar
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

// Mobile Tutorial Button Component
const MobileTutorialButton: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const { startTutorial, hasSeenTutorial } = useTutorial();

  const defaultSteps = [
    {
      id: 'circuit-builder',
      title: 'Build Quantum Circuits',
      content: 'Design quantum circuits by selecting gates and adding them to qubits. Each gate transforms the quantum state in unique ways.',
      target: '[data-tutorial="circuit-builder"]',
      position: 'right' as const,
      action: 'Try Adding a Gate'
    },
    {
      id: 'bloch-sphere',
      title: 'Visualize Quantum States',
      content: 'Watch how quantum states evolve on the Bloch sphere. This 3D visualization shows the quantum state of your qubits in real-time.',
      target: '[data-tutorial="bloch-sphere"]',
      position: 'left' as const,
      action: 'Explore the Visualization'
    }
  ];

  const handleStartTutorial = () => {
    startTutorial(defaultSteps);
    onClose();
  };

  return (
    <button
      onClick={handleStartTutorial}
      className="flex items-center gap-3 w-full p-3 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/50 transition-all text-left"
    >
      <HelpCircle className="w-5 h-5 text-yellow-400" />
      <div>
        <div className="font-medium flex items-center gap-2">
          Interactive Tour
          {!hasSeenTutorial && (
            <span className="px-1.5 py-0.5 text-xs bg-blue-500 text-white rounded-full animate-pulse">
              New
            </span>
          )}
        </div>
        <div className="text-xs text-slate-400">Take a guided tour of QMLab</div>
      </div>
    </button>
  );
};

// Search Button Component
const SearchButton: React.FC<{ onClick: () => void }> = ({ onClick }) => {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onClick}
      className="flex items-center gap-2 text-slate-300 hover:text-white hover:bg-slate-800/50"
      aria-label="Search quantum concepts and tutorials"
    >
      <Search className="h-4 w-4" />
      <span className="hidden sm:inline">Search</span>
      <kbd className="hidden md:inline-flex items-center gap-1 rounded border bg-slate-800 px-1.5 py-0.5 text-xs text-slate-400 border-slate-600">
        <span className="text-xs">⌘</span>K
      </kbd>
    </Button>
  );
};

// Tutorial Button Component
const TutorialButton: React.FC = () => {
  const { startTutorial, hasSeenTutorial, showTourPreference, setShowTourPreference } = useTutorial();

  const defaultSteps = [
    {
      id: 'circuit-builder',
      title: 'Build Quantum Circuits',
      content: 'Design quantum circuits by selecting gates and adding them to qubits. Each gate transforms the quantum state in unique ways.',
      target: '[data-tutorial="circuit-builder"]',
      position: 'right' as const,
      action: 'Try Adding a Gate'
    },
    {
      id: 'bloch-sphere',
      title: 'Visualize Quantum States',
      content: 'Watch how quantum states evolve on the Bloch sphere. This 3D visualization shows the quantum state of your qubits in real-time.',
      target: '[data-tutorial="bloch-sphere"]',
      position: 'left' as const,
      action: 'Explore the Visualization'
    }
  ];

  const handleStartTutorial = () => {
    startTutorial(defaultSteps);
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={handleStartTutorial}
      className="flex items-center gap-2 text-slate-300 hover:text-white hover:bg-slate-800/50"
      aria-label={hasSeenTutorial ? "Replay tutorial" : "Start tutorial"}
    >
      <HelpCircle className="h-4 w-4" />
      <span className="hidden sm:inline">Tour</span>
      {!hasSeenTutorial && (
        <span className="ml-1 px-1.5 py-0.5 text-xs bg-blue-500 text-white rounded-full animate-pulse">
          New
        </span>
      )}
    </Button>
  );
};

// Skip Navigation Link (Accessibility)
const SkipToContent: React.FC = () => (
  <a
    href="#main-content"
    className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-600 text-white px-4 py-2 rounded-md font-medium z-50 transition-all"
    tabIndex={0}
  >
    Skip to main content
  </a>
);

// Main Header Component
export const Header: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);
  const [searchOpen, setSearchOpen] = React.useState(false);

  // Keyboard shortcut for search (Cmd+K / Ctrl+K)
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      <SkipToContent />
      <header className="fixed top-0 left-0 right-0 z-40 bg-slate-900/90 backdrop-blur-lg border-b border-slate-700/50">
        {/* Quantum particle effects */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-2 left-1/4 w-1 h-1 bg-blue-400 rounded-full animate-pulse opacity-60"></div>
          <div className="absolute top-4 right-1/3 w-1 h-1 bg-purple-400 rounded-full animate-pulse opacity-40" style={{ animationDelay: '1s' }}></div>
          <div className="absolute bottom-2 left-2/3 w-1 h-1 bg-indigo-400 rounded-full animate-pulse opacity-50" style={{ animationDelay: '2s' }}></div>
        </div>

        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <QuantumLogo />

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center gap-6">
              <QuantumNavigation />
              <SearchButton onClick={() => setSearchOpen(true)} />
              <TutorialButton />
              <div className="hidden md:block h-6 w-px bg-slate-600"></div>
              <ContactInfo />
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/50 transition-colors"
              aria-label="Toggle mobile menu"
              aria-expanded={mobileMenuOpen}
            >
              <div className="w-6 h-6 flex flex-col justify-center items-center">
                <span className={`block h-0.5 w-6 bg-current transition-all duration-300 ${mobileMenuOpen ? 'rotate-45 translate-y-1' : '-translate-y-1'}`} />
                <span className={`block h-0.5 w-6 bg-current transition-all duration-300 ${mobileMenuOpen ? 'opacity-0' : 'opacity-100'}`} />
                <span className={`block h-0.5 w-6 bg-current transition-all duration-300 ${mobileMenuOpen ? '-rotate-45 -translate-y-1' : 'translate-y-1'}`} />
              </div>
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <MobileMenu isOpen={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} />

        {/* Search Modal */}
        <QuantumSearch 
          isOpen={searchOpen} 
          onClose={() => setSearchOpen(false)}
          onNavigate={(section) => {
            const element = document.getElementById(section);
            if (element) {
              element.scrollIntoView({ behavior: 'smooth' });
            }
          }}
        />
      </header>
    </>
  );
};