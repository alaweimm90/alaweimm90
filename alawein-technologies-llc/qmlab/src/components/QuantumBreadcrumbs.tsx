import React from 'react';
import { ChevronRight, Home, BookOpen, Calculator, Atom, Brain } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface BreadcrumbItem {
  id: string;
  label: string;
  href?: string;
  icon?: React.ReactNode;
  isActive?: boolean;
}

interface QuantumBreadcrumbsProps {
  items: BreadcrumbItem[];
  onNavigate?: (id: string) => void;
  className?: string;
}

const getQuantumIcon = (id: string) => {
  switch (id) {
    case 'home':
      return <Home className="w-3 h-3" />;
    case 'lab':
      return <Calculator className="w-3 h-3" />;
    case 'circuit-builder':
      return <Calculator className="w-3 h-3" />;
    case 'bloch-sphere':
      return <Atom className="w-3 h-3" />;
    case 'training-dashboard':
      return <Brain className="w-3 h-3" />;
    case 'learning':
      return <BookOpen className="w-3 h-3" />;
    default:
      return <Calculator className="w-3 h-3" />;
  }
};

export const QuantumBreadcrumbs: React.FC<QuantumBreadcrumbsProps> = ({
  items,
  onNavigate,
  className = ''
}) => {
  const handleNavigate = (item: BreadcrumbItem) => {
    if (item.isActive) return; // Don't navigate to active item
    
    if (onNavigate) {
      onNavigate(item.id);
    } else if (item.href) {
      const element = document.getElementById(item.href);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <nav 
      aria-label="Breadcrumb navigation"
      className={`flex items-center space-x-1 ${className}`}
    >
      <ol className="flex items-center space-x-1">
        {items.map((item, index) => (
          <li key={item.id} className="flex items-center">
            {index > 0 && (
              <ChevronRight className="w-3 h-3 text-slate-500 mx-2" aria-hidden="true" />
            )}
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleNavigate(item)}
              disabled={item.isActive}
              className={`flex items-center gap-2 px-3 py-1 text-xs rounded-lg transition-all duration-200 ${
                item.isActive
                  ? 'text-blue-300 bg-blue-500/20 border border-blue-400/30 cursor-default'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
              aria-current={item.isActive ? 'page' : undefined}
            >
              {item.icon || getQuantumIcon(item.id)}
              <span className="hidden sm:inline">{item.label}</span>
              <span className="sm:hidden">{item.label.split(' ')[0]}</span>
            </Button>
          </li>
        ))}
      </ol>
    </nav>
  );
};

// Hook to generate breadcrumbs based on current section
export const useQuantumBreadcrumbs = () => {
  const [breadcrumbs, setBreadcrumbs] = React.useState<BreadcrumbItem[]>([
    { id: 'home', label: 'QMLab', href: 'hero-section' }
  ]);

  const updateBreadcrumbs = React.useCallback((currentSection: string) => {
    const baseBreadcrumbs: BreadcrumbItem[] = [
      { id: 'home', label: 'QMLab', href: 'hero-section' }
    ];

    switch (currentSection) {
      case 'lab-section':
        setBreadcrumbs([
          ...baseBreadcrumbs,
          { id: 'lab', label: 'Interactive Lab', href: 'lab-section', isActive: true }
        ]);
        break;
      
      case 'circuit-builder':
        setBreadcrumbs([
          ...baseBreadcrumbs,
          { id: 'lab', label: 'Interactive Lab', href: 'lab-section' },
          { id: 'circuit-builder', label: 'Circuit Builder', href: 'circuit-builder', isActive: true }
        ]);
        break;
        
      case 'bloch-sphere':
        setBreadcrumbs([
          ...baseBreadcrumbs,
          { id: 'lab', label: 'Interactive Lab', href: 'lab-section' },
          { id: 'bloch-sphere', label: 'Bloch Sphere', href: 'bloch-sphere', isActive: true }
        ]);
        break;
        
      case 'training-dashboard':
        setBreadcrumbs([
          ...baseBreadcrumbs,
          { id: 'lab', label: 'Interactive Lab', href: 'lab-section' },
          { id: 'training-dashboard', label: 'ML Training', href: 'training-dashboard', isActive: true }
        ]);
        break;
        
      case 'learning-resources':
        setBreadcrumbs([
          ...baseBreadcrumbs,
          { id: 'learning', label: 'Learning Resources', href: 'learning-resources', isActive: true }
        ]);
        break;
        
      case 'advanced-features':
        setBreadcrumbs([
          ...baseBreadcrumbs,
          { id: 'advanced', label: 'Advanced Features', href: 'advanced-features', isActive: true }
        ]);
        break;
        
      default:
        setBreadcrumbs(baseBreadcrumbs);
    }
  }, []);

  // Auto-update breadcrumbs based on scroll position
  React.useEffect(() => {
    const sections = [
      'hero-section',
      'lab-section', 
      'circuit-builder',
      'bloch-sphere',
      'training-dashboard',
      'learning-resources',
      'advanced-features'
    ];

    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100; // Offset for header

      for (let i = sections.length - 1; i >= 0; i--) {
        const element = document.getElementById(sections[i]);
        if (element && scrollPosition >= element.offsetTop) {
          updateBreadcrumbs(sections[i]);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial call

    return () => window.removeEventListener('scroll', handleScroll);
  }, [updateBreadcrumbs]);

  return { breadcrumbs, updateBreadcrumbs };
};