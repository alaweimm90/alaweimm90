import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { QLogo } from '@/components/QLogo';
import { StatusChip } from '@/components/ui/status-chip';
import { Menu, X, Home, Zap, Eye, TrendingUp, BookOpen, Settings } from 'lucide-react';

interface NavbarProps {
  className?: string;
}

export const Navbar: React.FC<NavbarProps> = ({ className = "" }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navItems = [
    { name: 'Playground', href: '#main-content', icon: Home },
    { name: 'Docs', href: '#docs', icon: BookOpen },
    { name: 'Examples', href: '#examples', icon: Zap },
    { name: 'GitHub', href: 'https://github.com/your-org/qmlab', icon: Eye, external: true },
    { name: 'Community', href: '#community', icon: TrendingUp },
  ];

  const scrollToSection = (href: string) => {
    if (href.startsWith('http')) {
      window.open(href, '_blank', 'noopener,noreferrer');
      return;
    }
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
    setIsMenuOpen(false);
  };

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 bg-surface-1/90 backdrop-blur-md border-b border-border ${className}`} role="navigation" aria-label="Main navigation">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <QLogo variant="full" className="h-8" />
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => (
              <Button
                key={item.name}
                variant="ghost"
                size="sm"
                onClick={() => scrollToSection(item.href)}
                className="flex items-center gap-2 text-text hover:text-primary transition-colors"
              >
                <item.icon className="w-4 h-4" />
                {item.name}
              </Button>
            ))}
          </div>

          {/* OSS Status Indicators */}
          <div className="hidden md:flex items-center gap-3">
            <StatusChip variant="success" size="sm">Open Source</StatusChip>
            <a 
              href="https://github.com/your-org/qmlab" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-text hover:text-primary transition-colors text-small"
              aria-label="Star QMLab on GitHub"
            >
              ⭐ Star on GitHub
            </a>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 bg-surface-2/90 rounded-lg mb-4 border border-border">
              {navItems.map((item) => (
                <Button
                  key={item.name}
                  variant="ghost"
                  size="sm"
                  onClick={() => scrollToSection(item.href)}
                  className="w-full justify-start gap-2 text-text hover:text-primary"
                >
                  <item.icon className="w-4 h-4" />
                  {item.name}
                </Button>
              ))}
              <div className="pt-2 border-t border-border">
                <StatusChip variant="success" size="sm">Online</StatusChip>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};