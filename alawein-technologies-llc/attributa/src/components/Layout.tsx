import { ReactNode, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FileSearch, History, Github } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/store';
import { MobileNav } from '@/components/MobileNav';
import { SettingsDrawer } from '@/components/SettingsDrawer';
import AISecurityIcon from '@/components/dev/AISecurityIcon';
import { A11yToolbar } from '@/components/ui/a11y-toolbar';
import { useA11y } from '@/hooks/useA11y';
import AIAssistantPanel from '@/components/ai/AIAssistantPanel';
import AIFloatingButton from '@/components/ai/AIFloatingButton';

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();
  const { analysisOptions } = useAppStore();
  const a11yPreferences = useA11y();
  
  const isActive = (path: string) => {
    if (path === '/' && location.pathname === '/') return true;
    return path !== '/' && location.pathname.startsWith(path);
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary engineering-pattern">
      {/* Skip Link for Keyboard Navigation - WCAG SC 2.4.1 */}
      <a 
        href="#main-content" 
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
      >
        Skip to main content
      </a>
      
      {/* Header */}
      <header className="glass-nav sticky top-0 z-50 transition-all duration-300">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <div className="flex items-center space-x-6">
            <Link to="/" className="flex items-center space-x-3 hover:opacity-80 transition-all duration-300 hover:scale-105">
              <AISecurityIcon size={32} className="text-primary" />
              <span className="text-xl font-bold font-mono bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Attributa
              </span>
            </Link>
            
            <nav className="hidden md:flex space-x-1">
              <Button
                variant={isActive('/workspace') ? 'secondary' : 'ghost'}
                size="sm"
                asChild
              >
                <Link to="/workspace" className="flex items-center space-x-2">
                  <History className="h-4 w-4" />
                  <span>Workspace</span>
                </Link>
              </Button>
            </nav>
          </div>

          <div className="flex items-center space-x-2">
            {analysisOptions.useLocalOnly && (
              <Badge variant="secondary" className="bg-success/10 text-success border-success/20 text-xs">
                Local-only
              </Badge>
            )}
            
            {/* Settings Gear */}
            <SettingsDrawer />
            
            <Button size="sm" asChild>
              <Link to="/scan" className="flex items-center space-x-2">
                <FileSearch className="h-4 w-4" />
                <span>Analyze</span>
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main id="main-content" className="flex-1">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t bg-muted/50 py-8">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center space-y-6">
            <div className="text-center space-y-2">
              <p className="text-sm text-muted-foreground">
                Text analysis for AI pattern detection
              </p>
              <p className="text-xs text-muted-foreground/70">
                Research tool with explicit uncertainty bounds
              </p>
            </div>
            
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Badge variant="outline" className="text-xs">
                Open Source
              </Badge>
              
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="ghost" size="sm" asChild>
                  <a href="https://github.com/alaweimm90" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-xs">
                    <Github className="h-3 w-3" />
                    GitHub
                  </a>
                </Button>
                <Button variant="ghost" size="sm" asChild>
                  <a href="https://www.linkedin.com/in/meshal-alawein" target="_blank" rel="noopener noreferrer" className="text-xs">
                    LinkedIn
                  </a>
                </Button>
                <Button variant="ghost" size="sm" asChild>
                  <a href="https://malawein.com" target="_blank" rel="noopener noreferrer" className="text-xs">
                    Website
                  </a>
                </Button>
                <Button variant="ghost" size="sm" asChild>
                  <a href="https://www.simcore.dev" target="_blank" rel="noopener noreferrer" className="text-xs">
                    Simcore
                  </a>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </footer>
      

      {/* Accessibility Toolbar */}
      <A11yToolbar />

      {/* Mobile Navigation */}
      <MobileNav />

      {/* AI Assistant */}
      <AIAssistantPanel />
      <AIFloatingButton />
    </div>
  );
};

export default Layout;