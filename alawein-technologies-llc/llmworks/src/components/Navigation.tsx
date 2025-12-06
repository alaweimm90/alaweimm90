import { memo, useState, useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Swords, Zap, BarChart3, Settings, Github, Menu, X, Keyboard, Command } from "lucide-react";
import { KeyboardShortcutsModal } from "@/components/accessibility/KeyboardShortcutsModal";
import { trackEvent } from "@/lib/analytics";
import { useCommandPalette } from "@/hooks/useCommandPalette";

const NavigationComponent = () => {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const logoRef = useRef<HTMLDivElement>(null);
  const { openPalette } = useCommandPalette();
  
  const isActive = (path: string) => location.pathname === path;
  
  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);
  const closeMobileMenu = () => setIsMobileMenuOpen(false);

  const handleNavClick = (target: string) => {
    trackEvent("nav_click", { target, from: location.pathname });
    closeMobileMenu();
  };

  // Scroll detection for enhanced nav styling
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Logo magnetic effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!logoRef.current) return;
      const rect = logoRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left - rect.width / 2) * 0.1;
      const y = (e.clientY - rect.top - rect.height / 2) * 0.1;
      logoRef.current.style.transform = `translate(${x}px, ${y}px)`;
    };

    const handleMouseLeave = () => {
      if (!logoRef.current) return;
      logoRef.current.style.transform = 'translate(0px, 0px)';
    };

    const element = logoRef.current;
    if (element) {
      element.addEventListener('mousemove', handleMouseMove);
      element.addEventListener('mouseleave', handleMouseLeave);
    }

    return () => {
      if (element) {
        element.removeEventListener('mousemove', handleMouseMove);
        element.removeEventListener('mouseleave', handleMouseLeave);
      }
    };
  }, []);

  return (
    <nav 
      className={`
        glass-panel border-0 border-b border-border/10 sticky top-0 z-50 transition-all duration-500
        ${scrolled ? 'shadow-lg backdrop-blur-xl' : 'backdrop-blur-lg'}
      `} 
      role="navigation" 
      aria-label="Main navigation"
      style={{
        background: scrolled 
          ? 'hsl(var(--glass-background))' 
          : 'hsl(var(--color-background) / 0.8)'
      }}
    >
      <div className="container-elegant">
        <div className="flex items-center justify-between h-20">
          {/* Sophisticated Logo */}
          <Link 
            to="/" 
            className="flex items-center gap-4 group focus-elegant transition-all duration-300 rounded-xl p-2 -m-2" 
            aria-label="LLM Works Home"
            onClick={() => handleNavClick("home")}
          >
            <div 
              ref={logoRef}
              className="glass-panel p-3 rounded-xl group-hover:shadow-lg transition-all duration-500 relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-secondary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <Swords className="h-7 w-7 text-primary relative z-10 group-hover:rotate-12 transition-transform duration-500" aria-hidden="true" />
            </div>
            <div className="flex flex-col">
              <span className="heading-refined text-2xl text-primary group-hover:text-primary/90 transition-colors">
                LLM WORKS
              </span>
              <span className="text-xs text-primary/60 font-mono tracking-wider -mt-1">
                STRATEGIC COMMAND
              </span>
            </div>
          </Link>

          {/* Sophisticated Navigation Links */}
          <div className="hidden md:flex items-center gap-1">
            <Button
              variant={isActive("/") ? "default" : "ghost"}
              size="sm"
              asChild
              className="btn-elegant relative group hover:bg-primary/10 transition-all duration-300"
              aria-current={isActive("/") ? "page" : undefined}
            >
              <Link to="/" onClick={() => handleNavClick("platform")} className="flex items-center gap-2">
                <Swords className="h-4 w-4 transition-all duration-300 group-hover:rotate-12" />
                <span className="font-medium">Platform</span>
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/20 to-secondary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
              </Link>
            </Button>

            <Button
              variant={isActive("/arena") ? "default" : "ghost"}
              size="sm"
              asChild
              className="btn-elegant relative group hover:bg-primary/10 transition-all duration-300"
              aria-current={isActive("/arena") ? "page" : undefined}
            >
              <Link to="/arena" onClick={() => handleNavClick("arena")} className="flex items-center gap-2">
                <Zap className="h-4 w-4 transition-all duration-300 group-hover:scale-110 group-hover:text-yellow-400" />
                <span className="font-medium">Arena</span>
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-yellow-500/20 to-orange-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
              </Link>
            </Button>

            <Button
              variant={isActive("/bench") ? "default" : "ghost"}
              size="sm"
              asChild
              className="btn-elegant relative group hover:bg-primary/10 transition-all duration-300"
              aria-current={isActive("/bench") ? "page" : undefined}
            >
              <Link to="/bench" onClick={() => handleNavClick("bench")} className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 transition-all duration-300 group-hover:scale-110" />
                <span className="font-medium">Bench</span>
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-green-500/20 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
              </Link>
            </Button>
            
            <div className="h-6 w-px bg-gradient-to-b from-transparent via-border/50 to-transparent mx-3" aria-hidden="true" />
            
            <Button
              variant="ghost"
              size="sm"
              asChild
              className="btn-elegant relative group hover:bg-primary/10 transition-all duration-300"
            >
              <a href="https://github.com/alaweimm90/aegis-ai-evaluator" target="_blank" rel="noopener noreferrer" onClick={() => handleNavClick("github")} className="flex items-center gap-2">
                <Github className="h-4 w-4 transition-all duration-300 group-hover:scale-110" />
                <span className="font-medium">GitHub</span>
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-slate-500/20 to-gray-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
              </a>
            </Button>
            
            {/* Command Palette Trigger */}
            <Button
              variant="ghost"
              size="sm"
              className="btn-elegant relative group hover:bg-primary/10 transition-all duration-300"
              onClick={() => {
                handleNavClick("command-palette");
                openPalette();
              }}
            >
              <Command className="h-4 w-4 mr-2 transition-all duration-300 group-hover:scale-110" />
              <span className="font-medium text-xs">⌘K</span>
              <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-purple-500/20 to-violet-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
            </Button>
            
            <KeyboardShortcutsModal 
              trigger={
                <Button
                  variant="ghost"
                  size="sm"
                  className="btn-elegant relative group hover:bg-primary/10 transition-all duration-300"
                  onClick={() => handleNavClick("shortcuts")}
                >
                  <Keyboard className="h-4 w-4 mr-2 transition-all duration-300 group-hover:scale-110" />
                  <span className="font-medium">Help</span>
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-500/20 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                </Button>
              }
            />
          </div>

          <div className="flex items-center gap-4">
            {/* Security Status Badge */}
            <Badge 
              variant="secondary" 
              className="hidden md:inline-flex text-xs bg-success/10 text-success border-success/30"
            >
              ✓ Local Processing
            </Badge>
            
            <Button 
              variant="outline" 
              size="sm" 
              asChild
              className="hidden sm:inline-flex transition-smooth hover:bg-primary/10 border-primary/30 text-primary min-h-[44px]"
            >
              <Link to="/dashboard" onClick={() => handleNavClick("dashboard")}>
                Dashboard
              </Link>
            </Button>

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden p-2"
              onClick={toggleMobileMenu}
              aria-label="Toggle mobile menu"
              aria-expanded={isMobileMenuOpen}
            >
              {isMobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </Button>
          </div>
        </div>

        {/* Sophisticated Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden glass-panel border-t border-border/10 shadow-elegant">
            <div className="container-elegant py-6 space-y-3 stagger-children">
              <div style={{'--stagger-index': 0} as React.CSSProperties}>
                <Button
                  variant={isActive("/") ? "default" : "ghost"}
                  size="sm"
                  asChild
                  className="btn-elegant w-full justify-start relative group hover:bg-primary/10"
                  aria-current={isActive("/") ? "page" : undefined}
                >
                  <Link to="/" onClick={() => handleNavClick("platform")} className="flex items-center gap-3 p-4">
                    <div className="glass-subtle p-2 rounded-lg group-hover:shadow-md transition-all">
                      <Swords className="h-4 w-4 transition-all group-hover:rotate-12" />
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Platform</span>
                      <span className="text-xs opacity-60">Overview & Stats</span>
                    </div>
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/20 to-secondary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                  </Link>
                </Button>
              </div>

              <div style={{'--stagger-index': 1} as React.CSSProperties}>
                <Button
                  variant={isActive("/arena") ? "default" : "ghost"}
                  size="sm"
                  asChild
                  className="btn-elegant w-full justify-start relative group hover:bg-primary/10"
                  aria-current={isActive("/arena") ? "page" : undefined}
                >
                  <Link to="/arena" onClick={() => handleNavClick("arena")} className="flex items-center gap-3 p-4">
                    <div className="glass-subtle p-2 rounded-lg group-hover:shadow-md transition-all">
                      <Zap className="h-4 w-4 transition-all group-hover:scale-110 group-hover:text-yellow-400" />
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Arena</span>
                      <span className="text-xs opacity-60">Battle Interface</span>
                    </div>
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-yellow-500/20 to-orange-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                  </Link>
                </Button>
              </div>

              <div style={{'--stagger-index': 2} as React.CSSProperties}>
                <Button
                  variant={isActive("/bench") ? "default" : "ghost"}
                  size="sm"
                  asChild
                  className="btn-elegant w-full justify-start relative group hover:bg-primary/10"
                  aria-current={isActive("/bench") ? "page" : undefined}
                >
                  <Link to="/bench" onClick={() => handleNavClick("bench")} className="flex items-center gap-3 p-4">
                    <div className="glass-subtle p-2 rounded-lg group-hover:shadow-md transition-all">
                      <BarChart3 className="h-4 w-4 transition-all group-hover:scale-110" />
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Bench</span>
                      <span className="text-xs opacity-60">Benchmarks</span>
                    </div>
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-green-500/20 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                  </Link>
                </Button>
              </div>

              <div style={{'--stagger-index': 3} as React.CSSProperties}>
                <Button
                  variant={isActive("/dashboard") ? "default" : "ghost"}
                  size="sm"
                  asChild
                  className="btn-elegant w-full justify-start relative group hover:bg-primary/10"
                  aria-current={isActive("/dashboard") ? "page" : undefined}
                >
                  <Link to="/dashboard" onClick={() => handleNavClick("dashboard")} className="flex items-center gap-3 p-4">
                    <div className="glass-subtle p-2 rounded-lg group-hover:shadow-md transition-all">
                      <Settings className="h-4 w-4 transition-all group-hover:rotate-90" />
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">Dashboard</span>
                      <span className="text-xs opacity-60">Management</span>
                    </div>
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-500/20 to-indigo-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                  </Link>
                </Button>
              </div>

              <div className="h-px bg-gradient-to-r from-transparent via-border/50 to-transparent my-4" aria-hidden="true" />

              <div style={{'--stagger-index': 4} as React.CSSProperties}>
                <Button
                  variant="ghost"
                  size="sm"
                  asChild
                  className="btn-elegant w-full justify-start relative group hover:bg-primary/10"
                >
                  <a href="https://github.com/alaweimm90/aegis-ai-evaluator" target="_blank" rel="noopener noreferrer" onClick={() => handleNavClick("github")} className="flex items-center gap-3 p-4">
                    <div className="glass-subtle p-2 rounded-lg group-hover:shadow-md transition-all">
                      <Github className="h-4 w-4 transition-all group-hover:scale-110" />
                    </div>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">GitHub</span>
                      <span className="text-xs opacity-60">Source Code</span>
                    </div>
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-slate-500/20 to-gray-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                  </a>
                </Button>
              </div>
              
              <div style={{'--stagger-index': 5} as React.CSSProperties}>
                <KeyboardShortcutsModal 
                  trigger={
                    <Button
                      variant="ghost"
                      size="sm"
                      className="btn-elegant w-full justify-start relative group hover:bg-primary/10"
                      onClick={() => handleNavClick("shortcuts")}
                    >
                      <div className="flex items-center gap-3 p-4 w-full">
                        <div className="glass-subtle p-2 rounded-lg group-hover:shadow-md transition-all">
                          <Keyboard className="h-4 w-4 transition-all group-hover:scale-110" />
                        </div>
                        <div className="flex flex-col items-start">
                          <span className="font-medium">Help</span>
                          <span className="text-xs opacity-60">Shortcuts</span>
                        </div>
                        <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-purple-500/20 to-violet-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                      </div>
                    </Button>
                  }
                />
              </div>

              <div className="pt-4" style={{'--stagger-index': 6} as React.CSSProperties}>
                <div className="glass-minimal p-3 rounded-lg">
                  <Badge 
                    variant="secondary" 
                    className="text-xs bg-success/10 text-success border-success/30 font-medium"
                  >
                    <div className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse"></div>
                    Local Processing
                  </Badge>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export const Navigation = memo(NavigationComponent);