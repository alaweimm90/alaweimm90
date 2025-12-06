import { useState } from "react";
import { 
  Atom, 
  Layers3, 
  Waves, 
  Zap, 
  Gem, 
  Magnet,
  Orbit,
  Shuffle,
  FileText,
  AudioWaveform,
  Sparkles,
  Calculator,
  Brain,
  BarChart3,
  Home,
  Menu,
  ChevronRight,
  Search,
  Bell,
  BookOpen,
  Book,
  MessageCircle,
  Github,
  Linkedin,
  ChevronDown,
  ExternalLink,
  Mail,
  Globe,
  GraduationCap,
  Monitor,
  Cpu,
  Activity,
  TrendingUp,
  Layers,
  Settings,
  Gauge,
  LineChart,
  Eye,
  Download
} from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
  useSidebar,
} from "@/components/ui/sidebar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { UserPreferences } from "@/components/UserPreferences";
import { NotificationCenter } from "@/components/NotificationCenter";
import { useIsMobile } from "@/hooks/use-mobile";
import BrandLogo from "@/components/ui/BrandLogo";
import { categoryToGroupDomain } from '@/lib/category-domain-map';
import DynamicIcon from '@/components/ui/DynamicIcon';

// Organized module categories
const moduleCategories = {
  "Band Structure": [
    { title: "Graphene Band Structure", url: "/modules/graphene-band-structure", icon: Layers3 },
    { title: "Phonon Band Structure", url: "/modules/phonon-band-structure", icon: AudioWaveform },
    { title: "BZ Folding", url: "/modules/bz-folding", icon: Shuffle },
  ],
  "Quantum Dynamics": [
    { title: "TDSE Solver", url: "/modules/tdse-solver", icon: Waves },
    { title: "Bloch Sphere", url: "/modules/bloch-sphere", icon: Orbit },
    { title: "Quantum Tunneling", url: "/modules/quantum-tunneling", icon: Zap },
  ],
  "Materials & Crystals": [
    { title: "MoS₂ Valley Physics", url: "/modules/mos2-valley-physics", icon: Gem },
    { title: "Crystal Visualizer", url: "/modules/crystal-visualizer", icon: Atom },
  ],
  "Spin & Magnetism": [
    { title: "LLG Dynamics", url: "/modules/llg-dynamics", icon: Magnet },
  ],
  "Statistical Physics": [
    { title: "2D Ising Model", url: "/modules/ising-model", icon: Calculator },
    { title: "Boltzmann Distribution", url: "/modules/boltzmann-distribution", icon: BarChart3 },
    { title: "Microstates & Entropy", url: "/modules/microstates-entropy", icon: Shuffle },
    { title: "Canonical Ensemble", url: "/modules/canonical-ensemble", icon: Calculator },
    { title: "Brownian Motion", url: "/modules/brownian-motion", icon: Waves },
  ],
  "Field Theory": [
    { title: "Quantum Field Theory", url: "/modules/quantum-field-theory", icon: Atom },
    { title: "Laplace Eigenmodes", url: "/modules/laplace-eigenmodes", icon: FileText },
  ],
  // Scientific ML subfields
  "Physics-Informed Models": [
    { title: "PINN Schrödinger", url: "/modules/pinn-schrodinger", icon: Brain },
  ],
  "Operator Learning": [
    { title: "Neural Operator PDE Surrogate", url: "/modules/ml-showcase", icon: Cpu },
  ],
  "Symbolic Regression": [
    { title: "Symbolic Regression Discovery", url: "/modules/ml-showcase", icon: Sparkles },
  ],
  "Analytics & Patterns": [
    { title: "ML Showcase", url: "/modules/ml-showcase", icon: BarChart3 },
  ],
  "Advanced Features": [
    { title: "Simulation Dashboard", url: "/simulation-dashboard", icon: Monitor },
    { title: "About Platform", url: "/about-platform", icon: BookOpen },
  ]
} as const;

// ... keep existing code (removed local category→domain map; using shared utility)


// Flatten for search
const allModules = Object.values(moduleCategories).flat();

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();
  const currentPath = location.pathname;
  const unreadNotifications = 3; // This would come from a notification hook in real app
  const isMobile = useIsMobile();
  
  // State management
  const [searchQuery, setSearchQuery] = useState("");
  const [showPreferences, setShowPreferences] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    Object.entries(moduleCategories).forEach(([category, modules]) => {
      initial[category] = modules.some(m => m.url === currentPath);
    });
    return initial;
  });
const [expandedDomains, setExpandedDomains] = useState<Record<'Physics' | 'Mathematics' | 'Scientific ML', boolean>>(() => {
    const initial: Record<'Physics' | 'Mathematics' | 'Scientific ML', boolean> = {
      Physics: false,
      Mathematics: false,
      'Scientific ML': false,
    };
    Object.entries(moduleCategories).forEach(([category, modules]) => {
      const domain = categoryToGroupDomain(category);
      if (domain === 'Physics' || domain === 'Mathematics' || domain === 'Scientific ML') {
        if (modules.some(m => m.url === currentPath)) {
          initial[domain] = true;
        }
      }
    });
    return initial;
  });
  

  // Helper functions
  const isActive = (path: string) => currentPath === path;
  const getNavClasses = (active: boolean) =>
    `nav-link flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 ${
      active 
        ? "bg-primary/10 text-primary border border-primary/20 shadow-quantum" 
        : "hover:bg-muted/50 text-muted-foreground hover:text-foreground"
    }`;

  // Filter modules based on search
  const filteredCategories = Object.entries(moduleCategories).reduce((acc, [category, modules]) => {
    const filteredModules = modules.filter(module =>
      module.title.toLowerCase().includes(searchQuery.toLowerCase())
    );
    if (filteredModules.length > 0) {
      acc[category] = filteredModules;
    }
    return acc;
  }, {} as typeof moduleCategories);

  // Toggle domain/category expansion
  const toggleDomain = (domain: 'Physics' | 'Mathematics' | 'Scientific ML') => {
    setExpandedDomains(prev => ({ ...prev, [domain]: !prev[domain] }));
  };
  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  // Collapse all categories
  const collapseAll = () => {
    setExpandedCategories(Object.keys(moduleCategories).reduce((acc, category) => {
      acc[category] = false;
      return acc;
    }, {} as Record<string, boolean>));
  };

  // External links
  const handleExternalLink = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <Sidebar
      className="transition-all duration-300 border-r border-border/50 bg-card/50 backdrop-blur-sm"
      collapsible="offcanvas"
      variant="sidebar"
    >
      <SidebarHeader className="p-4 border-b border-border/50">
        {!collapsed ? (
          <div className="flex items-center justify-center">
            <BrandLogo inline size="md" withIcon />
          </div>
        ) : (
          <div className="flex items-center justify-center mb-4">
            <BrandLogo inline size="sm" withIcon className="icon-only" />
          </div>
        )}

        {/* Search Bar */}
        {!collapsed && (
          <div className="relative mt-4">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search simulations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 h-8 text-sm"
            />
          </div>
        )}
      </SidebarHeader>

      <SidebarContent className="px-3 py-4">
        {/* Home Section */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <NavLink to="/" className={getNavClasses(isActive("/"))}>
                    <Home className="w-5 h-5" />
                    {!collapsed && <span>Dashboard Home</span>}
                  </NavLink>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>


        {/* Quick Access to Advanced Features */}
        {!collapsed && (
          <>
            <SidebarGroup>
              <SidebarGroupLabel className="text-xs font-semibold uppercase tracking-wider gradient-text-quantum text-center">
                Quick Access
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <div className="card-surface-glass rounded-xl p-2">
                  <SidebarMenu className="space-y-1">
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild>
                        <NavLink to="/simulation-dashboard" className={getNavClasses(isActive("/simulation-dashboard"))}>
                          <Monitor className="w-4 h-4" />
                          <span className="text-sm">Live Dashboard</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild>
                        <NavLink to="/interactive-learning" className={getNavClasses(isActive("/interactive-learning"))}>
                          <Book className="w-4 h-4" />
                          <span className="text-sm">Interactive Learning</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild>
                        <NavLink to="/enhanced-visualization" className={getNavClasses(isActive("/enhanced-visualization"))}>
                          <Eye className="w-4 h-4" />
                          <span className="text-sm">Enhanced Visualization</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild>
                        <NavLink to="/data-export" className={getNavClasses(isActive("/data-export"))}>
                          <Download className="w-4 h-4" />
                          <span className="text-sm">Data Export</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild>
                        <NavLink to="/documentation" className={getNavClasses(isActive("/documentation"))}>
                          <Book className="w-4 h-4" />
                          <span className="text-sm">Theory & Methods</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  </SidebarMenu>
                </div>
              </SidebarGroupContent>
            </SidebarGroup>
            <div className="my-3 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
          </>
        )}

        {/* Modules Panel */}
        <SidebarGroup>
          <SidebarGroupLabel className="text-xs font-semibold uppercase tracking-wider gradient-text-quantum text-center">
            Modules
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="card-surface-glass rounded-xl border border-primary/30 p-2">
              {(['Physics', 'Mathematics', 'Scientific ML'] as const).map((domain) => {
                const categoriesInDomain = Object.entries(filteredCategories).filter(([cat]) => categoryToGroupDomain(cat) === domain);
                if (categoriesInDomain.length === 0) return null;
                return (
                  <div key={domain} className="mb-3 last:mb-0">
                    <div
                      className="relative px-2 py-1.5 rounded-md bg-primary/5 border border-primary/10 flex items-center justify-center text-center cursor-pointer"
                      onClick={() => toggleDomain(domain)}
                    >
                      <span className="text-xs font-semibold">{domain}</span>
                      <ChevronRight
                        className={`absolute right-2 w-3 h-3 transition-transform duration-200 ${expandedDomains[domain] ? 'rotate-90' : ''}`}
                      />
                    </div>

                    <div className={`mt-2 transition-all duration-300 ${expandedDomains[domain] || collapsed ? 'opacity-100' : 'opacity-0 max-h-0 overflow-hidden'}`}>
                      {categoriesInDomain.map(([category, modules]) => (
                        <div key={category} className="mt-2">
                          <SidebarGroup>
                            {/* Avoid duplicate label when category matches domain name */}
                            {(category !== domain) && (
                              <SidebarGroupLabel 
                                className={`flex items-center justify-between cursor-pointer hover:text-foreground ${collapsed ? 'hidden' : ''}`}
                                onClick={() => toggleCategory(category)}
                              >
                                <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{category}</span>
                                <ChevronRight 
                                  className={`w-3 h-3 transition-transform duration-200 ${expandedCategories[category] ? 'rotate-90' : ''}`}
                                />
                              </SidebarGroupLabel>
                            )}
                            <SidebarGroupContent 
                              className={`transition-all duration-300 ${(expandedCategories[category] || collapsed || category === domain) ? 'opacity-100' : 'opacity-0 max-h-0 overflow-hidden'}`}
                            >
                              <SidebarMenu className="space-y-1">
                                {modules.map((module) => (
                                  <SidebarMenuItem key={module.title}>
                                    <SidebarMenuButton asChild>
                                      <NavLink 
                                        to={module.url} 
                                        className={getNavClasses(isActive(module.url))}
                                        title={module.title}
                                      >
                                        {(module as any).iconName ? (
                                          <DynamicIcon name={(module as any).iconName} className="w-4 h-4 flex-shrink-0" />
                                        ) : (
                                          <module.icon className="w-4 h-4 flex-shrink-0" />
                                        )}
                                        {!collapsed && (
                                          <span className="text-sm font-medium truncate">{module.title}</span>
                                        )}
                                      </NavLink>
                                    </SidebarMenuButton>
                                  </SidebarMenuItem>
                                ))}
                              </SidebarMenu>
                            </SidebarGroupContent>
                          </SidebarGroup>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

    </Sidebar>
  );
}