import { useState } from "react"
import { NavLink, useLocation } from "react-router-dom"
import { 
  Home, 
  Calendar, 
  TrendingUp, 
  MessageCircle, 
  User,
  Dumbbell,
  Activity,
  Users,
  Settings,
  Brain,
  Target,
  Heart,
  BarChart3,
  Shield,
  Zap,
  Clock,
  Star,
  Crown,
  ChevronDown,
  Search,
  Bell,
  LogOut,
  Apple,
  Syringe,
  Dna,
  UserCheck
} from "lucide-react"
import { useAuth } from "@/contexts/AuthContext"
import { RepzLogo } from "@/ui/organisms/RepzLogo"
import { Badge } from "@/ui/atoms/Badge"
import { Button } from "@/ui/atoms/Button"
import { Input } from "@/ui/atoms/Input"
import { Avatar, AvatarFallback, AvatarImage } from "@/ui/atoms/Avatar"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/ui/molecules/DropdownMenu"

interface NavItem {
  title: string
  url: string
  icon: React.ComponentType<{ className?: string }> | string
  badge?: number
  tier?: string
}

interface NavGroup {
  label: string
  items: NavItem[]
  defaultOpen?: boolean
}

export function AppSidebar() {
  const { open, setOpen } = useSidebar()
  const { user, signOut } = useAuth()
  const location = useLocation()
  const [searchQuery, setSearchQuery] = useState("")
  const currentPath = location.pathname

  const isActive = (path: string) => currentPath === path
  const getNavCls = ({ isActive }: { isActive: boolean }) =>
    isActive ? "bg-accent text-accent-foreground font-medium" : "hover:bg-accent/50"

  const isCoach = user?.role === 'coach'
  const isAdmin = user?.role === 'coach' || false;

  const getNavGroups = (): NavGroup[] => {
    if (isCoach) {
      return [
        {
          label: "Dashboard",
          defaultOpen: true,
          items: [
            { title: "Overview", url: "/coach", icon: Home },
            { title: "My Clients", url: "/coach/clients", icon: Users, badge: 12 },
            { title: "Schedule", url: "/sessions", icon: Calendar, badge: 3 },
            { title: "Messages", url: "/messages", icon: MessageCircle, badge: 5 },
          ]
        },
        {
          label: "Analytics & Insights",
          defaultOpen: false,
          items: [
            { title: "Performance", url: "/analytics", icon: BarChart3 },
            { title: "Client Progress", url: "/coach/progress", icon: TrendingUp },
            { title: "Business Metrics", url: "/coach/business", icon: Target },
          ]
        },
        {
          label: "Tools",
          defaultOpen: false,
          items: [
            { title: "Workout Builder", url: "/coach/workouts", icon: Dumbbell },
            { title: "AI Coaching", url: "/coach/ai", icon: Brain, tier: "Pro" },
            { title: "Marketplace", url: "/marketplace", icon: Star },
          ]
        }
      ]
    } else {
      return [
        {
          label: "Dashboard",
          defaultOpen: true,
          items: [
            { title: "Overview", url: "/client-dashboard", icon: Home },
            { title: "Today's Workout", url: "/workouts/today", icon: Dumbbell, badge: 1 },
            { title: "Progress", url: "/progress", icon: TrendingUp },
            { title: "Messages", url: "/messages", icon: MessageCircle, badge: 2 },
          ]
        },
        {
          label: "Training",
          defaultOpen: false,
          items: [
            { title: "Workout Plans", url: "/workouts", icon: Activity },
            { title: "Live Sessions", url: "/live-workouts", icon: Zap, tier: "Pro" },
            { title: "Form Analysis", url: "/form-analysis", icon: Brain, tier: "Premium" },
            { title: "Recovery", url: "/recovery", icon: Heart },
          ]
        },
        {
          label: "Nutrition",
          defaultOpen: false,
          items: [
            { title: "Food Database", url: "/nutrition/foods", icon: Apple },
            { title: "Meal Planning", url: "/nutrition/meals", icon: Calendar },
            { title: "Recipes", url: "/nutrition/recipes", icon: Star },
          ]
        },
        {
          label: "Advanced Protocols",
          defaultOpen: false,
          items: [
            { title: "PEDs Management", url: "/protocols/peds", icon: Syringe, tier: "Performance+" },
            { title: "Bioregulators", url: "/protocols/bioregulators", icon: Dna, tier: "Longevity" },
            { title: "Medical Oversight", url: "/medical/oversight", icon: UserCheck, tier: "Performance+" },
          ]
        },
        {
          label: "Community",
          defaultOpen: false,
          items: [
            { title: "Challenges", url: "/challenges", icon: Target },
            { title: "Leaderboard", url: "/leaderboard", icon: Crown },
            { title: "Find Coaches", url: "/marketplace", icon: Users },
          ]
        }
      ]
    }
  }

  const adminItems: NavItem[] = isAdmin ? [
    { title: "Admin Panel", url: "/admin", icon: Shield },
    { title: "System Health", url: "/system-health", icon: Activity },
    { title: "Production", url: "/production-dashboard", icon: Settings },
  ] : []

  const navGroups = getNavGroups()
  
  const filteredGroups = navGroups.map(group => ({
    ...group,
    items: group.items.filter(item => 
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.url.toLowerCase().includes(searchQuery.toLowerCase())
    )
  })).filter(group => group.items.length > 0)

  const hasActiveRoute = (items: NavItem[]) => 
    items.some(item => isActive(item.url))

  return (
    <Sidebar className="w-72">
      <SidebarHeader className="border-b border-border p-4">
        <div className="flex items-center gap-3">
          <div className="flex-shrink-0">
            <RepzLogo className="h-8 w-8" />
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-semibold text-foreground truncate">
              REPZ Coach Pro
            </h2>
            <p className="text-xs text-muted-foreground">
              {isCoach ? 'Coach Portal' : 'Training Hub'}
            </p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="px-2">
        {/* Search Bar */}
        <div className="p-2 mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search features..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 h-8 bg-background/50"
            />
          </div>
        </div>

        {/* Navigation Groups */}
        {filteredGroups.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel className="text-xs font-medium text-muted-foreground mb-2">
              {group.label}
            </SidebarGroupLabel>

            <SidebarGroupContent>
              <SidebarMenu className="space-y-1">
                {group.items.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink 
                        to={item.url} 
                        end 
                        className={`${getNavCls({ isActive: isActive(item.url) })} group relative flex items-center gap-3 p-2 rounded-lg transition-all duration-200`}
                        title={item.title}
                      >
                        <item.icon className="h-4 w-4 flex-shrink-0" />
                        <span className="flex-1 text-sm font-medium truncate">
                          {item.title}
                        </span>
                        <div className="flex items-center gap-2">
                          {item.tier && (
                            <Badge variant="secondary" className="text-xs px-1.5 py-0.5">
                              {item.tier}
                            </Badge>
                          )}
                          {item.badge && (
                            <Badge variant="destructive" className="text-xs px-1.5 py-0.5">
                              {item.badge}
                            </Badge>
                          )}
                        </div>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}

        {/* Admin Section */}
        {adminItems.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel className="text-xs font-medium text-muted-foreground mb-2">
              Administration
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu className="space-y-1">
                {adminItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink 
                        to={item.url} 
                        className={`${getNavCls({ isActive: isActive(item.url) })} group relative flex items-center gap-3 p-2 rounded-lg transition-all duration-200`}
                         title={item.title}
                       >
                         <item.icon className="h-4 w-4 flex-shrink-0" />
                         <span className="flex-1 text-sm font-medium truncate">
                           {item.title}
                         </span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
      </SidebarContent>

      <SidebarFooter className="border-t border-border p-4">
        <div className="space-y-3">
          {/* User Profile */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="w-full justify-start gap-3 p-2 h-auto" onClick={() => console.log("app-sidebar button clicked")}>
                <Avatar className="h-8 w-8">
                  <AvatarImage src="/placeholder-avatar.png" />
                  <AvatarFallback>
                    {user?.email?.charAt(0).toUpperCase() || 'U'}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0 text-left">
                  <p className="text-sm font-medium truncate">
                    {user?.email?.split('@')[0] || 'User'}
                  </p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {user?.role || 'Member'}
                  </p>
                </div>
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuItem>
                <User className="h-4 w-4 mr-2" />
                Profile Settings
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Bell className="h-4 w-4 mr-2" />
                Notifications
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="h-4 w-4 mr-2" />
                Preferences
              </DropdownMenuItem>
              <DropdownMenuItem onClick={signOut} className="text-destructive">
                <LogOut className="h-4 w-4 mr-2" />
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Quick Actions */}
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="flex-1" onClick={() => console.log("app-sidebar button clicked")}>
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" className="flex-1" onClick={() => console.log("app-sidebar button clicked")}>
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}