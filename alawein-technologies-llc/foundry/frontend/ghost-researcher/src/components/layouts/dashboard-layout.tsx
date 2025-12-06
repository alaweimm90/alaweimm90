'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  Home, Search, FileText, Brain, Users, Settings,
  LogOut, Menu, X, ChevronLeft, Bell, User, HelpCircle,
  BookOpen, BarChart, Archive, Star
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Avatar } from '@/components/ui/avatar'
import { useAuth } from '@/components/providers/auth-provider'
import { cn } from '@/lib/utils'
import { useTheme } from 'next-themes'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

const sidebarItems = [
  { href: '/dashboard', label: 'Dashboard', icon: Home },
  { href: '/research', label: 'Research', icon: Search },
  { href: '/papers', label: 'Papers', icon: FileText },
  { href: '/hypotheses', label: 'Hypotheses', icon: Brain },
  { href: '/projects', label: 'Projects', icon: Archive },
  { href: '/collaborators', label: 'Collaborators', icon: Users },
  { href: '/library', label: 'Library', icon: BookOpen },
  { href: '/analytics', label: 'Analytics', icon: BarChart },
  { href: '/starred', label: 'Starred', icon: Star },
]

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const pathname = usePathname()
  const { user, logout } = useAuth()
  const { theme, setTheme } = useTheme()

  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar - Desktop */}
      <aside
        className={cn(
          'fixed left-0 top-0 z-40 h-screen bg-card border-r transition-all duration-300 hidden lg:block',
          sidebarOpen ? 'w-64' : 'w-20'
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-between p-6">
          <Link href="/dashboard" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center shrink-0">
              <Search className="w-5 h-5 text-primary-foreground" />
            </div>
            {sidebarOpen && (
              <span className="font-bold text-xl">Ghost</span>
            )}
          </Link>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="hidden lg:flex"
          >
            <ChevronLeft className={cn(
              'h-4 w-4 transition-transform',
              !sidebarOpen && 'rotate-180'
            )} />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="px-3">
          <ul className="space-y-1">
            {sidebarItems.map((item) => {
              const isActive = pathname === item.href
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-secondary text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <item.icon className="w-5 h-5 shrink-0" />
                    {sidebarOpen && (
                      <span className="text-sm font-medium">{item.label}</span>
                    )}
                  </Link>
                </li>
              )
            })}
          </ul>
        </nav>

        {/* Bottom Actions */}
        <div className="absolute bottom-0 left-0 right-0 p-3 space-y-1">
          <Link
            href="/settings"
            className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          >
            <Settings className="w-5 h-5 shrink-0" />
            {sidebarOpen && <span className="text-sm font-medium">Settings</span>}
          </Link>
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          >
            <LogOut className="w-5 h-5 shrink-0" />
            {sidebarOpen && <span className="text-sm font-medium">Logout</span>}
          </button>
        </div>
      </aside>

      {/* Mobile Sidebar */}
      {mobileMenuOpen && (
        <motion.div
          initial={{ x: -300 }}
          animate={{ x: 0 }}
          exit={{ x: -300 }}
          className="fixed inset-0 z-50 lg:hidden"
        >
          <div className="fixed inset-0 bg-background/80 backdrop-blur-sm" onClick={() => setMobileMenuOpen(false)} />
          <aside className="fixed left-0 top-0 h-screen w-64 bg-card border-r">
            {/* Mobile sidebar content - same as desktop */}
            <div className="flex items-center justify-between p-6">
              <Link href="/dashboard" className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                  <Search className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="font-bold text-xl">Ghost</span>
              </Link>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setMobileMenuOpen(false)}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>

            <nav className="px-3">
              <ul className="space-y-1">
                {sidebarItems.map((item) => {
                  const isActive = pathname === item.href
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        onClick={() => setMobileMenuOpen(false)}
                        className={cn(
                          'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
                          isActive
                            ? 'bg-primary text-primary-foreground'
                            : 'hover:bg-secondary text-muted-foreground hover:text-foreground'
                        )}
                      >
                        <item.icon className="w-5 h-5" />
                        <span className="text-sm font-medium">{item.label}</span>
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </nav>
          </aside>
        </motion.div>
      )}

      {/* Main Content */}
      <div className={cn(
        'transition-all duration-300',
        sidebarOpen ? 'lg:pl-64' : 'lg:pl-20'
      )}>
        {/* Top Bar */}
        <header className="sticky top-0 z-30 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
          <div className="flex items-center justify-between px-6 py-3">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setMobileMenuOpen(true)}
                className="lg:hidden"
              >
                <Menu className="h-5 w-5" />
              </Button>

              {/* Search Bar */}
              <div className="relative hidden sm:block">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                <input
                  type="search"
                  placeholder="Search papers, projects, hypotheses..."
                  className="pl-10 pr-4 py-2 w-64 lg:w-96 bg-secondary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Notifications */}
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-5 w-5" />
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
              </Button>

              {/* Help */}
              <Button variant="ghost" size="icon">
                <HelpCircle className="h-5 w-5" />
              </Button>

              {/* User Menu */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="relative h-10 w-10 rounded-full">
                    <Avatar className="h-10 w-10">
                      <div className="w-full h-full bg-primary flex items-center justify-center text-primary-foreground">
                        {user?.name?.[0]?.toUpperCase() || 'U'}
                      </div>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" align="end" forceMount>
                  <DropdownMenuLabel className="font-normal">
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm font-medium leading-none">{user?.name}</p>
                      <p className="text-xs leading-none text-muted-foreground">
                        {user?.email}
                      </p>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem asChild>
                    <Link href="/profile">
                      <User className="mr-2 h-4 w-4" />
                      Profile
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/settings">
                      <Settings className="mr-2 h-4 w-4" />
                      Settings
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={logout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  )
}