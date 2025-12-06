import React, { ReactNode } from 'react';
import { ResponsiveHeader } from './ResponsiveHeader';
import { Footer } from '@/components/shared/Footer';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { User, Settings, LogOut, LayoutDashboard } from 'lucide-react';

interface PageLayoutProps {
  children: ReactNode;
  title?: string;
  showHeader?: boolean;
  showFooter?: boolean;
  showNavigation?: boolean;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '7xl' | 'full';
  className?: string;
}

const MAX_WIDTH_CLASSES = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '7xl': 'max-w-7xl',
  full: 'max-w-full',
};

export const PageLayout: React.FC<PageLayoutProps> = ({
  children,
  title,
  showHeader = true,
  showFooter = true,
  showNavigation = true,
  maxWidth = '7xl',
  className = '',
}) => {
  const { user, signOut, isAdmin } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  const NavigationContent = () => {
    if (!showNavigation) return null;

    if (!user) {
      return (
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            onClick={() => navigate('/pricing')}
            className="text-white/80 hover:text-white hover:bg-white/10"
          >
            Pricing
          </Button>
          <Button
            variant="ghost"
            onClick={() => navigate('/login')}
            className="text-white/80 hover:text-white hover:bg-white/10"
          >
            Login
          </Button>
          <Button
            onClick={() => navigate('/signup')}
            className="bg-repz-orange hover:bg-repz-orange-dark text-white"
          >
            Get Started
          </Button>
        </div>
      );
    }

    return (
      <div className="flex items-center gap-4">
        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-4">
          <Button
            variant="ghost"
            onClick={() => navigate('/dashboard')}
            className="text-white/80 hover:text-white hover:bg-white/10"
          >
            Dashboard
          </Button>
          <Button
            variant="ghost"
            onClick={() => navigate('/pricing')}
            className="text-white/80 hover:text-white hover:bg-white/10"
          >
            Plans
          </Button>
          {isAdmin && (
            <Button
              variant="ghost"
              onClick={() => navigate('/coach-admin')}
              className="text-white/80 hover:text-white hover:bg-white/10"
            >
              Admin
            </Button>
          )}
        </nav>

        {/* User Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-repz-orange flex items-center justify-center">
                <User className="h-5 w-5 text-white" />
              </div>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>
              <div className="flex flex-col">
                <span className="text-sm font-medium">{user.email}</span>
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => navigate('/dashboard')}>
              <LayoutDashboard className="mr-2 h-4 w-4" />
              Dashboard
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => navigate('/account')}>
              <Settings className="mr-2 h-4 w-4" />
              Account Settings
            </DropdownMenuItem>
            {isAdmin && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate('/coach-admin')}>
                  Admin Panel
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/admin/non-portal-clients')}>
                  Non-Portal Clients
                </DropdownMenuItem>
              </>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleSignOut}>
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    );
  };

  return (
    <div className="min-h-screen flex flex-col bg-surface-base">
      {showHeader && (
        <ResponsiveHeader sticky={true}>
          <NavigationContent />
        </ResponsiveHeader>
      )}

      <main className={`flex-1 ${className}`}>
        <div className={`container mx-auto px-4 sm:px-6 lg:px-8 ${MAX_WIDTH_CLASSES[maxWidth]} py-8`}>
          {title && (
            <div className="mb-6">
              <h1 className="text-3xl font-bold text-white">{title}</h1>
            </div>
          )}
          {children}
        </div>
      </main>

      {showFooter && <Footer />}
    </div>
  );
};
