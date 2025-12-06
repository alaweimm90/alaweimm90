import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Button } from '@/ui/atoms/Button';
import { RepzLogo } from '@/ui/organisms/RepzLogo';
import { useAuth } from '@/contexts/AuthContext';
import { Menu, X, User, LogOut } from 'lucide-react';

/**
 * Atomic Design - Navigation Organism
 * Combines multiple molecules and atoms into a complete navigation system
 * All styling derived from design tokens for consistency
 */

export interface NavigationProps {
  variant?: 'header' | 'mobile' | 'sidebar';
  showAuth?: boolean;
  className?: string;
}

export interface NavigationItem {
  label: string;
  href: string;
  icon?: React.ReactNode;
  requiresAuth?: boolean;
  external?: boolean;
}

const navigationItems: NavigationItem[] = [
  { label: 'Home', href: '/' },
  { label: 'Pricing', href: '/pricing' },
  { label: 'Dashboard', href: '/dashboard', requiresAuth: true },
  { label: 'Progress', href: '/progress', requiresAuth: true },
  { label: 'Sessions', href: '/sessions', requiresAuth: true },
];

export const Navigation = React.forwardRef<HTMLElement, NavigationProps>(({
  variant = 'header',
  showAuth = true,
  className,
  ...props
}, ref) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, signOut, isAuthenticated } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const handleSignOut = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  const isActivePath = (href: string) => {
    if (href === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(href);
  };

  const getVisibleItems = () => {
    return navigationItems.filter(item => {
      if (item.requiresAuth && !isAuthenticated) return false;
      return true;
    });
  };

  if (variant === 'mobile') {
    return (
      <nav
        className={cn('repz-navigation repz-navigation--mobile', className)}
        ref={ref}
        {...props}
      >
        <div className="repz-navigation__container">
          <div className="repz-navigation__brand">
            <RepzLogo size="sm" />
          </div>
          
          <div className="repz-navigation__toggle">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-label="Toggle navigation menu"
              aria-expanded={isMobileMenuOpen}
            >
              {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </Button>
          </div>
        </div>

        {isMobileMenuOpen && (
          <div className="repz-navigation__menu repz-navigation__menu--mobile">
            <div className="repz-navigation__items">
              {getVisibleItems().map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  onClick={(e) => {
                    e.preventDefault();
                    navigate(item.href);
                    setIsMobileMenuOpen(false);
                  }}
                  className={cn(
                    'repz-navigation__link',
                    isActivePath(item.href) && 'repz-navigation__link--active'
                  )}
                >
                  {item.icon && <span className="repz-navigation__icon">{item.icon}</span>}
                  {item.label}
                </a>
              ))}
            </div>

            {showAuth && (
              <div className="repz-navigation__auth">
                {isAuthenticated ? (
                  <div className="repz-navigation__user">
                    <div className="repz-navigation__user-info">
                      <User size={16} />
                      <span>{user?.email}</span>
                    </div>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleSignOut}
                      className="repz-navigation__signout"
                    >
                      <LogOut size={16} />
                      Sign Out
                    </Button>
                  </div>
                ) : (
                  <div className="repz-navigation__auth-buttons">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        navigate('/login');
                        setIsMobileMenuOpen(false);
                      }}
                    >
                      Sign In
                    </Button>
                    <Button
                      variant="accent"
                      size="sm"
                      onClick={() => {
                        navigate('/signup');
                        setIsMobileMenuOpen(false);
                      }}
                    >
                      Sign Up
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </nav>
    );
  }

  if (variant === 'sidebar') {
    return (
      <nav
        className={cn('repz-navigation repz-navigation--sidebar', className)}
        ref={ref}
        {...props}
      >
        <div className="repz-navigation__brand repz-navigation__brand--sidebar">
          <RepzLogo size="md" />
        </div>

        <div className="repz-navigation__items repz-navigation__items--sidebar">
          {getVisibleItems().map((item) => (
            <a
              key={item.href}
              href={item.href}
              onClick={(e) => {
                e.preventDefault();
                navigate(item.href);
              }}
              className={cn(
                'repz-navigation__link repz-navigation__link--sidebar',
                isActivePath(item.href) && 'repz-navigation__link--active'
              )}
            >
              {item.icon && <span className="repz-navigation__icon">{item.icon}</span>}
              {item.label}
            </a>
          ))}
        </div>

        {showAuth && isAuthenticated && (
          <div className="repz-navigation__user repz-navigation__user--sidebar">
            <div className="repz-navigation__user-info">
              <User size={20} />
              <div>
                <div className="repz-navigation__user-email">{user?.email}</div>
                <div className="repz-navigation__user-role">{user?.role || 'User'}</div>
              </div>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleSignOut}
              className="repz-navigation__signout"
            >
              <LogOut size={16} />
              Sign Out
            </Button>
          </div>
        )}
      </nav>
    );
  }

  // Default header variant
  return (
    <nav
      className={cn('repz-navigation repz-navigation--header', className)}
      ref={ref}
      {...props}
    >
      <div className="repz-navigation__container">
        <div className="repz-navigation__brand">
          <RepzLogo size="md" />
        </div>

        <div className="repz-navigation__items repz-navigation__items--header">
          {getVisibleItems().map((item) => (
            <a
              key={item.href}
              href={item.href}
              onClick={(e) => {
                e.preventDefault();
                navigate(item.href);
              }}
              className={cn(
                'repz-navigation__link',
                isActivePath(item.href) && 'repz-navigation__link--active'
              )}
            >
              {item.label}
            </a>
          ))}
        </div>

        {showAuth && (
          <div className="repz-navigation__auth">
            {isAuthenticated ? (
              <div className="repz-navigation__user">
                <span className="repz-navigation__user-email">{user?.email}</span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleSignOut}
                >
                  Sign Out
                </Button>
              </div>
            ) : (
              <div className="repz-navigation__auth-buttons">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => navigate('/login')}
                >
                  Sign In
                </Button>
                <Button
                  variant="accent"
                  size="sm"
                  onClick={() => navigate('/signup')}
                >
                  Sign Up
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </nav>
  );
});

Navigation.displayName = 'Navigation';