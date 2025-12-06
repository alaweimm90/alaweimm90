import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { Button } from '@/ui/atoms/Button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { RepzLogo } from '@/ui/organisms/RepzLogo';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';

export const MobileAuthNavigation: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const { user, signOut } = useAuth();

  const handleSignOut = async () => {
    try {
      await signOut();
      toast.success('Signed out successfully');
      navigate('/');
      setIsOpen(false);
    } catch (error) {
      toast.error('Error signing out');
    }
  };

  const navItems = [
    { label: 'Dashboard', href: '/dashboard' },
    { label: 'Progress', href: '/progress' },
    { label: 'Coaching', href: '/monthly-coaching' },
    { label: 'Messages', href: '/messages' },
    { label: 'Sessions', href: '/sessions' },
  ];

  return (
    <div className="mobile-nav md:hidden fixed top-0 left-0 right-0 bg-card shadow-md z-50 border-b border-border">
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center">
          <RepzLogo size="sm" />
        </div>
        
        <Sheet open={isOpen} onOpenChange={setIsOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="sm" className="p-2" onClick={() => console.log("MobileAuthNavigation button clicked")}>
              <Menu className="w-5 h-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-[280px]">
            <div className="flex flex-col h-full">
              <div className="flex items-center justify-between mb-6">
                <RepzLogo size="sm" />
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setIsOpen(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              {user ? (
                <div className="space-y-2 flex-1">
                  <div className="mb-4 p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium truncate">{user.email}</p>
                    <p className="text-xs text-muted-foreground">
                      {user.role || 'User'}
                    </p>
                  </div>
                  
                  {navItems.map((item) => (
                    <Button
                      key={item.href}
                      variant="ghost"
                      className="w-full justify-start"
                      onClick={() => {
                        navigate(item.href);
                        setIsOpen(false);
                      }}
                    >
                      {item.label}
                    </Button>
                  ))}
                  
                  <div className="pt-4 mt-auto border-t border-border">
                    <Button
                      variant="ghost"
                      className="w-full justify-start text-destructive"
                      onClick={handleSignOut}
                    >
                      Sign Out
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <Button
                    className="w-full"
                    onClick={() => {
                      navigate('/login');
                      setIsOpen(false);
                    }}
                  >
                    Sign In
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => {
                      navigate('/signup');
                      setIsOpen(false);
                    }}
                  >
                    Sign Up
                  </Button>
                  <Button
                    variant="ghost"
                    className="w-full"
                    onClick={() => {
                      navigate('/');
                      setIsOpen(false);
                    }}
                  >
                    View Plans
                  </Button>
                </div>
              )}
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </div>
  );
};