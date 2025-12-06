import React, { useState } from 'react';
import { Button } from '@/ui/atoms/Button';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { AuthModal } from './AuthModal';
import { User, LogIn, UserPlus } from 'lucide-react';

interface AuthButtonProps {
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}

export const AuthButton: React.FC<AuthButtonProps> = ({
  variant = 'default',
  size = 'default',
  className = ''
}) => {
  const { isAuthenticated, user } = useAuth();
  const navigate = useNavigate();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');

  const handleAuthClick = (mode: 'login' | 'signup') => {
    setAuthMode(mode);
    setShowAuthModal(true);
  };

  const handleDashboard = () => {
    navigate('/dashboard');
  };

  if (isAuthenticated) {
    return (
      <Button
        variant={variant}
        size={size}
        className={className}
        onClick={handleDashboard}
      >
        <User className="mr-2 h-4 w-4" />
        Go to Dashboard
      </Button>
    );
  }

  return (
    <>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size={size}
          className={className}
          onClick={() => handleAuthClick('login')}
        >
          <LogIn className="mr-2 h-4 w-4" />
          Sign In
        </Button>
        <Button
          variant={variant}
          size={size}
          className={className}
          onClick={() => handleAuthClick('signup')}
        >
          <UserPlus className="mr-2 h-4 w-4" />
          Sign Up
        </Button>
      </div>

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        mode={authMode}
      />
    </>
  );
};