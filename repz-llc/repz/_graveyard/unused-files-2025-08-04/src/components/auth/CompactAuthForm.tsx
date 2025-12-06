import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { CollapsibleAuthPanel } from './CollapsibleAuthPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

type ActivePanel = 'login' | 'signup' | 'methods' | '';

export const CompactAuthForm: React.FC = () => {
  const [activePanel, setActivePanel] = useState<ActivePanel>('login');
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    role: 'client' as 'client' | 'coach'
  });
  const [loading, setLoading] = useState(false);
  const { signIn, signUp, isAuthenticated, loading: authLoading } = useAuth();
  const navigate = useNavigate();

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated && !authLoading) {
      console.log('[AUTH] User already authenticated, redirecting to dashboard');
      navigate('/dashboard');
    }
  }, [isAuthenticated, authLoading, navigate]);

  // Reset loading state when auth state changes
  useEffect(() => {
    if (isAuthenticated) {
      setLoading(false);
    }
  }, [isAuthenticated]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSignIn = async () => {
    if (!formData.email || !formData.password) {
      toast.error('Please fill in all fields');
      return;
    }

    if (loading) return; // Prevent double submission

    setLoading(true);
    try {
      console.log('[LOGIN FORM] Submitting login for:', formData.email);
      
      // Add timeout protection
      const loginPromise = signIn(formData.email, formData.password);
      const timeoutPromise = new Promise<{ error: Error }>((_, reject) => 
        setTimeout(() => reject(new Error('Login timeout after 15 seconds')), 15000)
      );
      
      const result = await Promise.race([loginPromise, timeoutPromise]);
      
      if (result.error) {
        console.error('[LOGIN FORM] Login failed:', result.error);
        toast.error(result.error.message);
        setLoading(false);
      } else {
        console.log('[LOGIN FORM] Login successful');
        toast.success('Signed in successfully!');
        // setLoading will be reset by useEffect when isAuthenticated becomes true
      }
    } catch (err) {
      console.error('[LOGIN FORM] Login error:', err);
      toast.error(err instanceof Error ? err.message : 'An unexpected error occurred');
      setLoading(false);
    }
  };

  const handleSignUp = async () => {
    if (!formData.email || !formData.password || !formData.name) {
      toast.error('Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      const { error } = await signUp(formData.email, formData.password, formData.role, formData.name);
      if (error) {
        toast.error(error.message);
      } else {
        toast.success('Account created successfully!');
        navigate('/dashboard');
      }
    } catch (err) {
      toast.error('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const togglePanel = (panel: ActivePanel) => {
    setActivePanel(activePanel === panel ? '' : panel);
  };

  return (
    <div className="auth-container max-w-md mx-auto p-4 space-y-4">
      {/* Quick Navigation Pills */}
      <div className="flex flex-wrap gap-2 mb-4">
        <Badge 
          variant={activePanel === 'login' ? 'default' : 'secondary'}
          className="px-4 py-2 cursor-pointer"
          onClick={() => togglePanel('login')}
        >
          Sign In
        </Badge>
        <Badge 
          variant={activePanel === 'signup' ? 'default' : 'secondary'}
          className="px-4 py-2 cursor-pointer"
          onClick={() => togglePanel('signup')}
        >
          Sign Up
        </Badge>
        <Badge 
          variant={activePanel === 'methods' ? 'default' : 'secondary'}
          className="px-4 py-2 cursor-pointer"
          onClick={() => togglePanel('methods')}
        >
          Other Methods
        </Badge>
      </div>

      {/* Login Panel */}
      <CollapsibleAuthPanel 
        isOpen={activePanel === 'login'} 
        onToggle={() => togglePanel('login')}
        title="Sign In to REPZ"
      >
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="login-email">Email</Label>
            <Input 
              id="login-email"
              type="email" 
              placeholder="Enter your email"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              className="text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="login-password">Password</Label>
            <Input 
              id="login-password"
              type="password" 
              placeholder="Enter your password"
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              className="text-sm"
            />
          </div>
           <Button 
            onClick={handleSignIn}
            disabled={loading || authLoading}
            className="w-full"
            size="sm"
          >
            {loading || authLoading ? 'Signing In...' : 'Sign In'}
          </Button>
          
          {(loading || authLoading) && (
            <div className="text-xs text-muted-foreground text-center mt-2">
              If this takes more than 15 seconds, please refresh and try again.
            </div>
          )}
        </div>
      </CollapsibleAuthPanel>

      {/* Signup Panel */}
      <CollapsibleAuthPanel 
        isOpen={activePanel === 'signup'} 
        onToggle={() => togglePanel('signup')}
        title="Create REPZ Account"
      >
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="signup-name">Full Name</Label>
            <Input 
              id="signup-name"
              type="text" 
              placeholder="Enter your full name"
              value={formData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
              className="text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="signup-email">Email</Label>
            <Input 
              id="signup-email"
              type="email" 
              placeholder="Enter your email"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              className="text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="signup-password">Password</Label>
            <Input 
              id="signup-password"
              type="password" 
              placeholder="Create a password"
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              className="text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label>Account Type</Label>
            <div className="flex gap-2">
              <Badge 
                variant={formData.role === 'client' ? 'default' : 'secondary'}
                className="cursor-pointer"
                onClick={() => handleInputChange('role', 'client')}
              >
                Client
              </Badge>
              <Badge 
                variant={formData.role === 'coach' ? 'default' : 'secondary'}
                className="cursor-pointer"
                onClick={() => handleInputChange('role', 'coach')}
              >
                Coach
              </Badge>
            </div>
          </div>
          <Button 
            onClick={handleSignUp}
            disabled={loading}
            className="w-full"
            size="sm"
          >
            {loading ? 'Creating Account...' : 'Sign Up'}
          </Button>
        </div>
      </CollapsibleAuthPanel>

      {/* Alternative Methods */}
      <CollapsibleAuthPanel 
        isOpen={activePanel === 'methods'} 
        onToggle={() => togglePanel('methods')}
        title="Other Sign In Methods"
      >
        <div className="space-y-2">
          <Button 
            variant="outline" 
            className="w-full justify-start text-sm"
            onClick={() => toast.info('Google auth coming soon!')}
          >
            <span className="mr-2">🔗</span>
            Continue with Google
          </Button>
          <Button 
            variant="outline" 
            className="w-full justify-start text-sm"
            onClick={() => toast.info('Phone auth coming soon!')}
          >
            <span className="mr-2">📱</span>
            Continue with Phone
          </Button>
        </div>
      </CollapsibleAuthPanel>

      {/* Quick Navigation to Homepage */}
      <div className="mt-6 pt-4 border-t border-border">
        {isAuthenticated ? (
          <Button 
            variant="link" 
            className="w-full text-primary"
            onClick={() => navigate('/dashboard')}
          >
            → Go to Dashboard
          </Button>
        ) : (
          <Button 
            variant="link" 
            className="w-full text-primary"
            onClick={() => navigate('/')}
          >
            → Go to Homepage
          </Button>
        )}
      </div>
    </div>
  );
};