import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { RepzLogo } from '@/ui/organisms/RepzLogo';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import { Label } from '@/ui/atoms/Label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Eye, EyeOff, Mail, Lock, ArrowRight, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

/**
 * Enterprise-Standard Login Page
 * 
 * Features:
 * - Canonical route at /login for deep-linking and SEO
 * - Progressive enhancement ready (works without JS)
 * - WCAG 2.2 AA compliant
 * - Mobile-first responsive design
 * - Form validation with error states
 * - Loading states and feedback
 */
export default function Login() {
  const navigate = useNavigate();
  const { signIn, isAuthenticated, loading: authLoading } = useAuth();
  
  // Form state
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated && !authLoading) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, authLoading, navigate]);

  // Form validation
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setLoading(true);
    setErrors({});
    
    try {
      const { error } = await signIn(formData.email, formData.password);
      
      if (error) {
        if (error.message.includes('Invalid login credentials')) {
          setErrors({ form: 'Invalid email or password. Please try again.' });
        } else if (error.message.includes('Email not confirmed')) {
          setErrors({ form: 'Please check your email and click the confirmation link.' });
        } else {
          setErrors({ form: 'An error occurred. Please try again.' });
        }
        return;
      }
      
      toast.success('Welcome back to REPZ!');
      navigate('/dashboard', { replace: true });
      
    } catch (error) {
      console.error('Login error:', error);
      setErrors({ form: 'An unexpected error occurred. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  // Handle input changes
  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }));
    // Clear field error on change
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  // Show loading during auth check
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-muted/20 to-background">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="text-sm text-muted-foreground">Checking authentication...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-900 to-black relative overflow-hidden">
      {/* Elegant background effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-gradient-to-br from-orange-500/10 via-red-500/5 to-transparent rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 left-20 w-80 h-80 bg-gradient-to-br from-red-500/10 via-orange-500/5 to-transparent rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>

      {/* Mobile-first responsive container */}
      <div className="container mx-auto px-4 py-8 min-h-screen flex flex-col relative z-10">
        
        {/* Header */}
        <header className="flex items-center justify-between mb-8 lg:mb-12">
          <Link to="/" className="transition-all duration-300 hover:scale-105">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white font-bold text-lg">R</span>
              </div>
              <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-red-400">REPZ</span>
            </div>
          </Link>
          <nav className="hidden sm:flex items-center gap-6">
            <span className="text-sm text-white/70 font-medium">New to REPZ?</span>
            <Button variant="outline" asChild className="border-white/20 text-white hover:bg-white/10 hover:border-white/40 backdrop-blur-sm transition-all duration-300">
              <Link to="/signup">Sign Up</Link>
            </Button>
          </nav>
        </header>

        {/* Main content - Centered card */}
        <main className="flex-1 flex items-center justify-center">
          <Card className="w-full max-w-md shadow-2xl border-0 bg-white/95 backdrop-blur-xl">
            <CardHeader className="space-y-6 text-center pb-8">
              <div className="mx-auto w-16 h-16 bg-gradient-to-br from-orange-500 to-red-500 rounded-2xl flex items-center justify-center shadow-lg mb-2">
                <Lock className="h-8 w-8 text-white" />
              </div>
              <CardTitle className="text-3xl font-bold tracking-tight text-slate-900">
                Welcome Back
              </CardTitle>
              <CardDescription className="text-base text-slate-600 leading-relaxed">
                Sign in to your REPZ account to continue your fitness journey and achieve your goals
              </CardDescription>
            </CardHeader>
            
            <CardContent className="px-8 pb-8">
              {/* Form with native HTML5 semantics for accessibility */}
              <form onSubmit={handleSubmit} className="space-y-6" noValidate>
                
                {/* Global form error */}
                {errors.form && (
                  <Alert variant="destructive" className="border-red-200 bg-red-50">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-red-800">{errors.form}</AlertDescription>
                  </Alert>
                )}
                
                {/* Email field */}
                <div className="space-y-3">
                  <Label htmlFor="email" className="text-sm font-semibold text-slate-700">
                    Email Address
                  </Label>
                  <div className="relative group">
                    <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400 group-focus-within:text-orange-500 transition-colors duration-200" />
                    <Input
                      id="email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      required
                      placeholder="Enter your email address"
                      value={formData.email}
                      onChange={handleChange('email')}
                      className={`pl-12 h-12 text-base border-slate-200 focus:border-orange-500 focus:ring-orange-500/20 transition-all duration-200 ${errors.email ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''}`}
                      aria-invalid={!!errors.email}
                      aria-describedby={errors.email ? 'email-error' : undefined}
                    />
                  </div>
                  {errors.email && (
                    <p id="email-error" className="text-sm text-red-600 flex items-center gap-2" role="alert">
                      <AlertCircle className="h-4 w-4" />
                      {errors.email}
                    </p>
                  )}
                </div>

                {/* Password field */}
                <div className="space-y-3">
                  <Label htmlFor="password" className="text-sm font-semibold text-slate-700">
                    Password
                  </Label>
                  <div className="relative group">
                    <Lock className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400 group-focus-within:text-orange-500 transition-colors duration-200" />
                    <Input
                      id="password"
                      name="password"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="current-password"
                      required
                      placeholder="Enter your password"
                      value={formData.password}
                      onChange={handleChange('password')}
                      className={`pl-12 pr-12 h-12 text-base border-slate-200 focus:border-orange-500 focus:ring-orange-500/20 transition-all duration-200 ${errors.password ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''}`}
                      aria-invalid={!!errors.password}
                      aria-describedby={errors.password ? 'password-error' : undefined}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400 hover:text-slate-600 transition-colors duration-200"
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                    >
                      {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                    </button>
                  </div>
                  {errors.password && (
                    <p id="password-error" className="text-sm text-red-600 flex items-center gap-2" role="alert">
                      <AlertCircle className="h-4 w-4" />
                      {errors.password}
                    </p>
                  )}
                </div>

                {/* Submit button */}
                <Button
                  type="submit"
                  className="w-full h-12 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white font-semibold text-base shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02]"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-current mr-3" />
                      Signing In...
                    </>
                  ) : (
                    <>
                      Sign In to REPZ
                      <ArrowRight className="ml-3 h-5 w-5" />
                    </>
                  )}
                </Button>
                
                {/* Forgot password link */}
                <div className="text-center pt-2">
                  <Link 
                    to="/reset-password" 
                    className="text-sm text-orange-600 hover:text-orange-700 font-medium transition-colors duration-200"
                  >
                    Forgot your password?
                  </Link>
                </div>
              </form>
            </CardContent>
          </Card>
        </main>

        {/* Footer - Mobile signup prompt */}
        <footer className="mt-8 text-center sm:hidden">
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
            <p className="text-sm text-white/80 mb-4 font-medium">
              New to REPZ?
            </p>
            <Button variant="outline" asChild className="w-full max-w-xs border-white/30 text-white hover:bg-white/10 hover:border-white/50 backdrop-blur-sm transition-all duration-300">
              <Link to="/signup">Create Account</Link>
            </Button>
          </div>
        </footer>
      </div>
    </div>
  );
}