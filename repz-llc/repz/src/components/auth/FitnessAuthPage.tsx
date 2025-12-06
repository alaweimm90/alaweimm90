import React, { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Label } from '@/ui/atoms/Label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { LoadingButton } from '@/components/ui/loading-button';
import { useAuth } from '@/contexts/AuthContext';
import { UserRole } from '@/types/fitness';
import { Dumbbell, Crown, Star, Zap } from 'lucide-react';

export const FitnessAuthPage = () => {
  const { signIn, signUp, user, loading } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');

  // Form states
  const [loginForm, setLoginForm] = useState({ email: '', password: '' });
  const [signupForm, setSignupForm] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: '',
    role: 'client' as UserRole
  });

  // Redirect if already authenticated
  if (user && !loading) {
    return user.role === 'coach' ? <Navigate to="/coach" replace /> : <Navigate to="/dashboard" replace />;
  }

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    const { error } = await signIn(loginForm.email, loginForm.password);
    
    if (error) {
      setError(error.message);
    }
    
    setIsLoading(false);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setSuccess('');

    if (signupForm.password !== signupForm.confirmPassword) {
      setError('Passwords do not match');
      setIsLoading(false);
      return;
    }

    if (signupForm.password.length < 6) {
      setError('Password must be at least 6 characters');
      setIsLoading(false);
      return;
    }

    const { error } = await signUp(
      signupForm.email,
      signupForm.password
    );

    if (error) {
      setError(error.message);
    } else {
      setSuccess('Account created successfully! Please check your email to verify your account.');
      setSignupForm({ email: '', password: '', confirmPassword: '', name: '', role: 'client' });
    }

    setIsLoading(false);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100">
        <div className="text-center">
          <Dumbbell className="h-12 w-12 animate-pulse mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Dumbbell className="h-10 w-10 text-primary mr-2" />
            <h1 className="text-3xl font-bold text-foreground">REPZ Elite</h1>
          </div>
          <p className="text-muted-foreground">Advanced Fitness Coaching Platform</p>
        </div>

        <Card className="shadow-xl border-0 bg-white/95 backdrop-blur">
          <Tabs defaultValue="login" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">Sign In</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>

            {/* Login Tab */}
            <TabsContent value="login">
              <form onSubmit={handleSignIn}>
                <CardHeader>
                  <CardTitle>Welcome Back</CardTitle>
                  <CardDescription>
                    Sign in to access your coaching platform
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="login-email">Email</Label>
                    <Input
                      id="login-email"
                      type="email"
                      placeholder="your@email.com"
                      value={loginForm.email}
                      onChange={(e) => setLoginForm(prev => ({ ...prev, email: e.target.value }))}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="login-password">Password</Label>
                    <Input
                      id="login-password"
                      type="password"
                      placeholder="••••••••"
                      value={loginForm.password}
                      onChange={(e) => setLoginForm(prev => ({ ...prev, password: e.target.value }))}
                      required
                    />
                  </div>
                </CardContent>
                <CardFooter>
                  <LoadingButton
                    type="submit"
                    className="w-full"
                    loading={isLoading}
                  >
                    Sign In
                  </LoadingButton>
                </CardFooter>
              </form>
            </TabsContent>

            {/* Signup Tab */}
            <TabsContent value="signup">
              <form onSubmit={handleSignUp}>
                <CardHeader>
                  <CardTitle>Create Account</CardTitle>
                  <CardDescription>
                    Join the elite fitness coaching platform
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="signup-name">Full Name</Label>
                    <Input
                      id="signup-name"
                      type="text"
                      placeholder="Your full name"
                      value={signupForm.name}
                      onChange={(e) => setSignupForm(prev => ({ ...prev, name: e.target.value }))}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="signup-email">Email</Label>
                    <Input
                      id="signup-email"
                      type="email"
                      placeholder="your@email.com"
                      value={signupForm.email}
                      onChange={(e) => setSignupForm(prev => ({ ...prev, email: e.target.value }))}
                      required
                    />
                  </div>

                  <div className="space-y-3">
                    <Label>Account Type</Label>
                    <RadioGroup
                      value={signupForm.role}
                      onValueChange={(value) => setSignupForm(prev => ({ ...prev, role: value as UserRole }))}
                    >
                      <div className="flex items-center space-x-2 p-3 border rounded-lg">
                        <RadioGroupItem value="client" id="client" />
                        <Label htmlFor="client" className="flex items-center cursor-pointer">
                          <Star className="h-4 w-4 mr-2 text-blue-500" />
                          Client - Get coached
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2 p-3 border rounded-lg">
                        <RadioGroupItem value="coach" id="coach" />
                        <Label htmlFor="coach" className="flex items-center cursor-pointer">
                          <Crown className="h-4 w-4 mr-2 text-amber-500" />
                          Coach - Manage clients
                        </Label>
                      </div>
                    </RadioGroup>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="signup-password">Password</Label>
                    <Input
                      id="signup-password"
                      type="password"
                      placeholder="••••••••"
                      value={signupForm.password}
                      onChange={(e) => setSignupForm(prev => ({ ...prev, password: e.target.value }))}
                      required
                      minLength={6}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="confirm-password">Confirm Password</Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      placeholder="••••••••"
                      value={signupForm.confirmPassword}
                      onChange={(e) => setSignupForm(prev => ({ ...prev, confirmPassword: e.target.value }))}
                      required
                      minLength={6}
                    />
                  </div>
                </CardContent>
                <CardFooter>
                  <LoadingButton
                    type="submit"
                    className="w-full"
                    loading={isLoading}
                  >
                    Create Account
                  </LoadingButton>
                </CardFooter>
              </form>
            </TabsContent>
          </Tabs>

          {/* Error/Success Messages */}
          {error && (
            <div className="px-6 pb-6">
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            </div>
          )}

          {success && (
            <div className="px-6 pb-6">
              <Alert>
                <Zap className="h-4 w-4" />
                <AlertDescription>{success}</AlertDescription>
              </Alert>
            </div>
          )}
        </Card>

        {/* Tier Preview */}
        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground mb-4">Available Subscription Tiers</p>
          <div className="flex justify-center space-x-2">
            <div className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">🔵 Core $96</div>
            <div className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs">🟢 Adaptive $178</div>
            <div className="px-3 py-1 bg-repz-orange/10 text-repz-orange rounded-full text-xs">🟣 Performance $298</div>
            <div className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs">🟡 Longevity $396</div>
          </div>
        </div>
      </div>
    </div>
  );
};