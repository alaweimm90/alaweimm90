import { useState } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { useNavigate } from 'react-router-dom';
import { useSEO } from '@/hooks/useSEO';
import LazyBackground from '@/components/dev/LazyBackground';

const Auth = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  useSEO({
    title: mode === 'signin' ? 'Sign in (optional) — Attributa.dev' : 'Create account (optional) — Attributa.dev',
    description: 'Sign in only to sync settings. Guest mode works fully locally.',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (mode === 'signup') {
        const redirectTo = `${window.location.origin}/`;
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: { emailRedirectTo: redirectTo }
        });
        if (error) throw error;
        toast({ title: 'Check your email', description: 'Confirm your email to finish sign up.' });
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        toast({ title: 'Signed in', description: 'Welcome back!' });
        navigate('/settings');
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Please try again.';
      toast({ title: 'Auth error', description: message, variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary engineering-pattern">
      <LazyBackground />
      
      <div className="relative z-10 container mx-auto px-4 py-8 max-w-md">
        <div className="animate-fade-in">
          <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold tracking-tight">{mode === 'signin' ? 'Sign in' : 'Create account'}</CardTitle>
          <CardDescription>Sign in is optional. Continue as guest for local use.</CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="default" className="w-full mb-4" type="button" onClick={() => navigate('/settings')}>
            Continue as guest (local only)
          </Button>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="email" className="text-sm">Email</label>
              <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
            <div>
              <label htmlFor="password" className="text-sm">Password</label>
              <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Please wait…' : mode === 'signin' ? 'Sign in' : 'Sign up'}
            </Button>
          </form>
          <div className="mt-4 text-center text-sm">
            {mode === 'signin' ? (
              <button className="underline" onClick={() => setMode('signup')}>Need an account? Sign up</button>
            ) : (
              <button className="underline" onClick={() => setMode('signin')}>Have an account? Sign in</button>
            )}
          </div>
        </CardContent>
      </Card>
        </div>
      </div>
    </div>
  );
};

export default Auth;
