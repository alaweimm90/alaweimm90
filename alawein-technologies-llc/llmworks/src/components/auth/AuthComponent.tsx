import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuth } from "@/hooks/useAuth";
import { useState } from "react";
import { Shield, Mail, Lock, User, LogOut } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export function AuthComponent() {
  const { user, loading, signIn, signUp, signOut } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);

  const [signInForm, setSignInForm] = useState({
    email: "",
    password: "",
  });

  const [signUpForm, setSignUpForm] = useState({
    email: "",
    password: "",
    displayName: "",
  });

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    const { error } = await signIn(signInForm.email, signInForm.password);
    
    if (error) {
      toast({
        title: "Sign in failed",
        description: error.message,
        variant: "destructive",
      });
    } else {
      toast({
        title: "Welcome back!",
        description: "You have successfully signed in.",
      });
    }
    setIsLoading(false);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    const { error } = await signUp(signUpForm.email, signUpForm.password, {
      display_name: signUpForm.displayName,
    });
    
    if (error) {
      toast({
        title: "Sign up failed",
        description: error.message,
        variant: "destructive",
      });
    } else {
      toast({
        title: "Account created!",
        description: "Please check your email to verify your account.",
      });
    }
    setIsLoading(false);
  };

  const handleSignOut = async () => {
    const { error } = await signOut();
    if (error) {
      toast({
        title: "Sign out failed",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  if (user) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-lg bg-primary/10">
            <User className="h-8 w-8 text-primary" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-bold text-foreground">Welcome back!</h3>
            <p className="text-muted-foreground">{user.email}</p>
          </div>
          <Button onClick={handleSignOut} variant="outline" size="sm">
            <LogOut className="h-4 w-4 mr-2" />
            Sign Out
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 max-w-md mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-lg bg-primary/10">
          <Shield className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-foreground">Authentication</h2>
          <p className="text-sm text-muted-foreground">Optional — enables sync & cloud features (planned)</p>
        </div>
      </div>

      <Tabs defaultValue="signin" className="w-full">
        <TabsList className="grid w-full grid-cols-2 mb-6">
          <TabsTrigger value="signin">Sign In</TabsTrigger>
          <TabsTrigger value="signup">Sign Up</TabsTrigger>
        </TabsList>

        <TabsContent value="signin">
          <form onSubmit={handleSignIn} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="signin-email">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="signin-email"
                  type="email"
                  placeholder="Enter your email"
                  className="pl-10"
                  value={signInForm.email}
                  onChange={(e) => setSignInForm(prev => ({ ...prev, email: e.target.value }))}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="signin-password">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="signin-password"
                  type="password"
                  placeholder="Enter your password"
                  className="pl-10"
                  value={signInForm.password}
                  onChange={(e) => setSignInForm(prev => ({ ...prev, password: e.target.value }))}
                  required
                />
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "Signing in..." : "Sign In"}
            </Button>
          </form>
        </TabsContent>

        <TabsContent value="signup">
          <form onSubmit={handleSignUp} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="signup-name">Display Name</Label>
              <div className="relative">
                <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="signup-name"
                  type="text"
                  placeholder="Enter your name"
                  className="pl-10"
                  value={signUpForm.displayName}
                  onChange={(e) => setSignUpForm(prev => ({ ...prev, displayName: e.target.value }))}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="signup-email">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="signup-email"
                  type="email"
                  placeholder="Enter your email"
                  className="pl-10"
                  value={signUpForm.email}
                  onChange={(e) => setSignUpForm(prev => ({ ...prev, email: e.target.value }))}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="signup-password">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="signup-password"
                  type="password"
                  placeholder="Create a password"
                  className="pl-10"
                  value={signUpForm.password}
                  onChange={(e) => setSignUpForm(prev => ({ ...prev, password: e.target.value }))}
                  required
                />
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "Creating account..." : "Create Account"}
            </Button>
          </form>
        </TabsContent>
      </Tabs>
    </Card>
  );
}