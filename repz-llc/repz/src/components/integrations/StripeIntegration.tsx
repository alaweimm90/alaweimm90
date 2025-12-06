import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { CreditCard, CheckCircle, Clock, DollarSign } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useTierAccess } from '@/hooks/useTierAccess';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

const TIER_PRICING = {
  core: 89,
  adaptive: 149,
  performance: 229,
  longevity: 349
};

export function StripeIntegration() {
  const { user } = useAuth();
  const { userTier } = useTierAccess();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState<string | null>(null);

  const handleSubscribe = async (tier: keyof typeof TIER_PRICING) => {
    if (!user) {
      toast({
        title: 'Authentication Required',
        description: 'Please sign in to subscribe',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(tier);
    
    try {
      const { data, error } = await supabase.functions.invoke('create-checkout', {
        body: { 
          tier,
          price: TIER_PRICING[tier],
          returnUrl: window.location.origin + '/dashboard'
        }
      });

      if (error) throw error;

      if (data?.url) {
        window.open(data.url, '_blank');
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create checkout session';
      toast({
        title: 'Subscription Error',
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(null);
    }
  };

  const handleManageSubscription = async () => {
    setIsLoading('manage');
    
    try {
      const { data, error } = await supabase.functions.invoke('customer-portal', {
        body: { returnUrl: window.location.origin }
      });

      if (error) throw error;

      if (data?.url) {
        window.open(data.url, '_blank');
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to open customer portal';
      toast({
        title: 'Portal Error',
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(null);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CreditCard className="w-5 h-5 text-tier-adaptive" />
            Subscription Management
          </CardTitle>
          <CardDescription>
            Manage your REPZ subscription and billing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {userTier && (
            <div className="flex items-center gap-2 p-4 rounded-lg bg-tier-adaptive/10 border border-tier-adaptive/20">
              <CheckCircle className="w-5 h-5 text-tier-adaptive" />
              <div>
                <p className="font-medium">Current Plan: {userTier.toUpperCase()}</p>
                <p className="text-sm text-muted-foreground">
                  ${TIER_PRICING[userTier as keyof typeof TIER_PRICING]}/month
                </p>
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <Button 
              onClick={handleManageSubscription}
              disabled={isLoading === 'manage'}
              variant="outline"
              className="flex-1"
            >
              {isLoading === 'manage' ? (
                <Clock className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <CreditCard className="w-4 h-4 mr-2" />
              )}
              Manage Billing
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(TIER_PRICING).map(([tier, price]) => (
          <Card key={tier} className={`glass-tier-card ${userTier === tier ? 'ring-2 ring-tier-adaptive' : ''}`}>
            <CardHeader>
              <CardTitle className="capitalize text-lg">{tier}</CardTitle>
              <div className="flex items-baseline gap-1">
                <DollarSign className="w-4 h-4" />
                <span className="text-2xl font-bold">{price}</span>
                <span className="text-sm text-muted-foreground">/month</span>
              </div>
            </CardHeader>
            <CardContent>
              {userTier === tier ? (
                <Badge variant="default" className="w-full justify-center">
                  Current Plan
                </Badge>
              ) : (
                <Button
                  onClick={() => handleSubscribe(tier as keyof typeof TIER_PRICING)}
                  disabled={isLoading === tier}
                  className="w-full"
                  size="sm"
                >
                  {isLoading === tier ? (
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    'Subscribe'
                  )}
                </Button>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}