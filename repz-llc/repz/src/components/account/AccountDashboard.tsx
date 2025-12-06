import React, { useState, useEffect } from 'react';
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs";
import { Calendar, Download, ExternalLink } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { AccountOverview } from './components/AccountOverview';
import { SubscriptionManagement } from './components/SubscriptionManagement';

interface Subscription {
  id: string;
  status: string;
  billing_period: string;
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
  pricing_plans: {
    display_name: string;
    description: string;
    features: Record<string, unknown> | string[] | null;
    metadata: Record<string, unknown> | null;
  };
}

interface Order {
  id: string;
  status: string;
  amount_cents: number;
  created_at: string;
  pricing_plans: {
    display_name: string;
    description: string;
  };
}

export const AccountDashboard: React.FC = () => {
  const { user } = useAuth();
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      fetchAccountData();
    }
  }, [user]);

  const fetchAccountData = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase.functions.invoke('subscription-management', {
        body: { action: 'get_status' }
      });

      if (error) throw error;

      setSubscriptions(data.subscriptions || []);
      setOrders(data.orders || []);
    } catch (error) {
      console.error('Error fetching account data:', error);
      toast.error('Failed to load account information');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelSubscription = async (subscriptionId: string) => {
    if (!confirm('Are you sure you want to cancel this subscription? It will remain active until the end of your current billing period.')) {
      return;
    }

    try {
      setActionLoading(`cancel-${subscriptionId}`);
      const { data, error } = await supabase.functions.invoke('subscription-management', {
        body: { 
          action: 'cancel_subscription',
          subscription_id: subscriptionId
        }
      });

      if (error) throw error;

      toast.success('Subscription cancelled successfully');
      fetchAccountData(); // Refresh data
    } catch (error) {
      console.error('Error cancelling subscription:', error);
      toast.error('Failed to cancel subscription');
    } finally {
      setActionLoading(null);
    }
  };

  const handleManageBilling = async () => {
    try {
      setActionLoading('portal');
      const { data, error } = await supabase.functions.invoke('subscription-management', {
        body: { action: 'create_portal_session' }
      });

      if (error) throw error;

      window.open(data.url, '_blank');
    } catch (error) {
      console.error('Error opening billing portal:', error);
      toast.error('Failed to open billing portal');
    } finally {
      setActionLoading(null);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const formatPrice = (cents: number) => {
    return `$${(cents / 100).toFixed(2)}`;
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      active: { variant: 'default' as const, label: 'Active' },
      canceled: { variant: 'secondary' as const, label: 'Cancelled' },
      past_due: { variant: 'destructive' as const, label: 'Past Due' },
      trialing: { variant: 'outline' as const, label: 'Trial' },
      paid: { variant: 'default' as const, label: 'Paid' },
      pending: { variant: 'secondary' as const, label: 'Pending' },
      failed: { variant: 'destructive' as const, label: 'Failed' }
    };

    const config = statusConfig[status as keyof typeof statusConfig] || { variant: 'outline' as const, label: status };
    
    return (
      <Badge variant={config.variant}>
        {config.label}
      </Badge>
    );
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-foreground mb-4">Access Denied</h2>
          <p className="text-muted-foreground mb-6">Please log in to view your account dashboard.</p>
          <Button onClick={() => window.location.href = '/login'}>
            Log In
          </Button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading account information...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">Account Dashboard</h1>
          <p className="text-muted-foreground">Manage your subscriptions, billing, and account settings</p>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="subscriptions">Subscriptions</TabsTrigger>
            <TabsTrigger value="billing">Billing History</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <AccountOverview 
              subscriptions={subscriptions}
              orders={orders}
              onManageBilling={handleManageBilling}
              actionLoading={actionLoading}
            />
          </TabsContent>

          <TabsContent value="subscriptions">
            <SubscriptionManagement 
              subscriptions={subscriptions}
              onCancelSubscription={handleCancelSubscription}
              onManageBilling={handleManageBilling}
              actionLoading={actionLoading}
            />
          </TabsContent>

          <TabsContent value="billing">
            <Card>
              <CardHeader>
                <CardTitle>Billing History</CardTitle>
                <p className="text-muted-foreground">View all your past orders and payments</p>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {orders.map((order) => (
                    <div key={order.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <Calendar className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="font-medium">{order.pricing_plans.display_name}</p>
                          <p className="text-sm text-muted-foreground">{order.pricing_plans.description}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-semibold">{formatPrice(order.amount_cents)}</p>
                        <div className="flex items-center gap-2 mt-1">
                          {getStatusBadge(order.status)}
                          <Button onClick={() => console.log("Button clicked")} variant="ghost" size="sm">
                            <Download className="h-4 w-4" />
                          </Button>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {formatDate(order.created_at)}
                        </p>
                      </div>
                    </div>
                  ))}

                  {orders.length === 0 && (
                    <div className="text-center py-8 text-muted-foreground">
                      <p>No billing history found</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="settings">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Account Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Email</label>
                      <p className="font-semibold">{user?.email}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Account Created</label>
                      <p className="font-semibold">{formatDate(new Date().toISOString())}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Billing Management</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <Button 
                      onClick={handleManageBilling}
                      disabled={actionLoading === 'portal'}
                      className="w-full md:w-auto"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      {actionLoading === 'portal' ? 'Loading...' : 'Open Customer Portal'}
                    </Button>
                    <p className="text-sm text-muted-foreground">
                      Manage your payment methods, download invoices, and update billing information.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};