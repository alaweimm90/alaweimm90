import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Zap, History, Settings, CreditCard, ExternalLink } from 'lucide-react';

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

interface AccountOverviewProps {
  subscriptions: Subscription[];
  orders: Order[];
  onManageBilling: () => void;
  actionLoading: string | null;
}

export const AccountOverview: React.FC<AccountOverviewProps> = ({
  subscriptions,
  orders,
  onManageBilling,
  actionLoading
}) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
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

    return statusConfig[status as keyof typeof statusConfig] || { variant: 'outline' as const, label: status };
  };

  return (
    <div className="space-y-8">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Subscriptions</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{subscriptions.filter(s => s.status === 'active').length}</div>
            <p className="text-xs text-muted-foreground">
              {subscriptions.length > 0 ? 'Subscription active' : 'No active subscriptions'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Orders</CardTitle>
            <History className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{orders.length}</div>
            <p className="text-xs text-muted-foreground">
              All-time purchases
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Quick Actions</CardTitle>
            <Settings className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="w-full"
              onClick={onManageBilling}
              disabled={actionLoading === 'portal'}
            >
              <CreditCard className="w-4 h-4 mr-2" />
              {actionLoading === 'portal' ? 'Loading...' : 'Manage Billing'}
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              className="w-full"
              onClick={() => window.location.href = '/pricing'}
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              View Plans
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...subscriptions.slice(0, 2), ...orders.slice(0, 3)].map((item, index) => {
              const config = getStatusBadge('status' in item ? item.status : 'unknown');
              return (
                <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="w-2 h-2 bg-primary rounded-full"></div>
                    <div>
                      <p className="font-medium">
                        {'pricing_plans' in item ? item.pricing_plans.display_name : 'Unknown Plan'}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {'current_period_start' in item 
                          ? `Subscription ${item.status}`
                          : `Order ${item.status}`
                        }
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      config.variant === 'default' ? 'bg-primary text-primary-foreground' :
                      config.variant === 'secondary' ? 'bg-secondary text-secondary-foreground' :
                      config.variant === 'destructive' ? 'bg-destructive text-destructive-foreground' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {config.label}
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {'current_period_start' in item 
                        ? formatDate(item.current_period_start)
                        : formatDate(item.created_at)
                      }
                    </p>
                  </div>
                </div>
              );
            })}
            
            {subscriptions.length === 0 && orders.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <p>No recent activity</p>
                <Button 
                  variant="outline" 
                  className="mt-4"
                  onClick={() => window.location.href = '/pricing'}
                >
                  Browse Plans
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};