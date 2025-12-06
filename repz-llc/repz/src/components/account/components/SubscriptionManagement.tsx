import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { CreditCard } from 'lucide-react';

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

interface SubscriptionManagementProps {
  subscriptions: Subscription[];
  onCancelSubscription: (subscriptionId: string) => void;
  onManageBilling: () => void;
  actionLoading: string | null;
}

export const SubscriptionManagement: React.FC<SubscriptionManagementProps> = ({
  subscriptions,
  onCancelSubscription,
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
      trialing: { variant: 'outline' as const, label: 'Trial' }
    };

    const config = statusConfig[status as keyof typeof statusConfig] || { variant: 'outline' as const, label: status };
    
    return (
      <Badge variant={config.variant}>
        {config.label}
      </Badge>
    );
  };

  if (subscriptions.length === 0) {
    return (
      <Card>
        <CardContent className="text-center py-12">
          <h3 className="text-lg font-semibold mb-2">No Active Subscriptions</h3>
          <p className="text-muted-foreground mb-6">
            Choose a plan to get started with our premium features.
          </p>
          <Button onClick={() => window.location.href = '/pricing'}>
            Browse Plans
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {subscriptions.map((subscription) => (
        <Card key={subscription.id}>
          <CardHeader>
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="text-xl">
                  {subscription.pricing_plans.display_name}
                </CardTitle>
                <p className="text-muted-foreground mt-1">
                  {subscription.pricing_plans.description}
                </p>
              </div>
              {getStatusBadge(subscription.status)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Billing Period</p>
                <p className="font-semibold capitalize">{subscription.billing_period}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Current Period</p>
                <p className="font-semibold">
                  {formatDate(subscription.current_period_start)} - {formatDate(subscription.current_period_end)}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Response Time</p>
                <p className="font-semibold">{subscription.pricing_plans.metadata.response_time_hours} hours</p>
              </div>
            </div>

            {subscription.cancel_at_period_end && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 mb-4">
                <p className="text-destructive font-medium">
                  This subscription will be cancelled at the end of the current billing period.
                </p>
              </div>
            )}

            <div className="flex gap-2">
              <Button 
                variant="outline"
                onClick={onManageBilling}
                disabled={actionLoading === 'portal'}
              >
                <CreditCard className="w-4 h-4 mr-2" />
                Manage Billing
              </Button>
              
              {subscription.status === 'active' && !subscription.cancel_at_period_end && (
                <Button 
                  variant="destructive"
                  onClick={() => onCancelSubscription(subscription.id)}
                  disabled={actionLoading === `cancel-${subscription.id}`}
                >
                  {actionLoading === `cancel-${subscription.id}` ? 'Cancelling...' : 'Cancel Subscription'}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};