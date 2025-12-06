import React from 'react';
import { ClientProfile } from '@/types/fitness';
import { Activity, Clock } from 'lucide-react';
import { Badge } from '@/ui/atoms/Badge';
// Import removed - these functions don't exist in useTierAccess
import { useComponentPerformance } from '@/utils/monitoring';
import { ResponsiveHeader } from '@/components/layout/ResponsiveHeader';
import { TierType } from '@/types/fitness';

// Local helper functions to replace missing imports
const getTierStyling = (tier: TierType) => {
  switch (tier) {
    case 'core':
      return { 
        badge: 'bg-blue-100 text-blue-700 border-blue-200',
        border: 'border-blue-500'
      };
    case 'adaptive':
      return { 
        badge: 'bg-orange-100 text-orange-700 border-orange-200',
        border: 'border-orange-500'
      };
    case 'performance':  
      return { 
        badge: 'bg-purple-100 text-purple-700 border-purple-200',
        border: 'border-purple-500'
      };
    case 'longevity':
      return { 
        badge: 'bg-yellow-100 text-yellow-700 border-yellow-200',
        border: 'border-yellow-500'
      };
    default:
      return { 
        badge: 'bg-gray-100 text-gray-700 border-gray-200',
        border: 'border-gray-500'
      };
  }
};

const getTierIcon = (tier: TierType) => {
  switch (tier) {
    case 'core': return '🌱';
    case 'adaptive': return '📈';
    case 'performance': return '⚡';
    case 'longevity': return '👑';
    default: return '🌱';
  }
};

interface DashboardHeaderProps {
  clientProfile: ClientProfile;
}

export const DashboardHeader: React.FC<DashboardHeaderProps> = ({ clientProfile }) => {
  useComponentPerformance('DashboardHeader');
  
  const tierStyling = getTierStyling(clientProfile.subscription_tier);
  const tierFeatures = clientProfile.tier_features as any;

  return (
    <>
      <ResponsiveHeader variant="dashboard">
        <div className="hidden md:flex items-center space-x-4">
          <Badge variant="outline" className={tierStyling.badge}>
            {clientProfile.subscription_tier} Plan
          </Badge>
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Response Time</p>
            <p className="font-semibold">
              <Clock className="inline h-4 w-4 mr-1" />
              {tierFeatures?.response_time_hours || '24'}h
            </p>
          </div>
        </div>
      </ResponsiveHeader>
      
      {/* Welcome Section */}
      <div className={`border-l-4 ${tierStyling.border} bg-muted/50 px-4 py-6`}>
        <div className="container mx-auto">
          <div className="flex items-center gap-3 mb-2">
            <div className="text-3xl">{getTierIcon(clientProfile.subscription_tier)}</div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-foreground">
                Welcome back, {clientProfile.client_name}
              </h1>
              <div className="md:hidden mt-2">
                <Badge variant="outline" className={tierStyling.badge}>
                  {clientProfile.subscription_tier} Plan
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};