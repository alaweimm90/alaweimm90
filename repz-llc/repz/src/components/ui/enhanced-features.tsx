import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Switch } from '@/ui/atoms/Switch';
import { 
  Smartphone, Monitor, Palette, Zap, Shield, 
  Globe, Bell, Settings, Users, BarChart3 
} from 'lucide-react';

interface FeatureToggle {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  category: 'ui' | 'performance' | 'security' | 'analytics';
  icon: React.ReactNode;
}

export const EnhancedFeatures: React.FC = () => {
  const [features, setFeatures] = useState<FeatureToggle[]>([
    {
      id: 'dark-mode',
      name: 'Dark Mode',
      description: 'Toggle between light and dark themes',
      enabled: true,
      category: 'ui',
      icon: <Palette className="h-4 w-4" />
    },
    {
      id: 'mobile-opt',
      name: 'Mobile Optimization',
      description: 'Enhanced mobile experience with native features',
      enabled: true,
      category: 'performance',
      icon: <Smartphone className="h-4 w-4" />
    },
    {
      id: 'real-time',
      name: 'Real-time Updates',
      description: 'Live data synchronization across devices',
      enabled: true,
      category: 'performance',
      icon: <Zap className="h-4 w-4" />
    },
    {
      id: 'advanced-security',
      name: 'Advanced Security',
      description: 'Enhanced authentication and data protection',
      enabled: true,
      category: 'security',
      icon: <Shield className="h-4 w-4" />
    },
    {
      id: 'multi-language',
      name: 'Multi-language Support',
      description: 'Support for multiple languages and locales',
      enabled: false,
      category: 'ui',
      icon: <Globe className="h-4 w-4" />
    },
    {
      id: 'push-notifications',
      name: 'Push Notifications',
      description: 'Real-time notifications and alerts',
      enabled: true,
      category: 'ui',
      icon: <Bell className="h-4 w-4" />
    },
    {
      id: 'advanced-analytics',
      name: 'Advanced Analytics',
      description: 'Comprehensive user behavior tracking',
      enabled: true,
      category: 'analytics',
      icon: <BarChart3 className="h-4 w-4" />
    },
    {
      id: 'team-collaboration',
      name: 'Team Collaboration',
      description: 'Multi-user workspace and sharing',
      enabled: false,
      category: 'ui',
      icon: <Users className="h-4 w-4" />
    }
  ]);

  const toggleFeature = (featureId: string) => {
    setFeatures(prev => prev.map(feature => 
      feature.id === featureId 
        ? { ...feature, enabled: !feature.enabled }
        : feature
    ));
  };

  const getCategoryColor = (category: FeatureToggle['category']) => {
    switch (category) {
      case 'ui': return 'bg-blue-500';
      case 'performance': return 'bg-green-500';
      case 'security': return 'bg-red-500';
      case 'analytics': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };

  const enabledCount = features.filter(f => f.enabled).length;

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Enhanced Features</h1>
          <p className="text-muted-foreground mt-1">
            Manage advanced platform capabilities and integrations
          </p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold">{enabledCount}/{features.length}</div>
          <p className="text-sm text-muted-foreground">Features Active</p>
        </div>
      </div>

      {/* Feature Categories */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {['ui', 'performance', 'security', 'analytics'].map(category => {
          const categoryFeatures = features.filter(f => f.category === category);
          const activeCount = categoryFeatures.filter(f => f.enabled).length;
          
          return (
            <Card key={category}>
              <CardContent className="p-4 text-center">
                <div className={`w-8 h-8 rounded-full ${getCategoryColor(category as string)} mx-auto mb-2 flex items-center justify-center`}>
                  <Settings className="h-4 w-4 text-white" />
                </div>
                <div className="text-lg font-semibold capitalize">{category}</div>
                <div className="text-sm text-muted-foreground">
                  {activeCount}/{categoryFeatures.length} active
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Feature List */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map(feature => (
          <Card key={feature.id} className={`transition-all ${feature.enabled ? 'ring-2 ring-primary/20' : ''}`}>
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${getCategoryColor(feature.category)} text-white`}>
                    {feature.icon}
                  </div>
                  <div>
                    <CardTitle className="text-lg">{feature.name}</CardTitle>
                    <Badge 
                      variant={feature.enabled ? "default" : "secondary"} 
                      className="mt-1"
                    >
                      {feature.enabled ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </div>
                </div>
                <Switch
                  checked={feature.enabled}
                  onCheckedChange={() => toggleFeature(feature.id)}
                />
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                {feature.description}
              </p>
              {feature.enabled && (
                <Button variant="outline" size="sm" className="w-full" onClick={() => console.log("enhanced-features button clicked")}>
                  <Settings className="h-4 w-4 mr-2" />
                  Configure
                </Button>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" onClick={() => features.forEach(f => toggleFeature(f.id))}>
              Toggle All Features
            </Button>
            <Button variant="outline" onClick={() => console.log("enhanced-features button clicked")}>
              <Monitor className="h-4 w-4 mr-2" />
              System Status
            </Button>
            <Button variant="outline" onClick={() => console.log("enhanced-features button clicked")}>
              <BarChart3 className="h-4 w-4 mr-2" />
              Usage Analytics
            </Button>
            <Button variant="outline" onClick={() => console.log("enhanced-features button clicked")}>
              <Settings className="h-4 w-4 mr-2" />
              Advanced Settings
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};