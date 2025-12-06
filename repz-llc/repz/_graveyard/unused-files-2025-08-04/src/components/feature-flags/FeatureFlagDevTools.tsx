/**
 * REPZ Feature Flag Dev Tools
 * Development tools for managing and testing feature flags
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { RepzFeatureFlags, FEATURE_FLAGS, featureFlagManager } from '@/lib/feature-flags';
import { useAuth } from '@/contexts/AuthContext';
import { createUserContext } from '@/lib/feature-flags';

interface FeatureFlagOverride {
  flagId: string;
  value: boolean | string | number;
  originalValue?: boolean;
}

/**
 * Feature Flag Development Tools Component
 * 
 * Provides a comprehensive interface for:
 * - Viewing all feature flags and their states
 * - Overriding flags for testing
 * - Monitoring flag evaluations
 * - Testing different user contexts
 */
export function FeatureFlagDevTools(): JSX.Element {
  const { user } = useAuth();
  const [overrides, setOverrides] = useState<FeatureFlagOverride[]>([]);
  const [allFlags, setAllFlags] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [newOverride, setNewOverride] = useState({ flagId: '', value: '' });
  const [testContext, setTestContext] = useState({
    tier: user?.tier || 'core',
    role: user?.role || 'client',
    subscriptionStatus: 'active'
  });

  // Load current flag states
  useEffect(() => {
    if (user) {
      loadFlags();
    }
  }, [user, testContext]);

  const loadFlags = async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const userContext = createUserContext({
        ...user,
        tier: testContext.tier,
        role: testContext.role,
        subscription_status: testContext.subscriptionStatus
      });
      
      const flags = await RepzFeatureFlags.getAllFeatures(userContext);
      setAllFlags(flags);
    } catch (error) {
      console.error('Failed to load flags:', error);
    } finally {
      setLoading(false);
    }
  };

  const addOverride = () => {
    if (!newOverride.flagId || newOverride.value === '') return;

    let value: boolean | string | number = newOverride.value;
    
    // Parse value type
    if (value === 'true') value = true;
    else if (value === 'false') value = false;
    else if (!isNaN(Number(value))) value = Number(value);

    const originalValue = allFlags[newOverride.flagId];
    
    RepzFeatureFlags.override(newOverride.flagId, value);
    
    setOverrides(prev => [
      ...prev.filter(o => o.flagId !== newOverride.flagId),
      { flagId: newOverride.flagId, value, originalValue }
    ]);
    
    setNewOverride({ flagId: '', value: '' });
    loadFlags();
  };

  const removeOverride = (flagId: string) => {
    RepzFeatureFlags.clearOverride(flagId);
    setOverrides(prev => prev.filter(o => o.flagId !== flagId));
    loadFlags();
  };

  const clearAllOverrides = () => {
    overrides.forEach(override => {
      RepzFeatureFlags.clearOverride(override.flagId);
    });
    setOverrides([]);
    loadFlags();
  };

  if (!import.meta.env.DEV) {
    return <div>Feature Flag Dev Tools are only available in development mode.</div>;
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Feature Flag Dev Tools</h2>
        <Badge variant="outline">Development Only</Badge>
      </div>

      <Tabs defaultValue="flags" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="flags">All Flags</TabsTrigger>
          <TabsTrigger value="overrides">Overrides</TabsTrigger>
          <TabsTrigger value="context">Test Context</TabsTrigger>
          <TabsTrigger value="monitor">Monitor</TabsTrigger>
        </TabsList>

        <TabsContent value="flags" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Current Feature Flags</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div>Loading flags...</div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(FEATURE_FLAGS).map(([name, flagId]) => {
                    const isEnabled = allFlags[flagId];
                    const hasOverride = overrides.some(o => o.flagId === flagId);
                    
                    return (
                      <div
                        key={flagId}
                        className="flex items-center justify-between p-3 border rounded-lg"
                      >
                        <div className="flex flex-col">
                          <span className="font-medium text-sm">{name}</span>
                          <span className="text-xs text-gray-500">{flagId}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {hasOverride && (
                            <Badge variant="secondary" className="text-xs">
                              Override
                            </Badge>
                          )}
                          <Badge
                            variant={isEnabled ? 'default' : 'outline'}
                            className="text-xs"
                          >
                            {isEnabled ? 'ON' : 'OFF'}
                          </Badge>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="overrides" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Flag Overrides</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Flag ID"
                  value={newOverride.flagId}
                  onChange={(e) => setNewOverride(prev => ({ ...prev, flagId: e.target.value }))}
                />
                <Input
                  placeholder="Value (true/false/string/number)"
                  value={newOverride.value}
                  onChange={(e) => setNewOverride(prev => ({ ...prev, value: e.target.value }))}
                />
                <Button onClick={addOverride}>Add Override</Button>
              </div>

              {overrides.length > 0 && (
                <>
                  <div className="flex justify-between items-center">
                    <h4 className="font-medium">Active Overrides</h4>
                    <Button variant="outline" size="sm" onClick={clearAllOverrides}>
                      Clear All
                    </Button>
                  </div>
                  
                  <div className="space-y-2">
                    {overrides.map(override => (
                      <div
                        key={override.flagId}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                      >
                        <div>
                          <span className="font-medium">{override.flagId}</span>
                          <div className="text-sm text-gray-600">
                            Override: <code>{String(override.value)}</code>
                            {override.originalValue !== undefined && (
                              <span className="ml-2">
                                (was: <code>{String(override.originalValue)}</code>)
                              </span>
                            )}
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => removeOverride(override.flagId)}
                        >
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="context" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Test User Context</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="tier">Tier</Label>
                  <select
                    id="tier"
                    className="w-full p-2 border rounded-md"
                    value={testContext.tier}
                    onChange={(e) => setTestContext(prev => ({ ...prev, tier: e.target.value }))}
                  >
                    <option value="core">Core</option>
                    <option value="adaptive">Adaptive</option>
                    <option value="performance">Performance</option>
                    <option value="longevity">Longevity</option>
                  </select>
                </div>

                <div>
                  <Label htmlFor="role">Role</Label>
                  <select
                    id="role"
                    className="w-full p-2 border rounded-md"
                    value={testContext.role}
                    onChange={(e) => setTestContext(prev => ({ ...prev, role: e.target.value }))}
                  >
                    <option value="client">Client</option>
                    <option value="coach">Coach</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>

                <div>
                  <Label htmlFor="subscription">Subscription Status</Label>
                  <select
                    id="subscription"
                    className="w-full p-2 border rounded-md"
                    value={testContext.subscriptionStatus}
                    onChange={(e) => setTestContext(prev => ({ ...prev, subscriptionStatus: e.target.value }))}
                  >
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                    <option value="trial">Trial</option>
                    <option value="expired">Expired</option>
                  </select>
                </div>
              </div>

              <Button onClick={loadFlags} className="w-full">
                Refresh Flags with New Context
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitor" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Flag Evaluation Monitor</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-gray-600">
                <p>Flag evaluations are logged to the browser console in development mode.</p>
                <p className="mt-2">
                  Open Developer Tools → Console to see real-time flag evaluations,
                  including user context, evaluation results, and performance metrics.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}