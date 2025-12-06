import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { 
  TestTube, 
  Monitor, 
  Smartphone, 
  Tablet,
  Settings,
  Eye,
  BarChart3
} from 'lucide-react';

// Import all testing components
import { ComprehensiveIntegrationTester } from './ComprehensiveIntegrationTester';
import { ResponsiveTierDisplay } from '../pricing/ResponsiveTierDisplay';
import { MobileAccessibilityDashboard } from '../standards/MobileAccessibilityDashboard';
import { PerformanceDashboard } from '@/pages/PerformanceDashboard';

// Import existing components for testing
import { CoreWebVitalsMonitor } from '../performance/CoreWebVitalsMonitor';

export function IntegrationTestDashboard() {
  const [selectedTier, setSelectedTier] = useState<string>('');
  const [showTierCards, setShowTierCards] = useState(true);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Integration & Testing Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Comprehensive testing, integration validation, and cross-platform tier display
        </p>
      </div>

      <Tabs defaultValue="tier-display" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="tier-display" className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Tier Display
          </TabsTrigger>
          <TabsTrigger value="integration-tests" className="flex items-center gap-2">
            <TestTube className="h-4 w-4" />
            Integration Tests
          </TabsTrigger>
          <TabsTrigger value="mobile-standards" className="flex items-center gap-2">
            <Smartphone className="h-4 w-4" />
            Mobile Standards
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Performance
          </TabsTrigger>
          <TabsTrigger value="platform-testing" className="flex items-center gap-2">
            <Monitor className="h-4 w-4" />
            Platform Testing
          </TabsTrigger>
        </TabsList>

        <TabsContent value="tier-display" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Responsive Tier Card Display</CardTitle>
                  <p className="text-muted-foreground">
                    Test tier cards across different screen sizes and platforms
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowTierCards(!showTierCards)}
                  >
                    {showTierCards ? 'Hide' : 'Show'} Tier Cards
                  </Button>
                  {selectedTier && (
                    <Badge variant="default">
                      Selected: {selectedTier}
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            {showTierCards && (
              <CardContent>
                <ResponsiveTierDisplay 
                  onSelectTier={setSelectedTier}
                  currentTier={selectedTier}
                />
              </CardContent>
            )}
          </Card>

          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Monitor className="h-4 w-4" />
                  Desktop View
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Layout:</span>
                    <Badge variant="outline">4-column grid</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Card size:</span>
                    <span>Auto-height</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Hover effects:</span>
                    <Badge variant="secondary">Scale + Shadow</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Tablet className="h-4 w-4" />
                  Tablet View
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Layout:</span>
                    <Badge variant="outline">2-column grid</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Card size:</span>
                    <span>Responsive</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Touch targets:</span>
                    <Badge variant="secondary">44px minimum</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Smartphone className="h-4 w-4" />
                  Mobile View
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Layout:</span>
                    <Badge variant="outline">Single column</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Card size:</span>
                    <span>Full width</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Features:</span>
                    <Badge variant="secondary">Popular indicator</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="integration-tests">
          <ComprehensiveIntegrationTester />
        </TabsContent>

        <TabsContent value="mobile-standards">
          <MobileAccessibilityDashboard />
        </TabsContent>

        <TabsContent value="performance">
          <PerformanceDashboard />
        </TabsContent>

        <TabsContent value="platform-testing" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Platform Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid gap-2">
                    <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                      <span className="flex items-center gap-2">
                        <Monitor className="h-4 w-4" />
                        Desktop (≥1024px)
                      </span>
                      <Badge variant={window.innerWidth >= 1024 ? "default" : "outline"}>
                        {window.innerWidth >= 1024 ? 'Active' : 'Inactive'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                      <span className="flex items-center gap-2">
                        <Tablet className="h-4 w-4" />
                        Tablet (769-1023px)
                      </span>
                      <Badge variant={window.innerWidth >= 769 && window.innerWidth <= 1023 ? "default" : "outline"}>
                        {window.innerWidth >= 769 && window.innerWidth <= 1023 ? 'Active' : 'Inactive'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                      <span className="flex items-center gap-2">
                        <Smartphone className="h-4 w-4" />
                        Mobile (≤768px)
                      </span>
                      <Badge variant={window.innerWidth <= 768 ? "default" : "outline"}>
                        {window.innerWidth <= 768 ? 'Active' : 'Inactive'}
                      </Badge>
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Current viewport: {window.innerWidth}px × {window.innerHeight}px
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Browser Capabilities</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Touch Events:</span>
                    <Badge variant={'ontouchstart' in window ? "default" : "outline"}>
                      {'ontouchstart' in window ? 'Supported' : 'Not Supported'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Device Memory:</span>
                    <Badge variant="outline">
                      {navigator.deviceMemory ? `${navigator.deviceMemory}GB` : 'Unknown'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Connection:</span>
                    <Badge variant="outline">
                      {navigator.connection?.effectiveType || 'Unknown'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Performance API:</span>
                    <Badge variant={'performance' in window ? "default" : "outline"}>
                      {'performance' in window ? 'Available' : 'Not Available'}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Live Core Web Vitals Monitor</CardTitle>
            </CardHeader>
            <CardContent>
              <CoreWebVitalsMonitor />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}