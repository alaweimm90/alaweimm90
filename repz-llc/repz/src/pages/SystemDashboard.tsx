import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { useAuth } from '@/contexts/AuthContext';
import { useTierAccess } from '@/hooks/useTierAccess';
import UniversalSearchSystem from '@/components/search/UniversalSearchSystem';
import { Suspense, lazy } from 'react';
const AdvancedBusinessIntelligence = lazy(() => import('@/components/analytics/AdvancedBusinessIntelligence'));
const SecurityMonitor = lazy(() => import('@/components/security/SecurityHardening').then(m => ({ default: m.SecurityMonitor })));
const ProductionMonitor = lazy(() => import('@/components/production/ProductionMonitor').then(m => ({ default: m.ProductionMonitor })));
const AdvancedPerformanceMonitor = lazy(() => import('@/components/monitoring/AdvancedPerformanceMonitor').then(m => ({ default: m.AdvancedPerformanceMonitor })));
const EnterprisePerformanceOptimizer = lazy(() => import('@/components/performance/EnterprisePerformanceOptimizer'));
const ComprehensiveTestSuite = lazy(() => import('@/components/qa/ComprehensiveTestSuite'));
const AdvancedAIHub = lazy(() => import('@/components/ai/AdvancedAIHub'));
const EnterpriseAutomationEngine = lazy(() => import('@/components/automation/EnterpriseAutomationEngine'));
const GlobalInfrastructureManager = lazy(() => import('@/components/global/GlobalInfrastructureManager').then(m => ({ default: m.GlobalInfrastructureManager })));
const InternationalFeaturesManager = lazy(() => import('@/components/global/InternationalFeaturesManager').then(m => ({ default: m.InternationalFeaturesManager })));
const PerformanceMonitor = lazy(() => import('@/enterprise/PerformanceMonitor').then(m => ({ default: m.PerformanceMonitor })));
const ComplianceManager = lazy(() => import('@/enterprise/ComplianceManager').then(m => ({ default: m.ComplianceManager })));
const TestingDashboard = lazy(() => import('@/testing/TestingDashboard').then(m => ({ default: m.TestingDashboard })));
const AIConciergeService = lazy(() => import('@/components/premium/AIConciergeService').then(m => ({ default: m.AIConciergeService })));
const InPersonBookingSystem = lazy(() => import('@/components/premium/InPersonBookingSystem').then(m => ({ default: m.InPersonBookingSystem })));
const AdvancedBodyComposition = lazy(() => import('@/components/premium/AdvancedBodyComposition').then(m => ({ default: m.AdvancedBodyComposition })));
const LongevityOptimization = lazy(() => import('@/components/premium/LongevityOptimization').then(m => ({ default: m.LongevityOptimization })));

const SystemDashboard: React.FC = () => {
  const { user } = useAuth();
  const { userTier } = useTierAccess();

  const [searchParams] = useSearchParams();
  const requestedTab = searchParams.get('tab');
  const defaultTab = (requestedTab ?? (searchParams.get('trace_id') ? 'analytics' : 'search')) as 'search' | 'analytics' | 'security' | 'hardening' | 'production' | 'performance' | 'concierge' | 'booking' | 'composition' | 'longevity' | 'testing' | 'ai-hub' | 'automation' | 'global' | 'international';

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl">System Dashboard</CardTitle>
              <CardDescription>
                Access advanced platform features and enterprise tools
              </CardDescription>
            </div>
            <Badge variant="outline" className="capitalize">
              {userTier} Tier
            </Badge>
          </div>
        </CardHeader>
      </Card>

      <Tabs defaultValue={defaultTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 xl:grid-cols-8 2xl:grid-cols-12">
          <TabsTrigger value="search">Search</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="hardening">Hardening</TabsTrigger>
          <TabsTrigger value="production">Production</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="concierge">Concierge</TabsTrigger>
          <TabsTrigger value="booking">Training</TabsTrigger>
          <TabsTrigger value="composition">Body Comp</TabsTrigger>
          <TabsTrigger value="longevity">Longevity</TabsTrigger>
          <TabsTrigger value="testing">QA Suite</TabsTrigger>
          <TabsTrigger value="ai-hub">AI Hub</TabsTrigger>
          <TabsTrigger value="automation">Automation</TabsTrigger>
          <TabsTrigger value="global">Global</TabsTrigger>
          <TabsTrigger value="international">International</TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="space-y-4">
          <UniversalSearchSystem />
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading analytics…</div>}>
            <AdvancedBusinessIntelligence />
          </Suspense>
        </TabsContent>

        <TabsContent value="production" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading production…</div>}>
            <ProductionMonitor />
          </Suspense>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading performance…</div>}>
            <EnterprisePerformanceOptimizer />
          </Suspense>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading security…</div>}>
            <SecurityMonitor />
          </Suspense>
        </TabsContent>

        <TabsContent value="hardening" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading hardening…</div>}>
            <SecurityMonitor />
          </Suspense>
        </TabsContent>

        <TabsContent value="concierge" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading concierge…</div>}>
            <AIConciergeService />
          </Suspense>
        </TabsContent>

        <TabsContent value="booking" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading booking…</div>}>
            <InPersonBookingSystem />
          </Suspense>
        </TabsContent>

        <TabsContent value="composition" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading composition…</div>}>
            <AdvancedBodyComposition />
          </Suspense>
        </TabsContent>

        <TabsContent value="longevity" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading longevity…</div>}>
            <LongevityOptimization />
          </Suspense>
        </TabsContent>

        <TabsContent value="testing" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading QA suite…</div>}>
            <ComprehensiveTestSuite />
          </Suspense>
        </TabsContent>

        <TabsContent value="ai-hub" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading AI hub…</div>}>
            <AdvancedAIHub />
          </Suspense>
        </TabsContent>

        <TabsContent value="automation" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading automation…</div>}>
            <EnterpriseAutomationEngine />
          </Suspense>
        </TabsContent>

        <TabsContent value="global" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading global…</div>}>
            <GlobalInfrastructureManager />
          </Suspense>
        </TabsContent>

        <TabsContent value="international" className="space-y-4">
          <Suspense fallback={<div className="p-6">Loading international…</div>}>
            <InternationalFeaturesManager />
          </Suspense>
        </TabsContent>

      </Tabs>
    </div>
  );
};

export default SystemDashboard;
