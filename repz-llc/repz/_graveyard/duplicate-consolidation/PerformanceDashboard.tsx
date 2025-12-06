import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs";
import { CoreWebVitalsMonitor } from './CoreWebVitalsMonitor';
import { BundleAnalyzer } from './BundleAnalyzer';
import { DatabasePerformanceMonitor } from './DatabasePerformanceMonitor';
import { PerformanceOptimizationSuite } from './PerformanceOptimizationSuite';
import { Activity, Package, Database, Zap } from 'lucide-react';

export function PerformanceDashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Performance & Optimization Dashboard</h1>
        <p className="text-muted-foreground">
          Google Core Web Vitals • Bundle Optimization • Database Performance
        </p>
      </div>

      <Tabs defaultValue="vitals" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="vitals" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Core Web Vitals
          </TabsTrigger>
          <TabsTrigger value="bundle" className="flex items-center gap-2">
            <Package className="h-4 w-4" />
            Bundle Analysis
          </TabsTrigger>
          <TabsTrigger value="database" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Database Performance
          </TabsTrigger>
          <TabsTrigger value="optimization" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Optimization Suite
          </TabsTrigger>
        </TabsList>

        <TabsContent value="vitals">
          <CoreWebVitalsMonitor />
        </TabsContent>

        <TabsContent value="bundle">
          <BundleAnalyzer />
        </TabsContent>

        <TabsContent value="database">
          <DatabasePerformanceMonitor />
        </TabsContent>

        <TabsContent value="optimization">
          <PerformanceOptimizationSuite />
        </TabsContent>
      </Tabs>
    </div>
  );
}