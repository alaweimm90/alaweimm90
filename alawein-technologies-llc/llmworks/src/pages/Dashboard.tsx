import { useEffect } from "react";
import { Navigation } from "@/components/Navigation";
import { Dashboard } from "@/components/Dashboard";
import { AnalyticsDashboard } from "@/components/analytics/AnalyticsDashboard";
import { ModelManagementDashboard } from "@/components/dashboard/ModelManagementDashboard";
import { EvaluationMonitoringDashboard } from "@/components/dashboard/EvaluationMonitoringDashboard";
import { CostTrackingDashboard } from "@/components/dashboard/CostTrackingDashboard";
import { SystemHealthDashboard } from "@/components/dashboard/SystemHealthDashboard";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Home, BarChart3, Activity, Brain, Play, DollarSign, Shield } from "lucide-react";
import { Link } from "react-router-dom";
import { setSEO, injectJsonLd } from "@/lib/seo";

export default function DashboardPage() {
  useEffect(() => {
    setSEO({
      title: "Dashboard | LLM Works",
      description: "Monitor your AI evaluation activities, recent runs, and performance insights.",
      path: "/dashboard",
    });

    injectJsonLd({
      "@context": "https://schema.org",
      "@type": "WebPage",
      name: "LLM Works Dashboard",
      url: `${window.location.origin}/dashboard`,
      description: "Control center for AI evaluation, monitoring, and analytics.",
      isPartOf: { "@type": "SoftwareApplication", name: "LLM Works" }
    }, "ld-dashboard");
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main id="main" className="container mx-auto px-4 py-8">
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
            <p className="text-muted-foreground">Monitor your evaluation activities, recent runs, and performance insights</p>
          </div>
          <Button asChild variant="outline">
            <Link to="/" aria-label="Back to Home">
              <Home className="h-4 w-4 mr-2" aria-hidden="true" />
              Back to Home
            </Link>
          </Button>
        </header>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="mb-8 grid w-full grid-cols-2 lg:grid-cols-6">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Overview</span>
            </TabsTrigger>
            <TabsTrigger value="models" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              <span className="hidden sm:inline">Models</span>
            </TabsTrigger>
            <TabsTrigger value="monitoring" className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              <span className="hidden sm:inline">Monitoring</span>
            </TabsTrigger>
            <TabsTrigger value="costs" className="flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              <span className="hidden sm:inline">Costs</span>
            </TabsTrigger>
            <TabsTrigger value="health" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              <span className="hidden sm:inline">Health</span>
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <Dashboard />
          </TabsContent>

          <TabsContent value="models">
            <ModelManagementDashboard />
          </TabsContent>

          <TabsContent value="monitoring">
            <EvaluationMonitoringDashboard />
          </TabsContent>

          <TabsContent value="costs">
            <CostTrackingDashboard />
          </TabsContent>

          <TabsContent value="health">
            <SystemHealthDashboard />
          </TabsContent>

          <TabsContent value="analytics">
            <AnalyticsDashboard />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
