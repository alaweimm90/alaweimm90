import { useEffect, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Navigation } from "@/components/Navigation";
import { SettingsPage } from "@/components/settings/SettingsPage";
import { ModelManager } from "@/components/settings/ModelManager";
import { SystemMonitor } from "@/components/settings/SystemMonitor";
import { setSEO } from "@/lib/seo";

export default function Settings() {
  useEffect(() => {
    setSEO({
      title: "Settings | LLM Works",
      description: "Configure platform preferences, manage models, and monitor system performance.",
      path: "/settings",
    });
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div id="main" className="container mx-auto px-4 py-8">
        <h1 className="sr-only">Settings</h1>
        <Tabs defaultValue="general" className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8">
            <TabsTrigger value="general">Settings</TabsTrigger>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="monitor">Monitor</TabsTrigger>
          </TabsList>

          <TabsContent value="general">
            <SettingsPage />
          </TabsContent>

          <TabsContent value="models">
            <div className="space-y-6">
              <div>
                <h2 className="text-3xl font-bold text-foreground">Model Management</h2>
                <p className="text-muted-foreground">Configure and manage AI models for evaluation tasks</p>
              </div>
              <ModelManager />
            </div>
          </TabsContent>

          <TabsContent value="monitor">
            <div className="space-y-6">
              <div>
                <h2 className="text-3xl font-bold text-foreground">System Monitor</h2>
                <p className="text-muted-foreground">Real-time system status and performance metrics</p>
              </div>
              <SystemMonitor />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}