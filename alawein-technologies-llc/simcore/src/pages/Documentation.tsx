import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Code, Info, Lightbulb } from "lucide-react";

export default function Documentation() {
  return (
    <div className="min-h-screen p-6 bg-gradient-to-br from-background via-background to-muted/20">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            SimCore Documentation
          </h1>
          <p className="text-muted-foreground text-lg">
            Interactive Scientific Computing Laboratory
          </p>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">
              <Info className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="getting-started">
              <Lightbulb className="w-4 h-4 mr-2" />
              Getting Started
            </TabsTrigger>
            <TabsTrigger value="api">
              <Code className="w-4 h-4 mr-2" />
              API Reference
            </TabsTrigger>
            <TabsTrigger value="guides">
              <BookOpen className="w-4 h-4 mr-2" />
              Guides
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Welcome to SimCore</CardTitle>
                <CardDescription>
                  High-performance simulation framework for scientific computing
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p>
                  SimCore is an enterprise-grade simulation framework designed for complex systems modeling,
                  combining theoretical physics principles with modern web technologies.
                </p>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="p-4 border rounded-lg">
                    <h3 className="font-semibold mb-2">Key Features</h3>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                      <li>Interactive physics simulations</li>
                      <li>Real-time visualizations</li>
                      <li>Advanced computational methods</li>
                      <li>Export and analysis tools</li>
                    </ul>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h3 className="font-semibold mb-2">Technologies</h3>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                      <li>React 18 with TypeScript</li>
                      <li>Three.js for 3D graphics</li>
                      <li>Web Workers for performance</li>
                      <li>Progressive Web App</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="getting-started" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Getting Started</CardTitle>
                <CardDescription>Quick start guide for SimCore</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Installation</h3>
                  <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <code>npm install{"\n"}npm run dev</code>
                  </pre>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Basic Usage</h3>
                  <p className="text-sm text-muted-foreground mb-2">
                    Navigate through the available simulations using the sidebar or dashboard.
                    Each simulation provides interactive controls and real-time visualization.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="api" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>API Reference</CardTitle>
                <CardDescription>Core APIs and components</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Detailed API documentation coming soon. Explore the source code and
                  type definitions for comprehensive API information.
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="guides" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Guides & Tutorials</CardTitle>
                <CardDescription>Learn how to use SimCore effectively</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Comprehensive guides and tutorials are being developed.
                  Check back soon for step-by-step instructions on various simulations.
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
