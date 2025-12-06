import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, BarChart, Bar, PieChart, Pie, Cell } from "recharts";
import { Download, Eye, TrendingUp, Users, Activity, Clock } from "lucide-react";
import { trackEvent } from "@/lib/analytics";

const pageViewData = [
  { page: "/", views: 1250, bounceRate: 35 },
  { page: "/arena", views: 890, bounceRate: 28 },
  { page: "/bench", views: 720, bounceRate: 22 },
  { page: "/dashboard", views: 560, bounceRate: 18 },
  { page: "/settings", views: 340, bounceRate: 45 },
];

const userFlowData = [
  { hour: "00", pageViews: 45, events: 12, users: 8 },
  { hour: "04", pageViews: 52, events: 18, users: 12 },
  { hour: "08", pageViews: 180, events: 65, users: 35 },
  { hour: "12", pageViews: 285, events: 120, users: 58 },
  { hour: "16", pageViews: 220, events: 95, users: 42 },
  { hour: "20", pageViews: 160, events: 55, users: 28 },
];

const eventData = [
  { name: "CTA Clicks", value: 45, color: "hsl(var(--primary))" },
  { name: "Tab Changes", value: 32, color: "hsl(var(--accent))" },
  { name: "Page Views", value: 18, color: "hsl(var(--muted))" },
  { name: "Downloads", value: 5, color: "hsl(var(--secondary))" },
];

const chartConfig = {
  pageViews: { label: "Page Views", color: "hsl(var(--primary))" },
  events: { label: "Events", color: "hsl(var(--accent))" },
  users: { label: "Users", color: "hsl(var(--secondary))" },
  views: { label: "Views", color: "hsl(var(--primary))" },
  bounceRate: { label: "Bounce Rate", color: "hsl(var(--accent))" },
};

export function AnalyticsDashboard() {
  const handleExport = () => {
    trackEvent("analytics_export", { type: "csv", timestamp: Date.now() });
    // Export logic would go here
  };

  const metrics = [
    { label: "Total Page Views", value: "3,760", change: "+12.5%", icon: Eye },
    { label: "Unique Users", value: "183", change: "+8.2%", icon: Users },
    { label: "Avg Session Duration", value: "4m 32s", change: "+15.3%", icon: Clock },
    { label: "Event Interactions", value: "310", change: "+22.1%", icon: Activity },
  ];

  return (
    <div className="space-y-6">
      {/* Metrics Overview */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <Card key={index} className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{metric.label}</p>
                <p className="text-2xl font-bold text-foreground">{metric.value}</p>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  <span className="text-xs text-green-500">{metric.change}</span>
                </div>
              </div>
              <metric.icon className="h-8 w-8 text-primary" />
            </div>
          </Card>
        ))}
      </div>

      {/* Analytics Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <div className="flex items-center justify-between mb-6">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="pages">Pages</TabsTrigger>
            <TabsTrigger value="events">Events</TabsTrigger>
            <TabsTrigger value="users">User Flow</TabsTrigger>
          </TabsList>
          <Button onClick={handleExport} variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </Button>
        </div>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-bold text-foreground mb-4">Hourly Activity</h3>
              <ChartContainer config={chartConfig} className="h-[300px]">
                <LineChart data={userFlowData}>
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line 
                    type="monotone" 
                    dataKey="pageViews" 
                    stroke="var(--color-pageViews)" 
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="events" 
                    stroke="var(--color-events)" 
                    strokeWidth={2}
                  />
                </LineChart>
              </ChartContainer>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-bold text-foreground mb-4">Event Distribution</h3>
              <ChartContainer config={chartConfig} className="h-[300px]">
                <PieChart>
                  <Pie
                    data={eventData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                  >
                    {eventData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <ChartTooltip content={<ChartTooltipContent />} />
                </PieChart>
              </ChartContainer>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="pages">
          <Card className="p-6">
            <h3 className="text-lg font-bold text-foreground mb-4">Page Performance</h3>
            <ChartContainer config={chartConfig} className="h-[300px]">
              <BarChart data={pageViewData}>
                <XAxis dataKey="page" />
                <YAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="views" fill="var(--color-views)" />
              </BarChart>
            </ChartContainer>
          </Card>
        </TabsContent>

        <TabsContent value="events">
          <Card className="p-6">
            <h3 className="text-lg font-bold text-foreground mb-4">Event Timeline</h3>
            <div className="space-y-4">
              {[
                { event: "cta_click", source: "hero", target: "arena", time: "2 min ago" },
                { event: "arena_tab_change", tab: "debate", time: "5 min ago" },
                { event: "page_view", path: "/bench", time: "8 min ago" },
                { event: "cta_click", source: "home", target: "dashboard", time: "12 min ago" },
              ].map((event, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                  <div>
                    <div className="font-medium text-foreground">{event.event}</div>
                    <div className="text-sm text-muted-foreground">
                      {Object.entries(event).filter(([k]) => k !== "event" && k !== "time").map(([k, v]) => `${k}: ${v}`).join(", ")}
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">{event.time}</div>
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="users">
          <Card className="p-6">
            <h3 className="text-lg font-bold text-foreground mb-4">User Journey</h3>
            <ChartContainer config={chartConfig} className="h-[300px]">
              <LineChart data={userFlowData}>
                <XAxis dataKey="hour" />
                <YAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line 
                  type="monotone" 
                  dataKey="users" 
                  stroke="var(--color-users)" 
                  strokeWidth={3}
                />
              </LineChart>
            </ChartContainer>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}