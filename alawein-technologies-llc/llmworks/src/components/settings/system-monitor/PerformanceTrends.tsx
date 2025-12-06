import { Card } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, AreaChart, Area } from "recharts";
import { TrendingUp, TrendingDown, Activity, Clock, Zap } from "lucide-react";

const performanceData = [
  { time: "00:00", cpu: 45, memory: 62, responseTime: 120, throughput: 85 },
  { time: "04:00", cpu: 52, memory: 58, responseTime: 110, throughput: 92 },
  { time: "08:00", cpu: 78, memory: 72, responseTime: 95, throughput: 110 },
  { time: "12:00", cpu: 85, memory: 84, responseTime: 85, throughput: 125 },
  { time: "16:00", cpu: 72, memory: 79, responseTime: 92, throughput: 118 },
  { time: "20:00", cpu: 58, memory: 65, responseTime: 105, throughput: 98 },
];

const chartConfig = {
  cpu: { label: "CPU Usage", color: "hsl(var(--primary))" },
  memory: { label: "Memory Usage", color: "hsl(var(--accent))" },
  responseTime: { label: "Response Time", color: "hsl(var(--primary))" },
  throughput: { label: "Throughput", color: "hsl(var(--accent))" },
};

export const PerformanceTrends = () => {
  const metrics = [
    {
      title: "CPU Usage",
      value: "72%",
      change: "+5.2%",
      trend: "up",
      icon: Activity,
      color: "text-primary"
    },
    {
      title: "Memory Usage", 
      value: "84%",
      change: "-2.1%",
      trend: "down",
      icon: Zap,
      color: "text-accent"
    },
    {
      title: "Avg Response Time",
      value: "92ms",
      change: "-8.5%", 
      trend: "down",
      icon: Clock,
      color: "text-primary"
    },
    {
      title: "Throughput",
      value: "125 req/s",
      change: "+12.3%",
      trend: "up", 
      icon: TrendingUp,
      color: "text-accent"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Metrics Summary */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <Card key={index} className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{metric.title}</p>
                <p className="text-2xl font-bold text-foreground">{metric.value}</p>
                <div className="flex items-center gap-1 mt-1">
                  {metric.trend === "up" ? (
                    <TrendingUp className="h-3 w-3 text-green-500" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-red-500" />
                  )}
                  <span className={`text-xs ${metric.trend === "up" ? "text-green-500" : "text-red-500"}`}>
                    {metric.change}
                  </span>
                </div>
              </div>
              <metric.icon className={`h-8 w-8 ${metric.color}`} />
            </div>
          </Card>
        ))}
      </div>

      {/* Performance Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">System Resources</h3>
          <ChartContainer config={chartConfig} className="h-[300px]">
            <LineChart data={performanceData}>
              <XAxis dataKey="time" />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line 
                type="monotone" 
                dataKey="cpu" 
                stroke="var(--color-cpu)" 
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="memory" 
                stroke="var(--color-memory)" 
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ChartContainer>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Performance Metrics</h3>
          <ChartContainer config={chartConfig} className="h-[300px]">
            <AreaChart data={performanceData}>
              <XAxis dataKey="time" />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Area 
                type="monotone" 
                dataKey="responseTime" 
                stroke="var(--color-responseTime)" 
                fill="var(--color-responseTime)"
                fillOpacity={0.3}
              />
              <Area 
                type="monotone" 
                dataKey="throughput" 
                stroke="var(--color-throughput)" 
                fill="var(--color-throughput)"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ChartContainer>
        </Card>
      </div>
    </div>
  );
};
