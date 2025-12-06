import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, Target, Zap } from 'lucide-react';

const mockProgressData = [
  { week: 'Week 1', weight: 180, strength: 85, endurance: 75 },
  { week: 'Week 2', weight: 179, strength: 88, endurance: 78 },
  { week: 'Week 3', weight: 178, strength: 92, endurance: 82 },
  { week: 'Week 4', weight: 177, strength: 95, endurance: 85 },
];

export function ProgressCharts() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-tier-adaptive" />
            Weight Progress
          </CardTitle>
          <CardDescription>
            Tracking your weight loss journey
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={mockProgressData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="weight" 
                stroke="hsl(var(--tier-adaptive))" 
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-tier-performance" />
            Strength Gains
          </CardTitle>
          <CardDescription>
            Your strength improvement over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={mockProgressData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" />
              <YAxis />
              <Tooltip />
              <Bar 
                dataKey="strength" 
                fill="hsl(var(--tier-performance))" 
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="glass-tier-card md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-tier-longevity" />
            Endurance Development
          </CardTitle>
          <CardDescription>
            Your cardiovascular endurance progress
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={mockProgressData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="endurance" 
                stroke="hsl(var(--tier-longevity))" 
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}