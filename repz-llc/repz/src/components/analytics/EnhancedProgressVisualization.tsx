import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Gauge, Activity, Target, TrendingUp } from 'lucide-react';

export const EnhancedProgressVisualization: React.FC = () => {
  const [timeRange, setTimeRange] = useState('3months');

  const goals = [
    { goal: 'Lose 15 lbs', progress: 73, trend: 'up', category: 'body' },
    { goal: 'Bench Press 200 lbs', progress: 70, trend: 'up', category: 'strength' },
    { goal: '5K in under 25 min', progress: 80, trend: 'up', category: 'cardio' },
    { goal: 'Sleep 8h avg', progress: 90, trend: 'up', category: 'lifestyle' }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-blue-100 rounded-lg">
          <Gauge className="h-6 w-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold">Enhanced Progress Visualization</h2>
          <p className="text-muted-foreground">Comprehensive fitness journey tracking</p>
        </div>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="body">Body Metrics</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="goals">Goals</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Overall Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">88</div>
                <Progress value={88} className="mt-2 h-2" />
                <div className="flex items-center gap-1 mt-2">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  <span className="text-xs text-green-500">+12% this month</span>
                </div>
              </CardContent>
            </Card>

            {goals.slice(0, 3).map((goal, index) => (
              <Card key={index}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">{goal.goal}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xl font-bold">{goal.progress}%</div>
                  <Progress value={goal.progress} className="mt-2 h-2" />
                  <Badge variant="outline" className="mt-2 text-xs">
                    {goal.category}
                  </Badge>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="body" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Body Composition Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold">169 lbs</div>
                    <div className="text-sm text-muted-foreground">Current Weight</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">14.5%</div>
                    <div className="text-sm text-muted-foreground">Body Fat</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold">48.5 lbs</div>
                    <div className="text-sm text-muted-foreground">Muscle Mass</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Strength</span>
                    <span className="text-sm font-medium">88/100</span>
                  </div>
                  <Progress value={88} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Cardio</span>
                    <span className="text-sm font-medium">83/100</span>
                  </div>
                  <Progress value={83} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Flexibility</span>
                    <span className="text-sm font-medium">72/100</span>
                  </div>
                  <Progress value={72} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="goals" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {goals.map((goal, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{goal.goal}</span>
                    <Badge className={`${goal.category === 'body' ? 'bg-green-100 text-green-800' : 
                      goal.category === 'strength' ? 'bg-red-100 text-red-800' :
                      goal.category === 'cardio' ? 'bg-blue-100 text-blue-800' :
                      'bg-purple-100 text-purple-800'}`}>
                      {goal.category}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Progress value={goal.progress} className="h-3" />
                  <div className="flex justify-between mt-2 text-sm">
                    <span>{goal.progress}% Complete</span>
                    <span className="flex items-center gap-1">
                      <TrendingUp className="h-3 w-3 text-green-500" />
                      On Track
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};