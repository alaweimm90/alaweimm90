import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Calendar, Clock, Play, Pause, Settings, CheckCircle, XCircle } from 'lucide-react';

interface TestSchedule {
  id: string;
  name: string;
  type: 'unit' | 'integration' | 'e2e' | 'performance' | 'security';
  schedule: string;
  status: 'active' | 'paused' | 'disabled';
  lastRun?: Date;
  nextRun: Date;
  success: boolean;
  duration: number;
  environment: 'development' | 'staging' | 'production';
}

export function TestAutomationScheduler() {
  const [searchTerm, setSearchTerm] = useState('');

  const testSchedules: TestSchedule[] = useMemo(() => [
    {
      id: '1',
      name: 'Unit Tests - Core',
      type: 'unit',
      schedule: '0 */2 * * *',
      status: 'active',
      lastRun: new Date(Date.now() - 1000 * 60 * 30),
      nextRun: new Date(Date.now() + 1000 * 60 * 90),
      success: true,
      duration: 180,
      environment: 'development'
    }
  ], []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Test Automation Scheduler</h2>
          <p className="text-muted-foreground">Manage automated test schedules</p>
        </div>
        <Button className="flex items-center gap-2" onClick={() => console.log("TestAutomationScheduler button clicked")}>
          <Settings className="h-4 w-4" />
          Schedule Settings
        </Button>
      </div>

      <div className="grid gap-4">
        {testSchedules.map((schedule) => (
          <Card key={schedule.id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <h3 className="font-semibold">{schedule.name}</h3>
                    <Badge variant="outline">{schedule.type}</Badge>
                    <Badge variant="outline">{schedule.status}</Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-4 w-4" />
                      Next: {schedule.nextRun.toLocaleString()}
                    </div>
                    <span>Duration: {Math.floor(schedule.duration / 60)}m</span>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="flex items-center gap-1" onClick={() => console.log("TestAutomationScheduler button clicked")}>
                  <Play className="h-4 w-4" />
                  Run Now
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}