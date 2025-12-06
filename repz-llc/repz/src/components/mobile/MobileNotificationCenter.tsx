import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Switch } from '@/ui/atoms/Switch';
import { 
  Bell, 
  BellOff, 
  Clock, 
  Zap, 
  Calendar,
  Settings,
  CheckCircle
} from 'lucide-react';
import { usePushNotifications } from '@/hooks/usePushNotifications';
import { useMobileCapacitor } from '@/hooks/useMobileCapacitor';

const MobileNotificationCenter: React.FC = () => {
  const { 
    permission, 
    isSupported, 
    scheduleWorkoutReminder, 
    scheduleProgressReminder,
    scheduleWeeklyMotivation,
    clearAllNotifications 
  } = usePushNotifications();
  
  const { hapticFeedback, isNative } = useMobileCapacitor();
  
  const [settings, setSettings] = useState({
    workoutReminders: true,
    progressReminders: true,
    weeklyMotivation: true,
    coachMessages: true
  });

  const [upcomingReminders, setUpcomingReminders] = useState([
    {
      id: '1',
      type: 'workout',
      title: 'Upper Body Workout',
      time: '2:30 PM Today',
      description: 'Your strength training session is starting soon'
    },
    {
      id: '2',
      type: 'progress',
      title: 'Weekly Check-in',
      time: 'Tomorrow 9:00 AM',
      description: 'Time to log your progress and measurements'
    }
  ]);

  useEffect(() => {
    if (permission.granted && settings.workoutReminders) {
      // Schedule demo workout reminder for tomorrow
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setHours(14, 30, 0, 0);
      scheduleWorkoutReminder(tomorrow, 'Upper Body Workout');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [permission.granted, settings.workoutReminders]);

  const handleSettingChange = (setting: keyof typeof settings) => {
    setSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
    
    if (isNative) {
      hapticFeedback.light();
    }
  };

  const scheduleQuickReminder = async (type: string) => {
    if (isNative) {
      hapticFeedback.medium();
    }

    switch (type) {
      case 'workout': {
        const workoutTime = new Date();
        workoutTime.setHours(workoutTime.getHours() + 1);
        await scheduleWorkoutReminder(workoutTime, 'Quick Workout');
        break;
      }
      case 'progress':
        await scheduleProgressReminder();
        break;
      case 'motivation':
        await scheduleWeeklyMotivation();
        break;
    }
  };

  if (!isSupported) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <BellOff className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-lg font-semibold mb-2">Notifications Not Available</h3>
          <p className="text-muted-foreground">
            Notifications are only available in the mobile app.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Permission Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notification Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Notification Permission</span>
            <Badge variant={permission.granted ? "default" : "secondary"}>
              {permission.granted ? "Enabled" : "Disabled"}
            </Badge>
          </div>
          
          {permission.granted && permission.token && (
            <div className="text-xs text-muted-foreground">
              Device registered for push notifications
            </div>
          )}
        </CardContent>
      </Card>

      {/* Notification Types */}
      <Card>
        <CardHeader>
          <CardTitle>Notification Types</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              <span className="text-sm">Workout Reminders</span>
            </div>
            <Switch 
              checked={settings.workoutReminders}
              onCheckedChange={() => handleSettingChange('workoutReminders')}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              <span className="text-sm">Progress Reminders</span>
            </div>
            <Switch 
              checked={settings.progressReminders}
              onCheckedChange={() => handleSettingChange('progressReminders')}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              <span className="text-sm">Weekly Motivation</span>
            </div>
            <Switch 
              checked={settings.weeklyMotivation}
              onCheckedChange={() => handleSettingChange('weeklyMotivation')}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="h-4 w-4" />
              <span className="text-sm">Coach Messages</span>
            </div>
            <Switch 
              checked={settings.coachMessages}
              onCheckedChange={() => handleSettingChange('coachMessages')}
            />
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Schedule</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button 
            variant="outline" 
            className="w-full justify-start"
            onClick={() => scheduleQuickReminder('workout')}
          >
            <Clock className="mr-2 h-4 w-4" />
            Schedule Workout Reminder (1 hour)
          </Button>
          
          <Button 
            variant="outline" 
            className="w-full justify-start"
            onClick={() => scheduleQuickReminder('progress')}
          >
            <CheckCircle className="mr-2 h-4 w-4" />
            Schedule Progress Reminder (Tomorrow)
          </Button>
          
          <Button 
            variant="outline" 
            className="w-full justify-start"
            onClick={() => scheduleQuickReminder('motivation')}
          >
            <Zap className="mr-2 h-4 w-4" />
            Schedule Weekly Motivation
          </Button>
        </CardContent>
      </Card>

      {/* Upcoming Reminders */}
      <Card>
        <CardHeader>
          <CardTitle>Upcoming Reminders</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {upcomingReminders.length === 0 ? (
            <p className="text-muted-foreground text-sm">No upcoming reminders</p>
          ) : (
            upcomingReminders.map((reminder) => (
              <div key={reminder.id} className="p-3 border rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-sm">{reminder.title}</h4>
                    <p className="text-xs text-muted-foreground mb-1">
                      {reminder.time}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {reminder.description}
                    </p>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {reminder.type}
                  </Badge>
                </div>
              </div>
            ))
          )}
          
          {upcomingReminders.length > 0 && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={clearAllNotifications}
              className="w-full"
            >
              Clear All Reminders
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default MobileNotificationCenter;