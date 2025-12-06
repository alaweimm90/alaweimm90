import { useState, useEffect, useCallback } from 'react';
import { Capacitor } from '@capacitor/core';
import { PushNotifications, Token } from '@capacitor/push-notifications';
import { LocalNotifications } from '@capacitor/local-notifications';

export interface NotificationPermission {
  granted: boolean;
  token?: string;
}

export const usePushNotifications = () => {
  const [permission, setPermission] = useState<NotificationPermission>({ granted: false });
  const [isSupported, setIsSupported] = useState(false);

  const initializePushNotifications = useCallback(async () => {
    try {
      // Request permissions
      const permResult = await PushNotifications.requestPermissions();
      
      if (permResult.receive === 'granted') {
        // Register for push notifications
        await PushNotifications.register();
        
        setPermission({ granted: true });
      }

      // Listeners
      PushNotifications.addListener('registration', (token: Token) => {
        setPermission({ granted: true, token: token.value });
        console.log('Push registration success, token: ' + token.value);
      });

      PushNotifications.addListener('registrationError', (error: Error) => {
        console.error('Error on registration: ' + JSON.stringify(error));
      });

      PushNotifications.addListener('pushNotificationReceived', (notification) => {
        console.log('Push notification received: ', notification);
        // Handle foreground notification
        scheduleLocalNotification({
          title: notification.title || 'REPZ',
          body: notification.body || 'New notification',
        });
      });

      PushNotifications.addListener('pushNotificationActionPerformed', (notification) => {
        console.log('Push notification action performed', notification.actionId, notification.inputValue);
        // Handle notification tap
      });

    } catch (error) {
      console.error('Error initializing push notifications:', error);
    }
  }, []);

  useEffect(() => {
    if (Capacitor.isNativePlatform()) {
      setIsSupported(true);
      initializePushNotifications();
    }
  }, [initializePushNotifications]);

  const scheduleLocalNotification = async (options: {
    title: string;
    body: string;
    scheduleAt?: Date;
    extra?: Record<string, unknown>;
  }) => {
    if (!Capacitor.isNativePlatform()) return;

    try {
      await LocalNotifications.schedule({
        notifications: [
          {
            title: options.title,
            body: options.body,
            id: Date.now(),
            schedule: options.scheduleAt ? { at: options.scheduleAt } : undefined,
            sound: 'default',
            attachments: undefined,
            actionTypeId: '',
            extra: options.extra || {}
          }
        ]
      });
    } catch (error) {
      console.error('Error scheduling local notification:', error);
    }
  };

  const scheduleWorkoutReminder = async (workoutTime: Date, workoutName: string) => {
    const reminderTime = new Date(workoutTime.getTime() - 30 * 60 * 1000); // 30 minutes before
    
    await scheduleLocalNotification({
      title: 'Workout Reminder',
      body: `Your ${workoutName} workout starts in 30 minutes!`,
      scheduleAt: reminderTime,
      extra: { type: 'workout_reminder', workoutName }
    });
  };

  const scheduleProgressReminder = async () => {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(9, 0, 0, 0); // 9 AM tomorrow
    
    await scheduleLocalNotification({
      title: 'Progress Check-in',
      body: "Don't forget to log your progress today!",
      scheduleAt: tomorrow,
      extra: { type: 'progress_reminder' }
    });
  };

  const scheduleWeeklyMotivation = async () => {
    const nextMonday = new Date();
    const daysUntilMonday = (1 + 7 - nextMonday.getDay()) % 7;
    nextMonday.setDate(nextMonday.getDate() + daysUntilMonday);
    nextMonday.setHours(8, 0, 0, 0); // 8 AM Monday
    
    await scheduleLocalNotification({
      title: 'New Week, New Goals!',
      body: 'Ready to crush this week? Check your training plan!',
      scheduleAt: nextMonday,
      extra: { type: 'weekly_motivation' }
    });
  };

  const clearAllNotifications = async () => {
    if (Capacitor.isNativePlatform()) {
      await LocalNotifications.cancel({ notifications: [] });
    }
  };

  return {
    permission,
    isSupported,
    scheduleLocalNotification,
    scheduleWorkoutReminder,
    scheduleProgressReminder,
    scheduleWeeklyMotivation,
    clearAllNotifications
  };
};