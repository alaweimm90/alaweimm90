import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  Bell,
  X,
  Check,
  Info,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Trash2,
  Settings,
  Filter
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
  read: boolean;
  actionable?: boolean;
  action?: () => void;
  actionLabel?: string;
}

interface NotificationCenterProps {
  className?: string;
  onClose?: () => void;
}

export const NotificationCenter: React.FC<NotificationCenterProps> = ({
  className = '',
  onClose
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const { toast } = useToast();

  useEffect(() => {
    // Load existing notifications from localStorage
    const saved = localStorage.getItem('simcore-notifications');
    if (saved) {
      try {
        const parsed = JSON.parse(saved).map((n: any) => ({
          ...n,
          timestamp: new Date(n.timestamp)
        }));
        setNotifications(parsed);
      } catch (error) {
        console.warn('Failed to parse saved notifications:', error);
      }
    }

    // Add some sample notifications for demo
    const sampleNotifications: Notification[] = [
      {
        id: '1',
        title: 'Welcome to SimCore!',
        message: 'Explore cutting-edge scientific simulations in your browser. Start with our tutorial for the best experience.',
        type: 'info',
        timestamp: new Date(Date.now() - 3600000), // 1 hour ago
        read: false,
        actionable: true,
        action: () => {
          // Restart tutorial
          localStorage.removeItem('simcore-tutorial-completed');
          window.location.reload();
        },
        actionLabel: 'Start Tutorial'
      },
      {
        id: '2',
        title: 'New Module Available',
        message: 'The Quantum Field Theory module has been updated with enhanced visualizations and improved performance.',
        type: 'success',
        timestamp: new Date(Date.now() - 86400000), // 1 day ago
        read: false,
        actionable: true,
        action: () => {
          window.location.href = '/modules/quantum-field-theory';
        },
        actionLabel: 'Explore Module'
      },
      {
        id: '3',
        title: 'Performance Enhancement',
        message: 'WebGPU acceleration is now available for supported browsers. Your simulations will run faster!',
        type: 'info',
        timestamp: new Date(Date.now() - 172800000), // 2 days ago
        read: true
      },
      {
        id: '4',
        title: 'Achievement Unlocked',
        message: 'Congratulations! You\'ve earned the "Quantum Explorer" achievement for completing 3 quantum mechanics modules.',
        type: 'success',
        timestamp: new Date(Date.now() - 259200000), // 3 days ago
        read: true
      },
      {
        id: '5',
        title: 'Maintenance Notice',
        message: 'Scheduled maintenance will occur tonight from 2-4 AM UTC. Some features may be temporarily unavailable.',
        type: 'warning',
        timestamp: new Date(Date.now() - 345600000), // 4 days ago
        read: true
      }
    ];

    setNotifications(prev => {
      const existing = prev.length > 0 ? prev : sampleNotifications;
      return existing;
    });
  }, []);

  // Save notifications to localStorage whenever they change
  useEffect(() => {
    if (notifications.length > 0) {
      localStorage.setItem('simcore-notifications', JSON.stringify(notifications));
    }
  }, [notifications]);

  const getIcon = (type: Notification['type']) => {
    switch (type) {
      case 'info': return Info;
      case 'success': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'error': return XCircle;
      default: return Info;
    }
  };

  const getTypeColor = (type: Notification['type']) => {
    switch (type) {
      case 'info': return 'bg-info/10 text-info border-info/20';
      case 'success': return 'bg-success/10 text-success border-success/20';
      case 'warning': return 'bg-warning/10 text-warning border-warning/20';
      case 'error': return 'bg-destructive/10 text-destructive border-destructive/20';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const markAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev =>
      prev.map(n => ({ ...n, read: true }))
    );
    toast({
      title: "All Notifications Read",
      description: "Marked all notifications as read.",
    });
  };

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
    toast({
      title: "Notification Removed",
      description: "Notification has been deleted.",
    });
  };

  const clearAll = () => {
    setNotifications([]);
    localStorage.removeItem('simcore-notifications');
    toast({
      title: "Notifications Cleared",
      description: "All notifications have been removed.",
    });
  };

  const filteredNotifications = notifications.filter(n => 
    filter === 'all' || !n.read
  );

  const unreadCount = notifications.filter(n => !n.read).length;

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    return 'Just now';
  };

  return (
    <Card className={`w-full max-w-md border-border/50 ${className}`}>
      <CardHeader className="pb-[--spacing-md]">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-[--spacing-sm]">
            <Bell className="w-5 h-5 text-primary" />
            Notifications
            {unreadCount > 0 && (
              <Badge variant="default" className="text-xs">
                {unreadCount}
              </Badge>
            )}
          </CardTitle>
          {onClose && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-6 w-6 p-0 min-h-[--touch-target-min] min-w-[--touch-target-min]"
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-[--spacing-sm] pt-[--spacing-sm]">
          <Button
            variant={filter === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilter('all')}
            className="min-h-[--touch-target-min]"
          >
            All ({notifications.length})
          </Button>
          <Button
            variant={filter === 'unread' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilter('unread')}
            className="min-h-[--touch-target-min]"
          >
            Unread ({unreadCount})
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {/* Action Bar */}
        {notifications.length > 0 && (
          <div className="px-[--spacing-lg] pb-[--spacing-md]">
            <div className="flex items-center justify-between text-sm">
              <Button
                variant="ghost"
                size="sm"
                onClick={markAllAsRead}
                disabled={unreadCount === 0}
                className="h-8 px-[--spacing-sm] min-h-[--touch-target-min]"
              >
                <Check className="w-3 h-3 mr-[--spacing-xs]" />
                Mark All Read
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAll}
                className="h-8 px-[--spacing-sm] text-destructive hover:text-destructive min-h-[--touch-target-min]"
              >
                <Trash2 className="w-3 h-3 mr-[--spacing-xs]" />
                Clear All
              </Button>
            </div>
          </div>
        )}

        <Separator />

        {/* Notifications List */}
        <ScrollArea className="h-96">
          {filteredNotifications.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No notifications</p>
              {filter === 'unread' && (
                <p className="text-sm mt-1">All caught up! 🎉</p>
              )}
            </div>
          ) : (
            <div className="space-y-1">
              {filteredNotifications.map((notification) => {
                const Icon = getIcon(notification.type);
                return (
                  <div
                    key={notification.id}
                    className={`p-4 border-b border-border/50 hover:bg-muted/30 transition-colors ${
                      !notification.read ? 'bg-muted/20' : ''
                    }`}
                  >
                    <div className="space-y-2">
                      {/* Header */}
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 flex-1">
                          <Icon className={`w-4 h-4 flex-shrink-0 ${getTypeColor(notification.type).split(' ')[1]}`} />
                          <h4 className="font-medium text-sm leading-tight">
                            {notification.title}
                          </h4>
                          {!notification.read && (
                            <div className="w-2 h-2 bg-primary rounded-full flex-shrink-0" />
                          )}
                        </div>
                        
                        <div className="flex items-center gap-1">
                          <Badge variant="outline" className={`text-xs ${getTypeColor(notification.type)}`}>
                            {notification.type}
                          </Badge>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeNotification(notification.id)}
                            className="h-6 w-6 p-0 hover:bg-destructive/10"
                          >
                            <X className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>

                      {/* Message */}
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {notification.message}
                      </p>

                      {/* Footer */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <Clock className="w-3 h-3" />
                          {formatTimeAgo(notification.timestamp)}
                        </div>

                        <div className="flex items-center gap-2">
                          {!notification.read && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => markAsRead(notification.id)}
                              className="h-6 px-2 text-xs"
                            >
                              Mark Read
                            </Button>
                          )}
                          
                          {notification.actionable && notification.action && (
                            <Button
                              size="sm"
                              onClick={() => {
                                notification.action?.();
                                markAsRead(notification.id);
                              }}
                              className="h-6 px-2 text-xs"
                            >
                              {notification.actionLabel || 'Take Action'}
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
};