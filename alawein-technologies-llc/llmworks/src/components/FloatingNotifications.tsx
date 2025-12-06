import { memo, useState, useEffect, createContext, useContext, ReactNode } from 'react';
import { X, CheckCircle, AlertTriangle, Info, Zap, Trophy, Star, Target } from 'lucide-react';

interface Notification {
  id: string;
  type: 'success' | 'warning' | 'info' | 'achievement' | 'battle' | 'xp';
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  icon?: React.ComponentType<{ className?: string }>;
}

interface NotificationContextType {
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};

interface NotificationProviderProps {
  children: ReactNode;
}

export const NotificationProvider = ({ children }: NotificationProviderProps) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = (notification: Omit<Notification, 'id'>) => {
    const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const newNotification: Notification = {
      ...notification,
      id,
      duration: notification.duration || 5000
    };

    setNotifications(prev => [...prev, newNotification]);

    // Auto remove notification
    setTimeout(() => {
      removeNotification(id);
    }, newNotification.duration);
  };

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  // Demo notifications for showcase
  useEffect(() => {
    const demoInterval = setInterval(() => {
      if (Math.random() > 0.7) {
        const demoNotifications = [
          {
            type: 'achievement' as const,
            title: 'New Achievement Unlocked!',
            message: 'Strategic Commander - Level 15 reached',
            icon: Trophy,
            duration: 4000
          },
          {
            type: 'xp' as const,
            title: '+250 XP Gained!',
            message: 'Excellent strategic evaluation performance',
            icon: Star,
            duration: 3000
          },
          {
            type: 'battle' as const,
            title: 'Battle Victory!',
            message: 'GPT-4 defeated Claude-3 in creative challenge',
            icon: Target,
            duration: 3500
          },
          {
            type: 'success' as const,
            title: 'System Online',
            message: 'All strategic modules operational',
            duration: 3000
          },
          {
            type: 'info' as const,
            title: 'New Model Available',
            message: 'Gemini Ultra Pro now in evaluation arena',
            duration: 4000
          }
        ];

        const randomNotification = demoNotifications[Math.floor(Math.random() * demoNotifications.length)];
        addNotification(randomNotification);
      }
    }, 8000);

    return () => clearInterval(demoInterval);
  }, []);

  return (
    <NotificationContext.Provider value={{ addNotification, removeNotification }}>
      {children}
      <FloatingNotifications 
        notifications={notifications} 
        onRemove={removeNotification} 
      />
    </NotificationContext.Provider>
  );
};

interface FloatingNotificationsProps {
  notifications: Notification[];
  onRemove: (id: string) => void;
}

const FloatingNotificationsComponent = ({ notifications, onRemove }: FloatingNotificationsProps) => {
  const getNotificationStyles = (type: string) => {
    switch (type) {
      case 'success':
        return {
          bg: 'bg-green-500/10 border-green-500/30',
          icon: 'text-green-400',
          accent: 'border-l-green-400'
        };
      case 'warning':
        return {
          bg: 'bg-yellow-500/10 border-yellow-500/30',
          icon: 'text-yellow-400',
          accent: 'border-l-yellow-400'
        };
      case 'info':
        return {
          bg: 'bg-blue-500/10 border-blue-500/30',
          icon: 'text-blue-400',
          accent: 'border-l-blue-400'
        };
      case 'achievement':
        return {
          bg: 'bg-purple-500/10 border-purple-500/30',
          icon: 'text-purple-400',
          accent: 'border-l-purple-400'
        };
      case 'battle':
        return {
          bg: 'bg-red-500/10 border-red-500/30',
          icon: 'text-red-400',
          accent: 'border-l-red-400'
        };
      case 'xp':
        return {
          bg: 'bg-orange-500/10 border-orange-500/30',
          icon: 'text-orange-400',
          accent: 'border-l-orange-400'
        };
      default:
        return {
          bg: 'bg-muted/10 border-muted/30',
          icon: 'text-muted-foreground',
          accent: 'border-l-muted-foreground'
        };
    }
  };

  const getDefaultIcon = (type: string) => {
    switch (type) {
      case 'success': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      case 'achievement': return Trophy;
      case 'battle': return Target;
      case 'xp': return Star;
      default: return Info;
    }
  };

  return (
    <div className="fixed top-20 right-4 z-50 space-y-3 max-w-sm">
      {notifications.map((notification, index) => {
        const styles = getNotificationStyles(notification.type);
        const IconComponent = notification.icon || getDefaultIcon(notification.type);
        
        return (
          <div
            key={notification.id}
            className={`glass-panel ${styles.bg} ${styles.accent} border-l-4 p-4 rounded-lg shadow-lg transform transition-all duration-500 ease-out animate-in slide-in-from-right-full`}
            style={{
              animationDelay: `${index * 100}ms`,
              animationFillMode: 'both'
            }}
          >
            <div className="flex items-start gap-3">
              {/* Icon */}
              <div className={`p-1 rounded-lg ${styles.bg}`}>
                <IconComponent className={`h-5 w-5 ${styles.icon}`} />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h4 className="heading-refined text-sm font-semibold mb-1">
                      {notification.title}
                    </h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {notification.message}
                    </p>
                  </div>
                  
                  {/* Close Button */}
                  <button
                    onClick={() => onRemove(notification.id)}
                    className="flex-shrink-0 p-1 rounded-lg hover:bg-muted/20 transition-colors"
                    aria-label="Close notification"
                  >
                    <X className="h-3 w-3 text-muted-foreground hover:text-foreground" />
                  </button>
                </div>

                {/* Action Button */}
                {notification.action && (
                  <button
                    onClick={notification.action.onClick}
                    className="mt-2 text-xs text-primary hover:text-primary/80 font-medium underline decoration-1 underline-offset-2"
                  >
                    {notification.action.label}
                  </button>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            <div className="absolute bottom-0 left-0 right-0 h-1 bg-muted/20 rounded-b-lg overflow-hidden">
              <div 
                className={`h-full ${styles.accent.replace('border-l-', 'bg-')} transition-all ease-linear`}
                style={{
                  width: '100%',
                  animation: `shrink-width ${notification.duration}ms linear`
                }}
              />
            </div>

            {/* Special Effects for Achievements */}
            {notification.type === 'achievement' && (
              <div className="absolute -top-1 -right-1">
                <div className="w-3 h-3 bg-purple-400 rounded-full animate-ping" />
                <div className="absolute top-0 right-0 w-3 h-3 bg-purple-400 rounded-full animate-pulse" />
              </div>
            )}

            {/* XP Glow Effect */}
            {notification.type === 'xp' && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-400/5 to-transparent animate-pulse rounded-lg" />
            )}

            {/* Battle Spark Effect */}
            {notification.type === 'battle' && (
              <div className="absolute top-2 right-8">
                <Zap className="h-4 w-4 text-yellow-400 animate-bounce" />
              </div>
            )}
          </div>
        );
      })}

      <style>{`
        @keyframes shrink-width {
          from { width: 100%; }
          to { width: 0%; }
        }

        @keyframes slide-in-from-right-full {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }

        .animate-in {
          animation-duration: 0.5s;
          animation-timing-function: cubic-bezier(0.16, 1, 0.3, 1);
        }
      `}</style>
    </div>
  );
};

const FloatingNotifications = memo(FloatingNotificationsComponent);