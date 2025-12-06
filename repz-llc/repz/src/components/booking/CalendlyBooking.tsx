import React, { useEffect } from 'react';
import { Button } from '@/ui/atoms/Button';
import { Calendar, ExternalLink } from 'lucide-react';
import { calendlyConfig, getBookingUrl, canBookEventType } from '@/config/calendly';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/ui/molecules/useToast';

// Calendly window interface
interface CalendlyWindow extends Window {
  Calendly?: {
    initPopupWidget: (options: CalendlyPopupOptions) => void;
  };
}

interface CalendlyPopupOptions {
  url: string;
  prefill?: {
    name?: string;
    email?: string;
    customAnswers?: Record<string, string>;
  };
  [key: string]: unknown;
}

interface CalendlyBookingProps {
  eventType: keyof typeof calendlyConfig.events;
  buttonText?: string;
  buttonVariant?: 'link' | 'admin' | 'default' | 'destructive' | 'premium' | 'outline' | 'secondary' | 'ghost' | 'hero' | 'coaching' | 'elegant' | 'inverse' | 'accent';
  className?: string;
  embed?: boolean;
  prefillData?: {
    name?: string;
    email?: string;
    customAnswers?: Record<string, string>;
  };
}

export const CalendlyBooking: React.FC<CalendlyBookingProps> = ({
  eventType,
  buttonText = 'Book Session',
  buttonVariant = 'accent',
  className = '',
  embed = false,
  prefillData = {},
}) => {
  const { user } = useAuth();
  const { toast } = useToast();

  // Load Calendly widget script if embedding
  useEffect(() => {
    if (embed && typeof window !== 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://assets.calendly.com/assets/external/widget.js';
      script.async = true;
      document.body.appendChild(script);

      return () => {
        document.body.removeChild(script);
      };
    }
  }, [embed]);

  const handleBookingClick = () => {
    // Determine user's tier from auth context
    const userMetadata = user?.user_metadata as Record<string, unknown> | undefined;
    const tier = (userMetadata?.subscription_tier as string) || (userMetadata?.profile as Record<string, unknown>)?.subscription_tier as string | undefined;

    // Check if user has access to this event type
    if (tier && !canBookEventType(tier, eventType)) {
      toast({
        title: "Upgrade Required",
        description: "This booking type is not available in your current tier.",
        variant: "destructive",
      });
      return;
    }

    const bookingUrl = getBookingUrl(eventType, tier);

    if (embed) {
      // Open Calendly popup
      const calendlyWindow = window as unknown as CalendlyWindow;
      if (typeof window !== 'undefined' && calendlyWindow.Calendly) {
        const profileMetadata = userMetadata?.profile as Record<string, unknown> | undefined;
        calendlyWindow.Calendly.initPopupWidget({
          url: bookingUrl,
          prefill: {
            name: prefillData.name || (profileMetadata?.client_name as string) || user?.email?.split('@')[0],
            email: prefillData.email || user?.email,
            customAnswers: prefillData.customAnswers || {},
          },
          ...calendlyConfig.embedOptions,
        });
      }
    } else {
      // Open in new tab
      window.open(bookingUrl, '_blank');
    }
  };

  // Event type display names
  const eventTypeNames = {
    consultation: 'Free Consultation (30 min)',
    gymTraining: 'Gym Training Session (1 hr)',
    homeTraining: 'Home Training Session (1 hr)',
    cityTraining: 'City Sports Club Session (1 hr)',
    virtualCheckIn: 'Virtual Check-In (15 min)',
  };

  return (
    <Button
      variant={buttonVariant}
      onClick={handleBookingClick}
      className={`flex items-center gap-2 ${className}`}
    >
      <Calendar className="h-4 w-4" />
      {buttonText || `Book ${eventTypeNames[eventType]}`}
      {!embed && <ExternalLink className="h-3 w-3" />}
    </Button>
  );
};

// Inline embed component
export const CalendlyEmbed: React.FC<{
  eventType: keyof typeof calendlyConfig.events;
  height?: string;
  className?: string;
}> = ({ eventType, height = '650px', className = '' }) => {
  const { user } = useAuth();
  const userMetadata = user?.user_metadata as Record<string, unknown> | undefined;
  const tier = (userMetadata?.subscription_tier as string) || (userMetadata?.profile as Record<string, unknown>)?.subscription_tier as string | undefined;
  const bookingUrl = getBookingUrl(eventType, tier);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://assets.calendly.com/assets/external/widget.js';
      script.async = true;
      document.body.appendChild(script);

      return () => {
        document.body.removeChild(script);
      };
    }
  }, []);

  return (
    <div
      className={`calendly-inline-widget ${className}`}
      data-url={bookingUrl}
      style={{ minWidth: '320px', height }}
    />
  );
};