import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardFooter, CardHeader } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { CalendlyBooking } from '@/components/booking/CalendlyBooking';
import { MapPin, Clock, Users, Star } from 'lucide-react';
import { cn } from "@/lib/utils";
import { motion } from 'framer-motion';
import { useAuth } from '@/contexts/AuthContext';

interface TrainingPackage {
  id: string;
  name: string;
  price: number;
  sessions: number;
}

interface TrainingLocation {
  id: string;
  name: string;
  address: string;
  isPreferred?: boolean;
  amenities: string[];
  rating: number;
  packages: TrainingPackage[];
}

interface InPersonTrainingCardProps {
  location: TrainingLocation;
  packages: TrainingPackage[];
  onBookSession: (sessionType: string, location: string) => void;
  className?: string;
}

/**
 * InPersonTrainingCard - Location-based training session pricing
 * Separate from monthly subscriptions with location-specific theming
 */
export function InPersonTrainingCard({
  location,
  packages,
  onBookSession,
  className
}: InPersonTrainingCardProps) {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [isHovered, setIsHovered] = useState(false);
  
  // Get user tier for proper booking access
  const userMetadata = user?.user_metadata as Record<string, unknown> | undefined;
  const userTier = (userMetadata?.subscription_tier as string) || (userMetadata?.profile as Record<string, unknown>)?.subscription_tier as string || 'core';
  
  // Map location IDs to Calendly event types
  const getEventType = (locationId: string) => {
    const eventTypeMap = {
      'city-sports-club': 'personalTrainingSemiWeekly' as const,
      'home-training': 'personalTrainingMonthly' as const,  
      'gym-partnership': 'personalTrainingMonthly' as const
    };
    return eventTypeMap[locationId as keyof typeof eventTypeMap] || 'personalTrainingMonthly' as const;
  };

  // Location-specific styling
  const locationStyles = {
    'your-gym': {
      gradient: 'from-gray-600/20 to-gray-700/20',
      border: 'border-gray-600/50',
      badge: 'bg-gray-600/20 text-gray-300',
      button: 'bg-gray-600 hover:bg-gray-700',
      glow: 'hover:shadow-gray-600/20'
    },
    'city-sports': {
      gradient: 'from-blue-600/20 to-blue-700/20',
      border: 'border-blue-600/50',
      badge: 'bg-blue-600/20 text-blue-300',
      button: 'bg-blue-600 hover:bg-blue-700',
      glow: 'hover:shadow-blue-600/20'
    },
    'home': {
      gradient: 'from-green-600/20 to-green-700/20',
      border: 'border-green-600/50',
      badge: 'bg-green-600/20 text-green-300',
      button: 'bg-green-600 hover:bg-green-700',
      glow: 'hover:shadow-green-600/20'
    }
  };

  const style = locationStyles['home']; // Default style

  const features = {
    'your-gym': [
      'Choose your preferred gym location',
      'Flexible scheduling options',
      'Equipment already available',
      'Familiar training environment'
    ],
    'city-sports': [
      'Premium facility access included',
      'State-of-the-art equipment',
      'Recovery amenities available',
      'Professional atmosphere'
    ],
    'home': [
      'Maximum convenience',
      'Privacy and comfort',
      'Equipment provided if needed',
      'No commute time'
    ]
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      whileHover={{ y: -4 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      <Card 
        className={cn(
          "relative overflow-hidden transition-all duration-300",
          "bg-gray-900/50 backdrop-blur-sm",
          style.border,
          style.glow,
          "hover:shadow-2xl",
          location.isPreferred && "scale-105 shadow-2xl",
          className
        )}
      >
        {/* Background Gradient */}
        <div className={cn(
          "absolute inset-0 bg-gradient-to-br opacity-10",
          style.gradient
        )} />
        
        {/* Popular Badge */}
        {location.isPreferred && (
          <div className="absolute -top-4 -right-4">
            <Badge className={cn(
              "px-4 py-1 text-sm font-bold transform rotate-12",
              style.badge
            )}>
              PREFERRED
            </Badge>
          </div>
        )}

        <CardHeader className="relative z-10 pb-4">
          <div className="flex items-center justify-between mb-4">
            <div className={cn("text-3xl", style.badge.replace('bg-', 'text-').replace('/20', ''))}>
              <MapPin />
            </div>
            <div className="flex items-center">
              <Star className="w-4 h-4 fill-yellow-500 text-yellow-500" />
              <span className="text-sm text-gray-300 ml-1">{location.rating}</span>
            </div>
          </div>
          
          <h3 className="text-2xl font-bold text-white">
            {location.name}
          </h3>
          <p className="text-gray-400 text-sm mt-1">
            {location.address}
          </p>
        </CardHeader>

        <CardContent className="relative z-10 space-y-6">
          {/* Pricing */}
          <div className="space-y-2">
            <div className="flex items-baseline">
              <span className="text-4xl font-bold text-white">
                ${packages[0]?.pricePerSession || 120}
              </span>
              <span className="text-gray-400 ml-2">/session</span>
            </div>
            <p className="text-sm text-gray-500">
              Package discounts available
            </p>
          </div>

          {/* Session Details */}
          <div className="grid grid-cols-2 gap-3 py-3 border-y border-gray-800">
            <div className="flex items-center text-sm">
              <Clock className="w-4 h-4 text-gray-400 mr-2" />
              <span className="text-gray-300">60 minutes</span>
            </div>
            <div className="flex items-center text-sm">
              <Users className="w-4 h-4 text-gray-400 mr-2" />
              <span className="text-gray-300">1-on-1</span>
            </div>
          </div>

          {/* Features */}
          <ul className="space-y-2">
            {location.amenities.map((feature, index) => (
              <li key={index} className="flex items-start">
                <MapPin className={cn(
                  "w-4 h-4 mt-0.5 mr-2 flex-shrink-0",
                  style.badge.replace('bg-', 'text-').replace('/20', '')
                )} />
                <span className="text-gray-300 text-sm">{feature}</span>
              </li>
            ))}
          </ul>

          {/* Availability Note */}
          <div className="pt-2 border-t border-gray-800">
            <p className="text-xs text-gray-500">
              Available in select areas. Schedule flexibility based on coach availability.
            </p>
          </div>
        </CardContent>

        <CardFooter className="relative z-10 pt-4">
          <CalendlyBooking
            eventType={getEventType(location.id)}
            buttonText="Book Session"
            buttonVariant="default"
            className={cn(
              "w-full py-6 text-lg font-semibold transition-all",
              style.button,
              isHovered && "scale-105"
            )}
            prefillData={{
              name: (user?.user_metadata as Record<string, unknown>)?.client_name as string | undefined || user?.email?.split('@')[0],
              email: user?.email,
              customAnswers: {
                tier: userTier,
                preferredLocation: location.id,
                sessionType: 'single'
              }
            }}
          />
        </CardFooter>
      </Card>
    </motion.div>
  );
}