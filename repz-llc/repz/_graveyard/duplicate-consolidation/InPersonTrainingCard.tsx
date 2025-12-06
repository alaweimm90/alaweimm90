import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { MapPin, Clock, Users, Star, Calendar } from 'lucide-react';
import { cn } from '@/lib/utils';

interface InPersonTrainingCardProps {
  location: {
    id: string;
    name: string;
    address: string;
    isPreferred?: boolean;
    amenities: string[];
    rating: number;
    imageUrl?: string;
  };
  packages: {
    id: string;
    name: string;
    sessionsCount: number;
    pricePerSession: number;
    totalPrice: number;
    discount?: number;
    isPopular?: boolean;
  }[];
  onBookSession: (locationId: string, packageId: string) => void;
  className?: string;
}

export const InPersonTrainingCard: React.FC<InPersonTrainingCardProps> = ({
  location,
  packages,
  onBookSession,
  className
}) => {
  const getLocationTheme = (locationName: string) => {
    if (locationName.toLowerCase().includes('city sports')) {
      return {
        bgGradient: 'from-amber-500/10 to-orange-500/10',
        borderColor: 'border-amber-500/30',
        accentColor: 'text-amber-400',
        buttonColor: 'bg-amber-500 hover:bg-amber-600'
      };
    } else if (locationName.toLowerCase().includes('home')) {
      return {
        bgGradient: 'from-green-500/10 to-emerald-500/10',
        borderColor: 'border-green-500/30',
        accentColor: 'text-green-400',
        buttonColor: 'bg-green-500 hover:bg-green-600'
      };
    } else {
      return {
        bgGradient: 'from-blue-500/10 to-cyan-500/10',
        borderColor: 'border-blue-500/30',
        accentColor: 'text-blue-400',
        buttonColor: 'bg-blue-500 hover:bg-blue-600'
      };
    }
  };

  const theme = getLocationTheme(location.name);

  return (
    <Card className={cn(
      "relative overflow-hidden transition-all duration-300 hover:scale-105",
      `bg-gradient-to-br ${theme.bgGradient}`,
      `border-2 ${theme.borderColor}`,
      "backdrop-blur-sm",
      className
    )}>
      {location.isPreferred && (
        <div className="absolute -top-2 left-1/2 transform -translate-x-1/2 z-10">
          <Badge className="bg-amber-500 text-white px-3 py-1 text-xs font-bold">
            ⭐ COACH'S CHOICE
          </Badge>
        </div>
      )}

      <CardHeader className="pb-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className={cn("text-xl font-bold", theme.accentColor)}>
              {location.name}
            </CardTitle>
            <CardDescription className="flex items-center gap-1 mt-1 text-muted-foreground">
              <MapPin className="h-4 w-4" />
              {location.address}
            </CardDescription>
          </div>
          <div className="flex items-center gap-1">
            <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
            <span className="text-sm font-medium">{location.rating}</span>
          </div>
        </div>

        {/* Amenities */}
        <div className="flex flex-wrap gap-1 mt-3">
          {location.amenities.slice(0, 3).map((amenity, index) => (
            <Badge key={index} variant="secondary" className="text-xs">
              {amenity}
            </Badge>
          ))}
          {location.amenities.length > 3 && (
            <Badge variant="secondary" className="text-xs">
              +{location.amenities.length - 3} more
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Package Options */}
        <div className="space-y-3">
          <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
            Training Packages
          </h4>
          
          {packages.map((pkg) => (
            <div key={pkg.id} className="relative">
              {pkg.isPopular && (
                <div className="absolute -top-2 right-2 z-10">
                  <Badge className="bg-orange-500 text-white text-xs">
                    MOST POPULAR
                  </Badge>
                </div>
              )}
              
              <div className={cn(
                "p-4 rounded-lg border transition-all duration-200",
                pkg.isPopular ? "border-orange-500/30 bg-orange-500/5" : "border-white/10 bg-white/5",
                "hover:border-white/20"
              )}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Users className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium">{pkg.name}</span>
                  </div>
                  {pkg.discount && (
                    <Badge className="bg-green-500 text-white text-xs">
                      {pkg.discount}% OFF
                    </Badge>
                  )}
                </div>
                
                <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                  <Clock className="h-4 w-4" />
                  <span>{pkg.sessionsCount} sessions</span>
                  <span>•</span>
                  <span className={theme.accentColor}>
                    ${pkg.pricePerSession}/session
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-lg font-bold">${pkg.totalPrice}</span>
                    {pkg.discount && (
                      <span className="text-sm text-muted-foreground ml-2 line-through">
                        ${Math.round(pkg.totalPrice / (1 - pkg.discount / 100))}
                      </span>
                    )}
                  </div>
                  
                  <Button
                    onClick={() => onBookSession(location.id, pkg.id)}
                    className={cn(
                      "flex items-center gap-2",
                      theme.buttonColor
                    )}
                    size="sm"
                  >
                    <Calendar className="h-4 w-4" />
                    Book Now
                  </Button>
                </div>

                {pkg.discount && (
                  <div className="mt-2 text-xs text-green-400">
                    Save ${Math.round(pkg.totalPrice * pkg.discount / 100)} total
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};