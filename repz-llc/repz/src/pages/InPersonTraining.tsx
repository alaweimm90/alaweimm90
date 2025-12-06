import React, { useState } from 'react';
import { RepzLogo } from '@/ui/organisms/RepzLogo';
import { InPersonTrainingCard } from '@/components/pricing/InPersonTrainingCard';
import { CalendlyBooking } from '@/components/booking/CalendlyBooking';
import { Button } from '@/ui/atoms/Button';
import { ArrowLeft, MapPin, Filter } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';

// Location and package data based on superprompt specifications
const TRAINING_LOCATIONS = [
  {
    id: 'city-sports-club',
    name: 'City Sports Club',
    address: 'Downtown San Francisco',
    isPreferred: true,
    amenities: ['Premium Equipment', 'Sauna', 'Pool', 'Recovery Room'],
    rating: 4.9,
    packages: [
      {
        id: 'single-session',
        name: 'Single Session',
        sessionsCount: 1,
        pricePerSession: 79,
        totalPrice: 79
      },
      {
        id: 'bulk-package',
        name: 'Bulk Package',
        sessionsCount: 3,
        pricePerSession: 71.10,
        totalPrice: 213.30,
        discount: 10,
        isPopular: true
      },
      {
        id: 'monthly-unlimited',
        name: 'Monthly Unlimited',
        sessionsCount: 8,
        pricePerSession: 65,
        totalPrice: 520,
        discount: 18
      }
    ]
  },
  {
    id: 'home-training',
    name: 'Home Training',
    address: 'Your Location',
    amenities: ['Mobile Equipment', 'Flexible Scheduling', 'Personalized Setup'],
    rating: 4.8,
    packages: [
      {
        id: 'home-single',
        name: 'Single Session',
        sessionsCount: 1,
        pricePerSession: 89,
        totalPrice: 89
      },
      {
        id: 'home-package',
        name: 'Weekly Package',
        sessionsCount: 2,
        pricePerSession: 82,
        totalPrice: 164,
        discount: 8
      }
    ]
  },
  {
    id: 'gym-partnership',
    name: 'Partner Gym Network',
    address: 'Multiple SF Locations',
    amenities: ['Multiple Locations', 'Standard Equipment', 'Group Classes'],
    rating: 4.6,
    packages: [
      {
        id: 'gym-single',
        name: 'Single Session',
        sessionsCount: 1,
        pricePerSession: 75,
        totalPrice: 75
      },
      {
        id: 'gym-monthly',
        name: 'Monthly Plan',
        sessionsCount: 4,
        pricePerSession: 68,
        totalPrice: 272,
        discount: 9
      }
    ]
  }
];

export default function InPersonTraining() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null);

  // Get user tier for proper booking access
  const userTier = (user as { subscription_tier?: string; profile?: { subscription_tier?: string } } | null)?.subscription_tier || (user as { subscription_tier?: string; profile?: { subscription_tier?: string } } | null)?.profile?.subscription_tier || 'core';

  const handleBookSession = (locationId: string, packageId: string) => {
    console.log('Booking session:', { locationId, packageId });
    // Use proper Calendly booking based on location
    const eventTypeMap = {
      'city-sports-club': 'cityTraining' as const,
      'home-training': 'homeTraining' as const,  
      'gym-partnership': 'gymTraining' as const
    };
    
    const eventType = eventTypeMap[locationId as keyof typeof eventTypeMap] || 'personalTrainingMonthly' as const;
    return { eventType, locationId, packageId };
  };

  const filteredLocations = selectedLocation 
    ? TRAINING_LOCATIONS.filter(loc => loc.id === selectedLocation)
    : TRAINING_LOCATIONS;

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate('/')}
                className="text-white hover:text-orange-400 flex items-center gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Home
              </Button>
              <RepzLogo />
            </div>
            
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedLocation(null)}
                className="flex items-center gap-2"
              >
                <Filter className="h-4 w-4" />
                All Locations
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            In-Person Training
            <span className="block text-2xl md:text-3xl text-orange-400 mt-2">
              Elite Bay Area Sessions
            </span>
          </h1>
          
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Experience personalized training with our elite coaches in premium San Francisco locations. 
            Available exclusively for Longevity tier members.
          </p>

          <div className="flex items-center justify-center gap-6 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4 text-orange-400" />
              <span>Bay Area Locations</span>
            </div>
            <span>•</span>
            <span>Premium Equipment</span>
            <span>•</span>
            <span>Bulk Discounts Available</span>
          </div>
        </div>
      </section>

      {/* Location Cards */}
      <section className="py-12 px-4">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {filteredLocations.map((location) => (
              <InPersonTrainingCard
                key={location.id}
                location={location}
                packages={location.packages}
                onBookSession={handleBookSession}
                className="w-full"
              />
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-16 px-4 bg-gradient-to-r from-orange-500/10 to-amber-500/10">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-4">
              Why Choose In-Person Training?
            </h2>
            <p className="text-gray-300 max-w-2xl mx-auto">
              Experience the difference of hands-on coaching with our elite training methodology
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: 'Expert Form Correction',
                description: 'Real-time adjustments for optimal technique and injury prevention'
              },
              {
                title: 'Premium Equipment Access',
                description: 'State-of-the-art facilities and specialized training tools'
              },
              {
                title: 'Personalized Programming',
                description: 'Custom workouts adapted to your goals and fitness level'
              },
              {
                title: 'Motivation & Accountability',
                description: 'Stay committed with personal coaching and support'
              },
              {
                title: 'Flexible Scheduling',
                description: 'Book sessions that fit your busy lifestyle'
              },
              {
                title: 'Bulk Discounts',
                description: 'Save more with package deals and monthly unlimited options'
              }
            ].map((benefit, index) => (
              <div
                key={index}
                className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-6 text-center"
              >
                <h3 className="text-xl font-semibold text-white mb-2">
                  {benefit.title}
                </h3>
                <p className="text-gray-400">
                  {benefit.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center">
          <div className="bg-gradient-to-r from-orange-500/20 to-amber-500/20 border border-orange-500/30 rounded-2xl p-8 max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold text-white mb-4">
              Ready to Start Training?
            </h2>
            <p className="text-gray-300 mb-6">
              Book your first session and experience the difference of elite personal training
            </p>
            <CalendlyBooking
              eventType="personalTrainingMonthly"
              buttonText="Schedule Your Session"
              buttonVariant="default"
              className="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 text-lg"
              prefillData={{
                name: (user && typeof user === 'object' && 'profile' in user && user.profile && typeof user.profile === 'object' && 'client_name' in user.profile ? user.profile.client_name as string : undefined) || user?.email?.split('@')[0],
                email: user?.email,
                customAnswers: {
                  tier: userTier,
                  preferredLocation: 'city-sports-club'
                }
              }}
            />
          </div>
        </div>
      </section>
    </div>
  );
}