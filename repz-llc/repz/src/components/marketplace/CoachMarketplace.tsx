import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/ui/atoms/Avatar';
import { Star, MapPin, Calendar, DollarSign, Users } from 'lucide-react';
import { TierGate } from '@/components/auth/TierGate';
import { useTierAccess } from '@/hooks/useTierAccess';

const mockCoaches = [
  {
    id: 1,
    name: 'Dr. Sarah Chen',
    specialty: 'Longevity & Anti-Aging',
    rating: 4.9,
    reviews: 156,
    location: 'Beverly Hills, CA',
    price: 300,
    availability: 'Available',
    image: '/lovable-uploads/user-profile.png',
    certifications: ['MD', 'Anti-Aging Specialist', 'Hormone Optimization'],
    description: 'Leading expert in longevity protocols and biomarker optimization.'
  },
  {
    id: 2,
    name: 'Marcus Rodriguez',
    specialty: 'Performance Enhancement',
    rating: 4.8,
    reviews: 203,
    location: 'Austin, TX',
    price: 250,
    availability: 'Available',
    image: '/lovable-uploads/user-profile.png',
    certifications: ['CSCS', 'PED Protocols', 'Athletic Performance'],
    description: 'Former Olympic trainer specializing in peak performance optimization.'
  },
  {
    id: 3,
    name: 'Dr. Emily Watson',
    specialty: 'Precision Medicine',
    rating: 4.9,
    reviews: 89,
    location: 'New York, NY',
    price: 350,
    availability: 'Booking Soon',
    image: '/lovable-uploads/user-profile.png',
    certifications: ['MD', 'Precision Medicine', 'Genetic Analysis'],
    description: 'Cutting-edge precision medicine approach to health optimization.'
  }
];

export function CoachMarketplace() {
  const { hasInPersonTraining } = useTierAccess();
  const [selectedSpecialty, setSelectedSpecialty] = useState('all');

  if (!hasInPersonTraining) {
    return (
      <TierGate 
        requiredTier="longevity" 
        feature="Elite Coach Marketplace"
      >
        <div className="p-6">Coach marketplace would be here</div>
      </TierGate>
    );
  }

  const specialties = ['all', 'Longevity & Anti-Aging', 'Performance Enhancement', 'Precision Medicine'];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {specialties.map((specialty) => (
          <Button
            key={specialty}
            variant={selectedSpecialty === specialty ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedSpecialty(specialty)}
            className="capitalize"
          >
            {specialty}
          </Button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {mockCoaches.map((coach) => (
          <Card key={coach.id} className="glass-tier-card hover:scale-105 transition-transform duration-300">
            <CardHeader>
              <div className="flex items-start gap-3">
                <Avatar className="w-12 h-12">
                  <AvatarImage src={coach.image} />
                  <AvatarFallback>{coach.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <CardTitle className="text-lg">{coach.name}</CardTitle>
                  <CardDescription className="font-medium text-tier-adaptive">
                    {coach.specialty}
                  </CardDescription>
                </div>
                <Badge 
                  variant={coach.availability === 'Available' ? 'default' : 'secondary'}
                  className="text-xs"
                >
                  {coach.availability}
                </Badge>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">{coach.description}</p>
              
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  <span className="font-medium">{coach.rating}</span>
                  <span className="text-muted-foreground">({coach.reviews})</span>
                </div>
                <div className="flex items-center gap-1 text-muted-foreground">
                  <MapPin className="w-4 h-4" />
                  {coach.location}
                </div>
              </div>

              <div className="flex flex-wrap gap-1">
                {coach.certifications.map((cert, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    {cert}
                  </Badge>
                ))}
              </div>

              <div className="flex items-center justify-between pt-2 border-t">
                <div className="flex items-center gap-1">
                  <DollarSign className="w-4 h-4 text-tier-longevity" />
                  <span className="font-bold text-lg">${coach.price}</span>
                  <span className="text-sm text-muted-foreground">/session</span>
                </div>
                <Button size="sm" className="bg-tier-adaptive hover:bg-tier-adaptive/80" onClick={() => console.log("CoachMarketplace button clicked")}>
                  <Calendar className="w-4 h-4 mr-2" />
                  Book Session
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="w-5 h-5 text-tier-performance" />
            Elite Coach Benefits
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4">
              <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-tier-adaptive/20 flex items-center justify-center">
                <Star className="w-6 h-6 text-tier-adaptive" />
              </div>
              <h3 className="font-semibold mb-1">World-Class Expertise</h3>
              <p className="text-sm text-muted-foreground">
                Access to the top 1% of coaches and medical professionals
              </p>
            </div>
            <div className="text-center p-4">
              <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-tier-performance/20 flex items-center justify-center">
                <MapPin className="w-6 h-6 text-tier-performance" />
              </div>
              <h3 className="font-semibold mb-1">In-Person Sessions</h3>
              <p className="text-sm text-muted-foreground">
                Premium 1-on-1 training at elite facilities worldwide
              </p>
            </div>
            <div className="text-center p-4">
              <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-tier-longevity/20 flex items-center justify-center">
                <DollarSign className="w-6 h-6 text-tier-longevity" />
              </div>
              <h3 className="font-semibold mb-1">Concierge Service</h3>
              <p className="text-sm text-muted-foreground">
                White-glove booking and premium facility access
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}