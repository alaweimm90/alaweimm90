import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { CoachMarketplace } from './CoachMarketplace';
import { CoachReviews } from './CoachReviews';
import { CoachAvailability } from './CoachAvailability';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierGate } from '@/components/auth/TierGate';

export const MarketplaceHub: React.FC = () => {
  const [selectedCoach, setSelectedCoach] = useState<string | null>(null);
  const tierAccess = useTierAccess();

  // Mock selected coach data
  const selectedCoachData = {
    id: '1',
    name: 'Sarah Johnson',
    rating: 4.9,
    totalReviews: 127
  };

  return (
      <TierGate 
        requiredTier="adaptive"
        feature="weekly_checkin"
      fallback={
        <Card>
          <CardHeader>
            <CardTitle>Coach Marketplace</CardTitle>
            <CardDescription>
              Browse and connect with certified fitness coaches
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-4">
              Access to our coach marketplace is available with Adaptive tier and above.
            </p>
            <p className="text-sm text-muted-foreground">
              Upgrade your plan to browse coaches, read reviews, and book sessions.
            </p>
          </CardContent>
        </Card>
      }
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Coach Marketplace</h1>
          <p className="text-muted-foreground">
            Find the perfect coach to help you reach your fitness goals
          </p>
        </div>

        <Tabs defaultValue="browse" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="browse">Browse Coaches</TabsTrigger>
            <TabsTrigger 
              value="reviews" 
              disabled={!selectedCoach}
              className={!selectedCoach ? 'opacity-50' : ''}
            >
              Reviews & Ratings
            </TabsTrigger>
            <TabsTrigger 
              value="booking" 
              disabled={!selectedCoach}
              className={!selectedCoach ? 'opacity-50' : ''}
            >
              Book Session
            </TabsTrigger>
          </TabsList>

          <TabsContent value="browse">
            <CoachMarketplace />
          </TabsContent>

          <TabsContent value="reviews">
            {selectedCoach ? (
              <CoachReviews
                coachId={selectedCoachData.id}
                coachName={selectedCoachData.name}
                averageRating={selectedCoachData.rating}
                totalReviews={selectedCoachData.totalReviews}
              />
            ) : (
              <Card>
                <CardContent className="text-center py-12">
                  <p className="text-muted-foreground">
                    Select a coach from the browse tab to view their reviews.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="booking">
            {selectedCoach ? (
              <CoachAvailability
                coachId={selectedCoachData.id}
                coachName={selectedCoachData.name}
              />
            ) : (
              <Card>
                <CardContent className="text-center py-12">
                  <p className="text-muted-foreground">
                    Select a coach from the browse tab to book a session.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Quick Stats */}
        <div className="grid md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">50+</div>
              <div className="text-sm text-muted-foreground">Certified Coaches</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">4.8</div>
              <div className="text-sm text-muted-foreground">Average Rating</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">1,200+</div>
              <div className="text-sm text-muted-foreground">Happy Clients</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">24/7</div>
              <div className="text-sm text-muted-foreground">Support Available</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </TierGate>
  );
};