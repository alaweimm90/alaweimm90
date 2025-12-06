import React, { useState } from 'react';
import { ClientProfile, TierFeatures } from '@/types/fitness';
import { Button } from '@/ui/atoms/Button';
import { Dumbbell } from 'lucide-react';
import { useMobileDetection } from '@/hooks/useMobileDetection';
import MobileWorkoutInterface from '@/components/mobile/MobileWorkoutInterface';
import { useComponentPerformance } from '@/utils/monitoring';

interface DashboardContentProps {
  clientProfile: ClientProfile;
  tierFeatures: TierFeatures | null;
  CoreDashboard: React.ComponentType<{ client: ClientProfile }>;
  InteractiveDashboard: React.ComponentType<{ 
    client: ClientProfile; 
    tier: 'adaptive' | 'performance' | 'longevity'; 
    tierFeatures: TierFeatures;
  }>;
}

export const DashboardContent: React.FC<DashboardContentProps> = ({
  clientProfile,
  tierFeatures,
  CoreDashboard,
  InteractiveDashboard
}) => {
  useComponentPerformance('DashboardContent');
  
  const { isMobile } = useMobileDetection();
  const [showMobileWorkout, setShowMobileWorkout] = useState(false);

  // Sample exercises for mobile workout interface
  const sampleExercises = [
    {
      id: '1',
      name: 'Bench Press',
      sets: 4,
      reps: '8-10',
      weight: '80kg',
      restTime: 120,
      instructions: [
        'Lie flat on the bench with feet firmly on the ground',
        'Grip the bar slightly wider than shoulder width',
        'Lower the bar to chest level with control',
        'Press up explosively while maintaining proper form'
      ]
    },
    {
      id: '2',
      name: 'Pull-ups',
      sets: 3,
      reps: '6-8',
      weight: 'Bodyweight',
      restTime: 90,
      instructions: [
        'Hang from the bar with arms fully extended',
        'Pull yourself up until chin is over the bar',
        'Lower yourself down with control',
        'Keep core engaged throughout the movement'
      ]
    },
    {
      id: '3',
      name: 'Dumbbell Rows',
      sets: 3,
      reps: '10-12',
      weight: '30kg',
      restTime: 90,
      instructions: [
        'Hinge at the hips and lean forward',
        'Pull dumbbells to your sides',
        'Squeeze shoulder blades together',
        'Lower with control'
      ]
    }
  ];

  // Mobile workout interface overlay
  if (isMobile && showMobileWorkout) {
    return (
      <MobileWorkoutInterface
        exercises={sampleExercises}
        onComplete={() => setShowMobileWorkout(false)}
        isLiveCoaching={tierFeatures.biomarker_integration}
      />
    );
  }

  // Handle null tierFeatures gracefully
  if (!tierFeatures) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <p className="text-muted-foreground">Loading tier features...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="container mx-auto px-4 py-8">
        {/* Render appropriate dashboard based on tier */}
        {tierFeatures.dashboard_type === 'static_fixed' ? (
          <CoreDashboard client={clientProfile} />
        ) : (
          <InteractiveDashboard 
            client={clientProfile} 
            tier={clientProfile.subscription_tier as 'adaptive' | 'performance' | 'longevity'} 
            tierFeatures={tierFeatures}
          />
        )}
      </div>

      {/* Mobile Quick Actions */}
      {isMobile && (
        <div className="fixed bottom-20 right-4 z-40">
          <Button
            onClick={() => setShowMobileWorkout(true)}
            className="rounded-full h-14 w-14 shadow-lg"
            size="lg"
            aria-label="Start mobile workout"
          >
            <Dumbbell className="h-6 w-6" />
          </Button>
        </div>
      )}
    </>
  );
};