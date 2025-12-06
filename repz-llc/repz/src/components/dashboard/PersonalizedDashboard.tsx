import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';

interface PersonalizedDashboardProps {
  className?: string;
}

interface Workout {
  id: string;
  client_id: string;
  exercises_completed: number;
  performance_score: number;
}

interface Analytics {
  totalUsers: number;
  activeUsers: number;
  revenue: number;
  conversionRate: number;
}

export const PersonalizedDashboard: React.FC<PersonalizedDashboardProps> = ({ className }) => {
  const { user, userTier, loading } = useAuth();
  const [workouts, setWorkouts] = useState<Workout[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [isLoadingWorkouts, setIsLoadingWorkouts] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchWorkouts = async () => {
      if (!user?.id) return;

      try {
        setIsLoadingWorkouts(true);
        const { data, error } = await supabase
          .from('workouts')
          .select('*')
          .eq('client_id', user.id);

        if (error) throw error;
        setWorkouts(data || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load workouts');
      } finally {
        setIsLoadingWorkouts(false);
      }
    };

    fetchWorkouts();
  }, [user?.id]);

  useEffect(() => {
    const fetchAnalytics = async () => {
      if (userTier !== 'performance' && userTier !== 'longevity') return;

      try {
        const { data, error } = await supabase.functions.invoke('get-analytics');
        if (error) throw error;
        setAnalytics(data);
      } catch (err) {
        console.error('Failed to load analytics:', err);
      }
    };

    fetchAnalytics();
  }, [userTier]);

  if (loading) {
    return (
      <div className={className} data-testid="dashboard-grid">
        <Card>
          <CardContent>
            <p>Loading dashboard...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className={className} data-testid="dashboard-grid">
        <Card>
          <CardContent>
            <p className="text-red-500">Error: {error}</p>
            <p className="text-sm text-gray-500 mt-2">Please check your network connection and try again.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const renderCoreDashboard = () => (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Welcome to Your Core Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold">Progress Photos</h3>
              <p className="text-sm text-gray-600">Track your transformation visually</p>
            </div>
            <div>
              <h3 className="font-semibold">Nutrition Tracking</h3>
              <p className="text-sm text-gray-600">Monitor your daily macros and calories</p>
            </div>
            <div>
              <h3 className="font-semibold">Workout Plan</h3>
              <p className="text-sm text-gray-600">Your personalized training program</p>
            </div>
          </div>
        </CardContent>
      </Card>
      {renderWorkoutSection()}
    </>
  );

  const renderAdaptiveDashboard = () => (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Welcome to Your Adaptive Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold">Weekly Check-ins</h3>
              <p className="text-sm text-gray-600">Schedule your next check-in</p>
            </div>
            <div>
              <h3 className="font-semibold">Form Analysis</h3>
              <p className="text-sm text-gray-600">AI-powered movement analysis</p>
            </div>
            <div>
              <h3 className="font-semibold">Body Composition</h3>
              <p className="text-sm text-gray-600">Track your body metrics over time</p>
            </div>
            <div>
              <h3 className="font-semibold">Wearable Integration</h3>
              <p className="text-sm text-gray-600">Connect your fitness devices</p>
            </div>
          </div>
        </CardContent>
      </Card>
      {renderWorkoutSection()}
    </>
  );

  const renderPerformanceDashboard = () => (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Welcome to Your Performance Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold">AI Coaching</h3>
              <p className="text-sm text-gray-600">24/7 intelligent fitness assistant</p>
            </div>
            <div>
              <h3 className="font-semibold">Live Sessions</h3>
              <p className="text-sm text-gray-600">Real-time training sessions</p>
            </div>
            <div>
              <h3 className="font-semibold">Advanced Analytics</h3>
              <p className="text-sm text-gray-600">Deep insights into your performance</p>
            </div>
            <div>
              <h3 className="font-semibold">PEDs Protocols</h3>
              <p className="text-sm text-gray-600">Performance enhancement guidance</p>
            </div>
          </div>
        </CardContent>
      </Card>
      {renderWorkoutSection()}
      {renderAnalyticsSection()}
    </>
  );

  const renderLongevityDashboard = () => (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Welcome to Your Longevity Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold">Bioregulator Protocols</h3>
              <p className="text-sm text-gray-600">Advanced peptide and bioregulator therapies</p>
            </div>
            <div>
              <h3 className="font-semibold">Peptide Therapy</h3>
              <p className="text-sm text-gray-600">Personalized peptide regimens</p>
            </div>
            <div>
              <h3 className="font-semibold">Exclusive Research</h3>
              <p className="text-sm text-gray-600">Access to cutting-edge longevity studies</p>
            </div>
            <div>
              <h3 className="font-semibold">Concierge Service</h3>
              <p className="text-sm text-gray-600">Premium 24/7 support</p>
            </div>
          </div>
        </CardContent>
      </Card>
      {renderWorkoutSection()}
      {renderAnalyticsSection()}
    </>
  );

  const renderWorkoutSection = () => (
    <Card>
      <CardHeader>
        <CardTitle>Recent Workouts</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoadingWorkouts ? (
          <p>Loading workouts...</p>
        ) : workouts.length === 0 ? (
          <div>
            <p>No workouts recorded yet</p>
            <button className="mt-2 px-4 py-2 bg-blue-500 text-white rounded">
              Create New Workout
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            {workouts.map((workout) => (
              <div key={workout.id} className="p-3 border rounded">
                <p className="font-semibold">{workout.exercises_completed} exercises completed</p>
                <p className="text-sm text-gray-600">{workout.performance_score}% performance score</p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );

  const renderAnalyticsSection = () => {
    if (!analytics) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Analytics Dashboard</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Analytics available for Performance and Longevity tiers</p>
            <button className="mt-2 px-4 py-2 bg-blue-500 text-white rounded">
              Upgrade to Access Analytics
            </button>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card>
        <CardHeader>
          <CardTitle>Analytics Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Total Users</p>
              <p className="text-2xl font-bold">{analytics.totalUsers}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Active Users</p>
              <p className="text-2xl font-bold">{analytics.activeUsers}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Revenue</p>
              <p className="text-2xl font-bold">${analytics.revenue.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Conversion Rate</p>
              <p className="text-2xl font-bold">{analytics.conversionRate}%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderDashboardByTier = () => {
    switch (userTier) {
      case 'core':
        return renderCoreDashboard();
      case 'adaptive':
        return renderAdaptiveDashboard();
      case 'performance':
        return renderPerformanceDashboard();
      case 'longevity':
        return renderLongevityDashboard();
      default:
        return (
          <Card>
            <CardHeader>
              <CardTitle>Welcome to Your Dashboard</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Your personalized fitness dashboard is being prepared...</p>
            </CardContent>
          </Card>
        );
    }
  };

  return (
    <div className={`grid gap-6 md:grid-cols-2 ${className}`} data-testid="dashboard-grid">
      {renderDashboardByTier()}
    </div>
  );
};

export default PersonalizedDashboard;
