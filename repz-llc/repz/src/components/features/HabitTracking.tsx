import React, { useState } from 'react';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { Input } from "@/ui/atoms/Input";
import { Label } from "@/ui/atoms/Label";
import { Checkbox } from "@/ui/atoms/Checkbox";
import { Progress } from "@/components/ui/progress";
import { 
  Target, 
  Plus, 
  Clock, 
  Calendar,
  TrendingUp,
  CheckCircle2,
  Circle,
  Flame,
  Award,
  RotateCcw
} from 'lucide-react';
import { toast } from 'sonner';
import { useTierAccess } from '@/hooks/useTierAccess';

interface HabitTrackingProps {
  className?: string;
}

interface Habit {
  id: string;
  name: string;
  description: string;
  category: 'fitness' | 'nutrition' | 'wellness' | 'mindset';
  frequency: 'daily' | 'weekly';
  target: number; // target per frequency period
  currentStreak: number;
  longestStreak: number;
  completedToday: boolean;
  completedThisWeek: number;
  totalCompleted: number;
  createdAt: Date;
  lastCompleted?: Date;
}

const predefinedHabits = [
  {
    name: 'Morning Workout',
    description: 'Complete 30-45 minute workout session',
    category: 'fitness' as const,
    frequency: 'daily' as const,
    target: 1
  },
  {
    name: 'Drink 8 Glasses of Water',
    description: 'Stay hydrated throughout the day',
    category: 'wellness' as const,
    frequency: 'daily' as const,
    target: 1
  },
  {
    name: 'Track All Meals',
    description: 'Log all meals and snacks in nutrition app',
    category: 'nutrition' as const,
    frequency: 'daily' as const,
    target: 1
  },
  {
    name: 'Get 8 Hours Sleep',
    description: 'Maintain consistent sleep schedule',
    category: 'wellness' as const,
    frequency: 'daily' as const,
    target: 1
  },
  {
    name: 'Meal Prep Session',
    description: 'Prepare meals for the week',
    category: 'nutrition' as const,
    frequency: 'weekly' as const,
    target: 1
  },
  {
    name: '10 Minutes Meditation',
    description: 'Practice mindfulness and stress relief',
    category: 'mindset' as const,
    frequency: 'daily' as const,
    target: 1
  },
  {
    name: 'Resistance Training',
    description: 'Strength training or weightlifting session',
    category: 'fitness' as const,
    frequency: 'weekly' as const,
    target: 3
  },
  {
    name: 'Read Fitness Article',
    description: 'Learn something new about health/fitness',
    category: 'mindset' as const,
    frequency: 'weekly' as const,
    target: 2
  }
];

export const HabitTracking: React.FC<HabitTrackingProps> = ({ className }) => {
  const [habits, setHabits] = useState<Habit[]>([
    {
      id: '1',
      name: 'Morning Workout',
      description: 'Complete 30-45 minute workout session',
      category: 'fitness',
      frequency: 'daily',
      target: 1,
      currentStreak: 5,
      longestStreak: 12,
      completedToday: true,
      completedThisWeek: 5,
      totalCompleted: 45,
      createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
      lastCompleted: new Date()
    },
    {
      id: '2',
      name: 'Drink 8 Glasses of Water',
      description: 'Stay hydrated throughout the day',
      category: 'wellness',
      frequency: 'daily',
      target: 1,
      currentStreak: 3,
      longestStreak: 8,
      completedToday: false,
      completedThisWeek: 4,
      totalCompleted: 23,
      createdAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000), // 20 days ago
    }
  ]);
  
  const [showAddHabit, setShowAddHabit] = useState(false);
  const [newHabitName, setNewHabitName] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<Habit['category']>('fitness');
  
  const tierAccess = useTierAccess();

  const categories = [
    { id: 'fitness', name: 'Fitness', color: 'bg-blue-100 text-blue-800' },
    { id: 'nutrition', name: 'Nutrition', color: 'bg-green-100 text-green-800' },
    { id: 'wellness', name: 'Wellness', color: 'bg-purple-100 text-purple-800' },
    { id: 'mindset', name: 'Mindset', color: 'bg-repz-orange/10 text-repz-orange' }
  ];

  const markHabitComplete = (habitId: string) => {
    setHabits(prev => prev.map(habit => {
      if (habit.id === habitId) {
        const wasCompletedToday = habit.completedToday;
        const now = new Date();
        
        return {
          ...habit,
          completedToday: !wasCompletedToday,
          completedThisWeek: !wasCompletedToday 
            ? habit.completedThisWeek + 1 
            : Math.max(0, habit.completedThisWeek - 1),
          currentStreak: !wasCompletedToday 
            ? habit.currentStreak + 1 
            : Math.max(0, habit.currentStreak - 1),
          longestStreak: !wasCompletedToday 
            ? Math.max(habit.longestStreak, habit.currentStreak + 1)
            : habit.longestStreak,
          totalCompleted: !wasCompletedToday 
            ? habit.totalCompleted + 1 
            : Math.max(0, habit.totalCompleted - 1),
          lastCompleted: !wasCompletedToday ? now : habit.lastCompleted
        };
      }
      return habit;
    }));

    const habit = habits.find(h => h.id === habitId);
    if (habit) {
      toast.success(
        habit.completedToday
          ? `Unmarked "${habit.name}"`
          : `Completed "${habit.name}"!`
      );
    }
  };

  const addPredefinedHabit = (predefined: typeof predefinedHabits[0]) => {
    const newHabit: Habit = {
      id: crypto.randomUUID(),
      ...predefined,
      currentStreak: 0,
      longestStreak: 0,
      completedToday: false,
      completedThisWeek: 0,
      totalCompleted: 0,
      createdAt: new Date()
    };
    
    setHabits(prev => [...prev, newHabit]);
    toast.success(`Added "${predefined.name}" to your habits!`);
  };

  const addCustomHabit = () => {
    if (!newHabitName.trim()) {
      toast.error('Please enter a habit name');
      return;
    }

    const newHabit: Habit = {
      id: crypto.randomUUID(),
      name: newHabitName,
      description: '',
      category: selectedCategory,
      frequency: 'daily',
      target: 1,
      currentStreak: 0,
      longestStreak: 0,
      completedToday: false,
      completedThisWeek: 0,
      totalCompleted: 0,
      createdAt: new Date()
    };
    
    setHabits(prev => [...prev, newHabit]);
    setNewHabitName('');
    setShowAddHabit(false);
    toast.success(`Added "${newHabitName}" to your habits!`);
  };

  const removeHabit = (habitId: string) => {
    setHabits(prev => prev.filter(h => h.id !== habitId));
    toast.success('Habit removed');
  };

  const getHabitProgress = (habit: Habit) => {
    if (habit.frequency === 'daily') {
      return habit.completedToday ? 100 : 0;
    } else {
      return Math.min((habit.completedThisWeek / habit.target) * 100, 100);
    }
  };

  const getTotalStreak = () => {
    return habits.reduce((sum, habit) => sum + habit.currentStreak, 0);
  };

  const getCompletedToday = () => {
    return habits.filter(habit => habit.completedToday).length;
  };

  const getCategoryColor = (category: Habit['category']) => {
    return categories.find(c => c.id === category)?.color || 'bg-gray-100 text-gray-800';
  };

  if (!tierAccess.hasMinimumTier('adaptive')) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Habit Tracking
          </CardTitle>
          <CardDescription>
            Build and track healthy habits for lasting results
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="font-semibold text-gray-900 mb-2">Premium Feature</h3>
            <p className="text-gray-600 mb-4">
              Habit tracking is available with Adaptive tier or higher.
            </p>
            <Button variant="outline" onClick={() => console.log("HabitTracking button clicked")}>Upgrade Plan</Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-green-100 rounded-full">
                <CheckCircle2 className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <div className="text-2xl font-bold">{getCompletedToday()}</div>
                <div className="text-sm text-gray-600">Completed Today</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-repz-orange/10 rounded-full">
                <Flame className="h-6 w-6 text-repz-orange" />
              </div>
              <div>
                <div className="text-2xl font-bold">{getTotalStreak()}</div>
                <div className="text-sm text-gray-600">Total Streak Days</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-blue-100 rounded-full">
                <Target className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <div className="text-2xl font-bold">{habits.length}</div>
                <div className="text-sm text-gray-600">Active Habits</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Habits */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Your Habits</CardTitle>
              <CardDescription>Track your daily and weekly habit goals</CardDescription>
            </div>
            <Button
              onClick={() => setShowAddHabit(!showAddHabit)}
              className="flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Add Habit
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {habits.length > 0 ? (
            <div className="space-y-4">
              {habits.map((habit) => (
                <div key={habit.id} className="border rounded-lg p-4 space-y-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <button
                        onClick={() => markHabitComplete(habit.id)}
                        className="mt-1"
                      >
                        {habit.completedToday ? (
                          <CheckCircle2 className="h-6 w-6 text-green-600" />
                        ) : (
                          <Circle className="h-6 w-6 text-gray-400 hover:text-green-600" />
                        )}
                      </button>
                      <div>
                        <h3 className="font-semibold">{habit.name}</h3>
                        {habit.description && (
                          <p className="text-sm text-gray-600">{habit.description}</p>
                        )}
                        <div className="flex items-center gap-2 mt-2">
                          <Badge className={getCategoryColor(habit.category)}>
                            {habit.category}
                          </Badge>
                          <Badge variant="outline">
                            {habit.frequency === 'daily' ? 'Daily' : `${habit.target}x/week`}
                          </Badge>
                        </div>
                      </div>
                    </div>
                    <div className="text-right text-sm">
                      <div className="flex items-center gap-2">
                        <Flame className="h-4 w-4 text-repz-orange" />
                        <span className="font-semibold">{habit.currentStreak}</span>
                      </div>
                      <div className="text-gray-600">streak</div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>
                        {habit.frequency === 'daily' 
                          ? (habit.completedToday ? 'Completed' : 'Not completed')
                          : `${habit.completedThisWeek}/${habit.target} this week`
                        }
                      </span>
                      <span>{Math.round(getHabitProgress(habit))}%</span>
                    </div>
                    <Progress value={getHabitProgress(habit)} className="h-2" />
                  </div>

                  {/* Stats */}
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <span>Total: {habit.totalCompleted} times</span>
                    <span>Best streak: {habit.longestStreak} days</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeHabit(habit.id)}
                      className="text-red-600 hover:text-red-800 h-auto p-1"
                    >
                      Remove
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No habits yet. Add your first habit to get started!
            </div>
          )}

          {/* Add Habit Section */}
          {showAddHabit && (
            <div className="border-t pt-6 mt-6 space-y-4">
              <h3 className="font-semibold">Add New Habit</h3>
              
              {/* Quick Add Predefined Habits */}
              <div className="space-y-3">
                <Label>Quick Add</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {predefinedHabits.map((habit, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50">
                      <div>
                        <div className="font-medium text-sm">{habit.name}</div>
                        <div className="text-xs text-gray-600">
                          {habit.category} • {habit.frequency}
                        </div>
                      </div>
                      <Button
                        size="sm"
                        onClick={() => addPredefinedHabit(habit)}
                      >
                        Add
                      </Button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Custom Habit */}
              <div className="space-y-3">
                <Label>Or Create Custom</Label>
                <div className="flex gap-2">
                  <Input
                    placeholder="Enter habit name..."
                    value={newHabitName}
                    onChange={(e) => setNewHabitName(e.target.value)}
                    className="flex-1"
                  />
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value as Habit['category'])}
                    className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-repz-orange"
                  >
                    {categories.map((category) => (
                      <option key={category.id} value={category.id}>
                        {category.name}
                      </option>
                    ))}
                  </select>
                  <Button onClick={addCustomHabit}>Add</Button>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};