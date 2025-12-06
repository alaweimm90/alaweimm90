import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Input } from '@/ui/atoms/Input';
import { Label } from '@/ui/atoms/Label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { useToast } from '@/hooks/use-toast';
import { 
  Dumbbell, 
  Plus, 
  FileText, 
  Copy, 
  Edit, 
  Trash2, 
  Clock, 
  Target, 
  Activity,
  Save,
  Eye
} from 'lucide-react';

interface Exercise {
  id: string;
  name: string;
  sets: number;
  reps: string;
  weight: string;
  restTime: number;
  notes: string;
  muscleGroups: string[];
}

interface WorkoutPlan {
  id: string;
  name: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: number; // in minutes
  exercises: Exercise[];
  targetMuscleGroups: string[];
  equipment: string[];
  createdAt: string;
  isTemplate: boolean;
}

const muscleGroups = [
  'Chest', 'Back', 'Shoulders', 'Arms', 'Legs', 'Core', 'Glutes', 'Calves'
];

const equipmentOptions = [
  'Dumbbells', 'Barbell', 'Resistance Bands', 'Bodyweight', 'Machines', 'Kettlebells', 'Cable'
];

export const WorkoutPlanCreator: React.FC = () => {
  const [workoutPlans, setWorkoutPlans] = useState<WorkoutPlan[]>([]);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [editingPlan, setEditingPlan] = useState<WorkoutPlan | null>(null);
  const [currentExercise, setCurrentExercise] = useState<Partial<Exercise>>({});
  const { toast } = useToast();

  // Mock data for demonstration
  useEffect(() => {
    const mockPlans: WorkoutPlan[] = [
      {
        id: '1',
        name: 'Upper Body Strength - Week 1',
        description: 'Focus on building upper body strength with compound movements',
        difficulty: 'intermediate',
        duration: 60,
        exercises: [
          {
            id: '1',
            name: 'Bench Press',
            sets: 4,
            reps: '8-10',
            weight: '80%',
            restTime: 120,
            notes: 'Focus on controlled movement',
            muscleGroups: ['Chest', 'Arms']
          },
          {
            id: '2',
            name: 'Pull-ups',
            sets: 3,
            reps: '6-8',
            weight: 'Bodyweight',
            restTime: 90,
            notes: 'Use assistance if needed',
            muscleGroups: ['Back', 'Arms']
          }
        ],
        targetMuscleGroups: ['Chest', 'Back', 'Arms'],
        equipment: ['Barbell', 'Bodyweight'],
        createdAt: '2024-01-15',
        isTemplate: false
      },
      {
        id: '2',
        name: 'Full Body Circuit Template',
        description: 'High-intensity full body circuit for conditioning',
        difficulty: 'beginner',
        duration: 45,
        exercises: [
          {
            id: '3',
            name: 'Squats',
            sets: 3,
            reps: '15',
            weight: 'Bodyweight',
            restTime: 60,
            notes: 'Keep chest up, knees tracking over toes',
            muscleGroups: ['Legs', 'Glutes']
          }
        ],
        targetMuscleGroups: ['Legs', 'Core', 'Arms'],
        equipment: ['Bodyweight', 'Dumbbells'],
        createdAt: '2024-01-10',
        isTemplate: true
      }
    ];
    setWorkoutPlans(mockPlans);
  }, []);

  const handleCreatePlan = () => {
    setEditingPlan({
      id: '',
      name: '',
      description: '',
      difficulty: 'beginner',
      duration: 45,
      exercises: [],
      targetMuscleGroups: [],
      equipment: [],
      createdAt: new Date().toISOString().split('T')[0],
      isTemplate: false
    });
    setIsCreateModalOpen(true);
  };

  const handleSavePlan = () => {
    if (!editingPlan) return;

    const newPlan = {
      ...editingPlan,
      id: editingPlan.id || Date.now().toString()
    };

    if (editingPlan.id) {
      setWorkoutPlans(plans => plans.map(p => p.id === editingPlan.id ? newPlan : p));
      toast({ title: "Plan updated successfully" });
    } else {
      setWorkoutPlans(plans => [...plans, newPlan]);
      toast({ title: "Plan created successfully" });
    }

    setIsCreateModalOpen(false);
    setEditingPlan(null);
  };

  const handleAddExercise = () => {
    if (!editingPlan || !currentExercise.name) return;

    const newExercise: Exercise = {
      id: Date.now().toString(),
      name: currentExercise.name || '',
      sets: currentExercise.sets || 3,
      reps: currentExercise.reps || '10',
      weight: currentExercise.weight || 'Bodyweight',
      restTime: currentExercise.restTime || 60,
      notes: currentExercise.notes || '',
      muscleGroups: currentExercise.muscleGroups || []
    };

    setEditingPlan({
      ...editingPlan,
      exercises: [...editingPlan.exercises, newExercise]
    });

    setCurrentExercise({});
    toast({ title: "Exercise added" });
  };

  const handleDeletePlan = (planId: string) => {
    setWorkoutPlans(plans => plans.filter(p => p.id !== planId));
    toast({ title: "Plan deleted" });
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-700';
      case 'intermediate': return 'bg-yellow-100 text-yellow-700';
      case 'advanced': return 'bg-red-100 text-red-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center">
        <div>
          <h2 className="text-2xl font-bold">Workout Plan Creator</h2>
          <p className="text-muted-foreground">Create and manage custom workout plans for your clients</p>
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => console.log("WorkoutPlanCreator button clicked")}>
            <FileText className="h-4 w-4 mr-2" />
            Templates
          </Button>
          <Button onClick={handleCreatePlan}>
            <Plus className="h-4 w-4 mr-2" />
            Create Plan
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{workoutPlans.length}</p>
              <p className="text-sm text-muted-foreground">Total Plans</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">
                {workoutPlans.filter(p => p.isTemplate).length}
              </p>
              <p className="text-sm text-muted-foreground">Templates</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-orange-600">
                {workoutPlans.filter(p => p.difficulty === 'advanced').length}
              </p>
              <p className="text-sm text-muted-foreground">Advanced</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">
                {workoutPlans.reduce((acc, plan) => acc + plan.exercises.length, 0)}
              </p>
              <p className="text-sm text-muted-foreground">Total Exercises</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Workout Plans List */}
      <div className="grid gap-4">
        {workoutPlans.map((plan) => (
          <Card key={plan.id} className="hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-lg font-semibold">{plan.name}</h3>
                    <Badge className={getDifficultyColor(plan.difficulty)}>
                      {plan.difficulty}
                    </Badge>
                    {plan.isTemplate && (
                      <Badge variant="outline">Template</Badge>
                    )}
                  </div>
                  
                  <p className="text-muted-foreground mb-3">{plan.description}</p>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Duration</p>
                      <p className="font-medium flex items-center">
                        <Clock className="h-4 w-4 mr-1" />
                        {plan.duration} min
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Exercises</p>
                      <p className="font-medium flex items-center">
                        <Activity className="h-4 w-4 mr-1" />
                        {plan.exercises.length}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Target</p>
                      <p className="font-medium">
                        {plan.targetMuscleGroups.slice(0, 2).join(', ')}
                        {plan.targetMuscleGroups.length > 2 && '...'}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Equipment</p>
                      <p className="font-medium">
                        {plan.equipment.slice(0, 2).join(', ')}
                        {plan.equipment.length > 2 && '...'}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" onClick={() => console.log("WorkoutPlanCreator button clicked")}>
                    <Eye className="h-4 w-4 mr-2" />
                    Preview
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      setEditingPlan(plan);
                      setIsCreateModalOpen(true);
                    }}
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    Edit
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => console.log("WorkoutPlanCreator button clicked")}>
                    <Copy className="h-4 w-4 mr-2" />
                    Copy
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleDeletePlan(plan.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Create/Edit Modal */}
      <Dialog open={isCreateModalOpen} onOpenChange={setIsCreateModalOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingPlan?.id ? 'Edit Workout Plan' : 'Create New Workout Plan'}
            </DialogTitle>
          </DialogHeader>
          
          {editingPlan && (
            <Tabs defaultValue="details" className="space-y-4">
              <TabsList>
                <TabsTrigger value="details">Plan Details</TabsTrigger>
                <TabsTrigger value="exercises">Exercises</TabsTrigger>
              </TabsList>
              
              <TabsContent value="details" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="planName">Plan Name</Label>
                    <Input 
                      id="planName"
                      value={editingPlan.name}
                      onChange={(e) => setEditingPlan({...editingPlan, name: e.target.value})}
                      placeholder="Enter plan name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="difficulty">Difficulty</Label>
                    <Select 
                      value={editingPlan.difficulty}
                      onValueChange={(value: 'beginner' | 'intermediate' | 'advanced') => 
                        setEditingPlan({...editingPlan, difficulty: value})
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="beginner">Beginner</SelectItem>
                        <SelectItem value="intermediate">Intermediate</SelectItem>
                        <SelectItem value="advanced">Advanced</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div>
                  <Label htmlFor="description">Description</Label>
                  <Textarea 
                    id="description"
                    value={editingPlan.description}
                    onChange={(e) => setEditingPlan({...editingPlan, description: e.target.value})}
                    placeholder="Describe the workout plan..."
                  />
                </div>
                
                <div>
                  <Label htmlFor="duration">Duration (minutes)</Label>
                  <Input 
                    id="duration"
                    type="number"
                    value={editingPlan.duration}
                    onChange={(e) => setEditingPlan({...editingPlan, duration: parseInt(e.target.value)})}
                  />
                </div>
              </TabsContent>
              
              <TabsContent value="exercises" className="space-y-4">
                <div className="border rounded-lg p-4 space-y-4">
                  <h4 className="font-semibold">Add Exercise</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Exercise Name</Label>
                      <Input 
                        value={currentExercise.name || ''}
                        onChange={(e) => setCurrentExercise({...currentExercise, name: e.target.value})}
                        placeholder="e.g., Bench Press"
                      />
                    </div>
                    <div>
                      <Label>Sets</Label>
                      <Input 
                        type="number"
                        value={currentExercise.sets || ''}
                        onChange={(e) => setCurrentExercise({...currentExercise, sets: parseInt(e.target.value)})}
                        placeholder="3"
                      />
                    </div>
                    <div>
                      <Label>Reps</Label>
                      <Input 
                        value={currentExercise.reps || ''}
                        onChange={(e) => setCurrentExercise({...currentExercise, reps: e.target.value})}
                        placeholder="8-10"
                      />
                    </div>
                    <div>
                      <Label>Weight/Load</Label>
                      <Input 
                        value={currentExercise.weight || ''}
                        onChange={(e) => setCurrentExercise({...currentExercise, weight: e.target.value})}
                        placeholder="80% or 135lbs"
                      />
                    </div>
                  </div>
                  <Button onClick={handleAddExercise} className="w-full">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Exercise
                  </Button>
                </div>
                
                {editingPlan.exercises.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="font-semibold">Exercises ({editingPlan.exercises.length})</h4>
                    {editingPlan.exercises.map((exercise, index) => (
                      <div key={exercise.id} className="border rounded p-3 flex justify-between items-center">
                        <div>
                          <p className="font-medium">{exercise.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {exercise.sets} sets × {exercise.reps} reps @ {exercise.weight}
                          </p>
                        </div>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => {
                            const newExercises = editingPlan.exercises.filter(e => e.id !== exercise.id);
                            setEditingPlan({...editingPlan, exercises: newExercises});
                          }}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          )}
          
          <div className="flex gap-2 pt-4">
            <Button onClick={handleSavePlan} className="flex-1">
              <Save className="h-4 w-4 mr-2" />
              Save Plan
            </Button>
            <Button variant="outline" onClick={() => setIsCreateModalOpen(false)}>
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};