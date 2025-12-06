import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Input } from '@/ui/atoms/Input';
import { Label } from '@/ui/atoms/Label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Slider } from '@/components/ui/slider';
import { Plus, Save, Clock, CheckCircle, Dumbbell, Target } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { LoadingButton } from '@/components/ui/loading-button';
import { SessionType } from '@/types/fitness';

interface WorkoutLogData {
  workout_date: string;
  session_type: SessionType;
  duration_minutes?: number;
  perceived_exertion: number;
  energy_pre_workout: number;
  energy_post_workout: number;
  pump_quality: number;
  focus_level: number;
  weight_progression: boolean;
  rep_progression: boolean;
  technique_rating: number;
  notes: string;
}

interface Exercise {
  name: string;
  sets: number;
  reps: string;
  weight: string;
  notes?: string;
}

interface WorkoutLoggerProps {
  clientId: string;
  onComplete?: () => void;
}

const WorkoutLogger: React.FC<WorkoutLoggerProps> = ({ clientId, onComplete }) => {
  const [formData, setFormData] = useState<WorkoutLogData>({
    workout_date: new Date().toISOString().split('T')[0],
    session_type: 'upper',
    perceived_exertion: 5,
    energy_pre_workout: 5,
    energy_post_workout: 5,
    pump_quality: 5,
    focus_level: 5,
    weight_progression: false,
    rep_progression: false,
    technique_rating: 5,
    notes: ''
  });

  const [exercises, setExercises] = useState<Exercise[]>([
    { name: '', sets: 3, reps: '', weight: '', notes: '' }
  ]);

  const [isSubmitting, setIsSubmitting] = useState(false);

  const sessionTypes: { value: SessionType; label: string }[] = [
    { value: 'upper', label: 'Upper Body' },
    { value: 'lower', label: 'Lower Body' },
    { value: 'push', label: 'Push' },
    { value: 'pull', label: 'Pull' },
    { value: 'legs', label: 'Legs' },
    { value: 'cardio', label: 'Cardio' },
    { value: 'rest', label: 'Active Recovery' }
  ];

  const addExercise = () => {
    setExercises(prev => [...prev, { name: '', sets: 3, reps: '', weight: '', notes: '' }]);
  };

  const updateExercise = (index: number, field: keyof Exercise, value: string | number) => {
    setExercises(prev => prev.map((exercise, i) => 
      i === index ? { ...exercise, [field]: value } : exercise
    ));
  };

  const removeExercise = (index: number) => {
    setExercises(prev => prev.filter((_, i) => i !== index));
  };

  const handleSliderChange = (field: keyof WorkoutLogData, value: number[]) => {
    setFormData(prev => ({ ...prev, [field]: value[0] }));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    
    try {
      // Validate required fields
      if (!formData.session_type) {
        throw new Error('Please select a session type');
      }

      const workoutData = {
        client_id: clientId,
        ...formData,
        // Store exercises as JSON in notes field for now
        // In a full implementation, you'd have a separate exercises table
        notes: `${formData.notes}\n\nExercises:\n${exercises
          .filter(ex => ex.name.trim())
          .map(ex => `${ex.name}: ${ex.sets} sets x ${ex.reps} reps @ ${ex.weight}${ex.notes ? ` (${ex.notes})` : ''}`)
          .join('\n')}`
      };

      const { error } = await supabase
        .from('workout_logs')
        .insert(workoutData);

      if (error) throw error;

      toast({
        title: "Workout Logged!",
        description: "Your workout has been saved and sent for coach review.",
      });

      // Reset form
      setFormData({
        workout_date: new Date().toISOString().split('T')[0],
        session_type: 'upper',
        perceived_exertion: 5,
        energy_pre_workout: 5,
        energy_post_workout: 5,
        pump_quality: 5,
        focus_level: 5,
        weight_progression: false,
        rep_progression: false,
        technique_rating: 5,
        notes: ''
      });
      setExercises([{ name: '', sets: 3, reps: '', weight: '', notes: '' }]);

      if (onComplete) onComplete();
    } catch (error) {
      console.error('Error logging workout:', error);
      toast({
        title: "Failed to Log Workout",
        description: error instanceof Error ? error.message : "Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const SliderField = ({ 
    label, 
    field, 
    description, 
    lowLabel = "Low", 
    highLabel = "High" 
  }: {
    label: string;
    field: keyof WorkoutLogData;
    description?: string;
    lowLabel?: string;
    highLabel?: string;
  }) => (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">{label}</Label>
        <Badge variant="secondary">{formData[field]}/10</Badge>
      </div>
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
      <Slider
        value={[formData[field] as number]}
        onValueChange={(value) => handleSliderChange(field, value)}
        max={10}
        min={1}
        step={1}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
      </div>
    </div>
  );

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Dumbbell className="h-5 w-5" />
          <CardTitle>Log Workout</CardTitle>
        </div>
        <CardDescription>
          Track your training session for coach review and form feedback
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-8">
        {/* Basic Workout Info */}
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="date">Workout Date</Label>
            <Input
              id="date"
              type="date"
              value={formData.workout_date}
              onChange={(e) => setFormData(prev => ({ ...prev, workout_date: e.target.value }))}
            />
          </div>

          <div className="space-y-2">
            <Label>Session Type</Label>
            <Select 
              value={formData.session_type} 
              onValueChange={(value: SessionType) => setFormData(prev => ({ ...prev, session_type: value }))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select session type" />
              </SelectTrigger>
              <SelectContent>
                {sessionTypes.map(type => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="duration">Duration (minutes)</Label>
            <Input
              id="duration"
              type="number"
              min="1"
              max="300"
              value={formData.duration_minutes || ''}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                duration_minutes: parseInt(e.target.value) || undefined 
              }))}
              placeholder="90"
            />
          </div>
        </div>

        {/* Exercises */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Target className="h-5 w-5" />
              Exercises
            </h3>
            <Button onClick={addExercise} variant="outline" size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Exercise
            </Button>
          </div>

          <div className="space-y-4">
            {exercises.map((exercise, index) => (
              <Card key={index} className="p-4">
                <div className="grid gap-4 md:grid-cols-5">
                  <div className="md:col-span-2">
                    <Label>Exercise Name</Label>
                    <Input
                      value={exercise.name}
                      onChange={(e) => updateExercise(index, 'name', e.target.value)}
                      placeholder="e.g., Bench Press"
                    />
                  </div>
                  <div>
                    <Label>Sets</Label>
                    <Input
                      type="number"
                      min="1"
                      max="20"
                      value={exercise.sets}
                      onChange={(e) => updateExercise(index, 'sets', parseInt(e.target.value) || 1)}
                    />
                  </div>
                  <div>
                    <Label>Reps</Label>
                    <Input
                      value={exercise.reps}
                      onChange={(e) => updateExercise(index, 'reps', e.target.value)}
                      placeholder="8-12"
                    />
                  </div>
                  <div>
                    <Label>Weight</Label>
                    <Input
                      value={exercise.weight}
                      onChange={(e) => updateExercise(index, 'weight', e.target.value)}
                      placeholder="100kg"
                    />
                  </div>
                </div>
                
                <div className="mt-3 flex gap-2">
                  <div className="flex-1">
                    <Label>Exercise Notes</Label>
                    <Input
                      value={exercise.notes || ''}
                      onChange={(e) => updateExercise(index, 'notes', e.target.value)}
                      placeholder="Form notes, RPE, etc."
                    />
                  </div>
                  {exercises.length > 1 && (
                    <Button 
                      onClick={() => removeExercise(index)}
                      variant="destructive"
                      size="sm"
                      className="mt-6"
                    >
                      Remove
                    </Button>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Subjective Ratings */}
        <div className="space-y-6">
          <h3 className="text-lg font-semibold">How did the workout feel?</h3>
          
          <div className="grid gap-6 md:grid-cols-2">
            <SliderField 
              label="Perceived Exertion (RPE)" 
              field="perceived_exertion"
              description="Overall difficulty (1 = very easy, 10 = maximum effort)"
              lowLabel="Very Easy"
              highLabel="Max Effort"
            />
            
            <SliderField 
              label="Pre-Workout Energy" 
              field="energy_pre_workout"
              description="Energy level before starting"
              lowLabel="Exhausted"
              highLabel="Energetic"
            />
            
            <SliderField 
              label="Post-Workout Energy" 
              field="energy_post_workout"
              description="Energy level after workout"
              lowLabel="Drained"
              highLabel="Energized"
            />
            
            <SliderField 
              label="Pump Quality" 
              field="pump_quality"
              description="Muscle pump and blood flow"
              lowLabel="No Pump"
              highLabel="Amazing Pump"
            />
            
            <SliderField 
              label="Focus Level" 
              field="focus_level"
              description="Mental focus during training"
              lowLabel="Distracted"
              highLabel="Laser Focused"
            />
            
            <SliderField 
              label="Technique Rating" 
              field="technique_rating"
              description="Form quality throughout workout"
              lowLabel="Poor Form"
              highLabel="Perfect Form"
            />
          </div>
        </div>

        {/* Progression */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Progression</h3>
          
          <div className="grid gap-4 md:grid-cols-2">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="weight-progression"
                checked={formData.weight_progression}
                onChange={(e) => setFormData(prev => ({ ...prev, weight_progression: e.target.checked }))}
                className="rounded"
              />
              <Label htmlFor="weight-progression">Weight Progression (increased weight this session)</Label>
            </div>
            
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="rep-progression"
                checked={formData.rep_progression}
                onChange={(e) => setFormData(prev => ({ ...prev, rep_progression: e.target.checked }))}
                className="rounded"
              />
              <Label htmlFor="rep-progression">Rep Progression (increased reps this session)</Label>
            </div>
          </div>
        </div>

        {/* Additional Notes */}
        <div className="space-y-2">
          <Label htmlFor="notes">Additional Notes</Label>
          <Textarea
            id="notes"
            placeholder="Any additional thoughts about this workout, how you felt, challenges, etc."
            value={formData.notes}
            onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
            className="min-h-[100px]"
          />
        </div>

        {/* Submit Button */}
        <div className="flex justify-end pt-6 border-t">
          <LoadingButton
            onClick={handleSubmit}
            loading={isSubmitting}
            className="min-w-[150px]"
          >
            <Save className="h-4 w-4 mr-2" />
            Log Workout
          </LoadingButton>
        </div>
      </CardContent>
    </Card>
  );
};

export default WorkoutLogger;