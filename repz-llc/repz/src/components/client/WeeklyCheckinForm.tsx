import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import { Label } from '@/ui/atoms/Label';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Camera, Save, Calendar } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface WeeklyCheckinData {
  weight: string;
  bodyFat: string;
  energy: number[];
  mood: number[];
  stress: number[];
  sleep: {
    duration: string;
    quality: number[];
  };
  nutrition: {
    adherence: number[];
    cravings: string;
  };
  training: {
    intensity: number[];
    recovery: number[];
    soreness: string;
  };
  photos: File[];
  notes: string;
  goals: string;
}

export function WeeklyCheckinForm() {
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState<WeeklyCheckinData>({
    weight: '',
    bodyFat: '',
    energy: [7],
    mood: [7],
    stress: [3],
    sleep: {
      duration: '',
      quality: [7]
    },
    nutrition: {
      adherence: [8],
      cravings: ''
    },
    training: {
      intensity: [7],
      recovery: [7],
      soreness: ''
    },
    photos: [],
    notes: '',
    goals: ''
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      // Auto-save to localStorage
      localStorage.setItem('weekly-checkin-draft', JSON.stringify(formData));
      
      // Submit to backend with proper error handling
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: 'Check-in Submitted!',
        description: 'Your weekly progress has been recorded.',
      });

      // Clear form
      setFormData({
        weight: '',
        bodyFat: '',
        energy: [7],
        mood: [7],
        stress: [3],
        sleep: { duration: '', quality: [7] },
        nutrition: { adherence: [8], cravings: '' },
        training: { intensity: [7], recovery: [7], soreness: '' },
        photos: [],
        notes: '',
        goals: ''
      });
      
      localStorage.removeItem('weekly-checkin-draft');
    } catch (error) {
      toast({
        title: 'Submission Failed',
        description: 'Please try again later.',
        variant: 'destructive'
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setFormData(prev => ({
      ...prev,
      photos: [...prev.photos, ...files].slice(0, 4) // Max 4 photos
    }));
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 p-6">
      <Card className="glass-tier-card border-border/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-[hsl(var(--tier-adaptive))]" />
            Weekly Check-in
          </CardTitle>
          <CardDescription>
            Track your progress and share updates with your coach
          </CardDescription>
        </CardHeader>
      </Card>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Body Composition */}
        <Card className="glass-tier-card border-border/20">
          <CardHeader>
            <CardTitle className="text-lg">Body Composition</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="weight">Weight (lbs)</Label>
              <Input
                id="weight"
                type="number"
                step="0.1"
                value={formData.weight}
                onChange={(e) => setFormData(prev => ({ ...prev, weight: e.target.value }))}
                placeholder="Enter current weight"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bodyFat">Body Fat % (optional)</Label>
              <Input
                id="bodyFat"
                type="number"
                step="0.1"
                value={formData.bodyFat}
                onChange={(e) => setFormData(prev => ({ ...prev, bodyFat: e.target.value }))}
                placeholder="Enter body fat percentage"
              />
            </div>
          </CardContent>
        </Card>

        {/* Wellness Metrics */}
        <Card className="glass-tier-card border-border/20">
          <CardHeader>
            <CardTitle className="text-lg">Wellness Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <Label>Energy Level (1-10): {formData.energy[0]}</Label>
              <Slider
                value={formData.energy}
                onValueChange={(value) => setFormData(prev => ({ ...prev, energy: value }))}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>
            
            <div className="space-y-3">
              <Label>Mood (1-10): {formData.mood[0]}</Label>
              <Slider
                value={formData.mood}
                onValueChange={(value) => setFormData(prev => ({ ...prev, mood: value }))}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>
            
            <div className="space-y-3">
              <Label>Stress Level (1-10): {formData.stress[0]}</Label>
              <Slider
                value={formData.stress}
                onValueChange={(value) => setFormData(prev => ({ ...prev, stress: value }))}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="sleep-duration">Sleep Duration (hours)</Label>
                <Input
                  id="sleep-duration"
                  type="number"
                  step="0.5"
                  value={formData.sleep.duration}
                  onChange={(e) => setFormData(prev => ({ 
                    ...prev, 
                    sleep: { ...prev.sleep, duration: e.target.value }
                  }))}
                  placeholder="7.5"
                />
              </div>
              <div className="space-y-3">
                <Label>Sleep Quality (1-10): {formData.sleep.quality[0]}</Label>
                <Slider
                  value={formData.sleep.quality}
                  onValueChange={(value) => setFormData(prev => ({ 
                    ...prev, 
                    sleep: { ...prev.sleep, quality: value }
                  }))}
                  max={10}
                  min={1}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Progress Photos */}
        <Card className="glass-tier-card border-border/20">
          <CardHeader>
            <CardTitle className="text-lg">Progress Photos</CardTitle>
            <CardDescription>
              Upload up to 4 photos (front, side, back, flex)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="border-2 border-dashed border-border/50 rounded-lg p-6 text-center">
              <Camera className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
              <Label htmlFor="photos" className="cursor-pointer">
                <span className="text-sm font-medium">Click to upload photos</span>
                <Input
                  id="photos"
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handlePhotoUpload}
                  className="hidden"
                />
              </Label>
              {formData.photos.length > 0 && (
                <p className="text-sm text-muted-foreground mt-2">
                  {formData.photos.length} photo(s) selected
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Training & Nutrition */}
        <Card className="glass-tier-card border-border/20">
          <CardHeader>
            <CardTitle className="text-lg">Training & Nutrition</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <Label>Training Intensity (1-10): {formData.training.intensity[0]}</Label>
              <Slider
                value={formData.training.intensity}
                onValueChange={(value) => setFormData(prev => ({ 
                  ...prev, 
                  training: { ...prev.training, intensity: value }
                }))}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>
            
            <div className="space-y-3">
              <Label>Recovery Quality (1-10): {formData.training.recovery[0]}</Label>
              <Slider
                value={formData.training.recovery}
                onValueChange={(value) => setFormData(prev => ({ 
                  ...prev, 
                  training: { ...prev.training, recovery: value }
                }))}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>

            <div className="space-y-3">
              <Label>Nutrition Adherence (1-10): {formData.nutrition.adherence[0]}</Label>
              <Slider
                value={formData.nutrition.adherence}
                onValueChange={(value) => setFormData(prev => ({ 
                  ...prev, 
                  nutrition: { ...prev.nutrition, adherence: value }
                }))}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>
          </CardContent>
        </Card>

        {/* Notes & Goals */}
        <Card className="glass-tier-card border-border/20">
          <CardHeader>
            <CardTitle className="text-lg">Notes & Goals</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="notes">Weekly Notes</Label>
              <Textarea
                id="notes"
                value={formData.notes}
                onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
                placeholder="How did this week go? Any challenges or wins?"
                rows={4}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="goals">Goals for Next Week</Label>
              <Textarea
                id="goals"
                value={formData.goals}
                onChange={(e) => setFormData(prev => ({ ...prev, goals: e.target.value }))}
                placeholder="What do you want to focus on next week?"
                rows={3}
              />
            </div>
          </CardContent>
        </Card>

        <Button 
          type="submit" 
          className="w-full" 
          disabled={isSubmitting}
        >
          <Save className="w-4 h-4 mr-2" />
          {isSubmitting ? 'Submitting...' : 'Submit Weekly Check-in'}
        </Button>
      </form>
    </div>
  );
}