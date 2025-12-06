import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import { Label } from '@/ui/atoms/Label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Apple, Plus, Target, Utensils } from 'lucide-react';
import { TierGate } from '@/components/auth/TierGate';
import { useTierAccess } from '@/hooks/useTierAccess';

const mockNutritionData = {
  calories: { consumed: 1850, target: 2200 },
  protein: { consumed: 145, target: 165 },
  carbs: { consumed: 180, target: 220 },
  fats: { consumed: 65, target: 80 }
};

export function NutritionTracking() {
  const { hasAutoGroceryLists } = useTierAccess();
  const [newFood, setNewFood] = useState('');

  if (!hasAutoGroceryLists) {
    return (
      <TierGate 
        requiredTier="adaptive" 
        feature="Nutrition Tracking"
      >
        <div className="p-6">Nutrition content would be here</div>
      </TierGate>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Apple className="w-5 h-5 text-tier-adaptive" />
            Daily Nutrition
          </CardTitle>
          <CardDescription>
            Track your macronutrients and reach your goals
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(mockNutritionData).map(([key, data]) => (
              <div key={key} className="space-y-2">
                <Label className="text-sm font-medium capitalize">{key}</Label>
                <Progress 
                  value={(data.consumed / data.target) * 100} 
                  className="h-2"
                />
                <div className="text-xs text-muted-foreground">
                  {data.consumed} / {data.target}
                  {key === 'calories' ? ' kcal' : 'g'}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5 text-tier-performance" />
            Add Food
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Search foods..."
              value={newFood}
              onChange={(e) => setNewFood(e.target.value)}
              className="flex-1"
            />
            <Button onClick={() => console.log("NutritionTracking button clicked")}>
              <Plus className="w-4 h-4 mr-2" />
              Add
            </Button>
          </div>
          
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              <Utensils className="w-3 h-3 mr-1" />
              Chicken Breast - 200g
            </Badge>
            <Badge variant="secondary">
              <Utensils className="w-3 h-3 mr-1" />
              Brown Rice - 150g
            </Badge>
            <Badge variant="secondary">
              <Utensils className="w-3 h-3 mr-1" />
              Mixed Vegetables - 100g
            </Badge>
          </div>
        </CardContent>
      </Card>

      <Card className="glass-tier-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-tier-longevity" />
            Smart Grocery List
          </CardTitle>
          <CardDescription>
            AI-generated shopping list based on your goals
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[
              'Lean Ground Turkey - 2 lbs',
              'Sweet Potatoes - 3 lbs',
              'Spinach - 2 bags',
              'Greek Yogurt - 32oz',
              'Almonds - 1 bag'
            ].map((item, index) => (
              <div key={index} className="flex items-center gap-2 p-2 rounded-lg bg-background/50">
                <input type="checkbox" className="rounded" />
                <span className="text-sm">{item}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}