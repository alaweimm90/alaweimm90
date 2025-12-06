// src/components/intake/steps/Step5Nutrition.tsx
import React from 'react';
import { Input } from "@/ui/atoms/Input";
import { Label } from "@/ui/atoms/Label";
import { Checkbox } from "@/ui/atoms/Checkbox";
import { TextAreaField, SelectField } from "@/components/ui/form-fields";
import { Apple, Coffee, Utensils } from 'lucide-react';

interface Step5FormData {
  dietaryPreferences?: string;
  allergies?: string;
  mealFrequency?: string;
  nutritionGoals?: string;
  [key: string]: string | undefined;
}

interface Step5NutritionProps {
  formData: Step5FormData;
  onFieldChange: (field: string, value: string) => void;
}

export const Step5Nutrition: React.FC<Step5NutritionProps> = ({
  formData,
  onFieldChange
}) => {
  const handleAllergyChange = (allergy: string, checked: boolean) => {
    const currentAllergies = formData.foodAllergies || [];
    if (checked) {
      onFieldChange('foodAllergies', [...currentAllergies, allergy]);
    } else {
      onFieldChange('foodAllergies', currentAllergies.filter((a: string) => a !== allergy));
    }
  };

  return (
    <div className="step-container w-full h-full">
      <div className="max-w-4xl mx-auto p-6 md:p-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 tracking-tight">
            Nutrition & Lifestyle
          </h1>
          <p className="text-base md:text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Help us create a nutrition plan that fits your lifestyle and preferences
          </p>
        </div>

        {/* Form Container */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm space-y-8">
          {/* Current Diet */}
          <div className="p-6 md:p-8 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Apple className="h-5 w-5 text-green-500" />
              <h3 className="text-lg font-semibold text-gray-900">Current Diet & Eating Habits</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SelectField
                id="currentDiet"
                label="Current Diet Type"
                value={formData.currentDiet || ''}
                onChange={(value) => onFieldChange('currentDiet', value)}
                placeholder="Select diet type"
                required
                options={[
                  { value: "standard", label: "Standard/Mixed diet" },
                  { value: "vegetarian", label: "Vegetarian" },
                  { value: "vegan", label: "Vegan" },
                  { value: "pescatarian", label: "Pescatarian" },
                  { value: "keto", label: "Ketogenic" },
                  { value: "paleo", label: "Paleo" },
                  { value: "mediterranean", label: "Mediterranean" },
                  { value: "other", label: "Other" }
                ]}
              />

              <SelectField
                id="mealsPerDay"
                label="How many meals do you typically eat per day?"
                value={formData.mealsPerDay || ''}
                onChange={(value) => onFieldChange('mealsPerDay', value)}
                placeholder="Select meals per day"
                required
                options={[
                  { value: "2", label: "2 meals" },
                  { value: "3", label: "3 meals" },
                  { value: "4-5", label: "4-5 meals" },
                  { value: "6-plus", label: "6+ meals" }
                ]}
              />
            </div>

            <TextAreaField
              id="typicalDay"
              label="Describe a typical day of eating"
              value={formData.typicalDay || ''}
              onChange={(value) => onFieldChange('typicalDay', value)}
              placeholder="What do you typically eat for breakfast, lunch, dinner, and snacks?"
              description="Include meal timing, portion sizes, and typical foods you eat"
              rows={5}
            />
          </div>

          {/* Food Preferences & Restrictions */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Utensils className="h-5 w-5 text-repz-orange" />
              <h3 className="text-lg font-semibold text-gray-900">Food Preferences & Restrictions</h3>
            </div>
            
            <div className="space-y-6">
              <div>
                <Label className="text-sm font-medium text-gray-700 mb-4 block">
                  Food allergies or intolerances (Check all that apply)
                </Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    'Dairy/Lactose', 'Gluten', 'Nuts', 'Shellfish',
                    'Eggs', 'Soy', 'Fish', 'None'
                  ].map((allergy) => (
                    <div key={allergy} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors">
                      <Checkbox
                        id={allergy}
                        checked={(formData.foodAllergies || []).includes(allergy)}
                        onCheckedChange={(checked) => handleAllergyChange(allergy, checked as boolean)}
                        className="h-5 w-5 data-[state=checked]:bg-repz-orange data-[state=checked]:border-repz-orange"
                      />
                      <Label htmlFor={allergy} className="text-sm font-medium text-gray-700 cursor-pointer">
                        {allergy}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <TextAreaField
                id="foodDislikes"
                label="Foods you dislike or prefer to avoid"
                value={formData.foodDislikes || ''}
                onChange={(value) => onFieldChange('foodDislikes', value)}
                placeholder="List any foods you don't like or want to avoid in your meal plan..."
                rows={3}
              />

              <TextAreaField
                id="favoriteFoods"
                label="Favorite foods you'd like to include"
                value={formData.favoriteFoods || ''}
                onChange={(value) => onFieldChange('favoriteFoods', value)}
                placeholder="List foods you love and would like to see in your meal plan..."
                rows={3}
              />
            </div>
          </div>

          {/* Lifestyle Factors */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Coffee className="h-5 w-5 text-brown-500" />
              <h3 className="text-lg font-semibold text-gray-900">Lifestyle Factors</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SelectField
                id="cookingExperience"
                label="Cooking Experience Level"
                value={formData.cookingExperience || ''}
                onChange={(value) => onFieldChange('cookingExperience', value)}
                placeholder="Select cooking level"
                required
                options={[
                  { value: "beginner", label: "Beginner (basic meals only)" },
                  { value: "intermediate", label: "Intermediate (can follow recipes)" },
                  { value: "advanced", label: "Advanced (comfortable with complex recipes)" },
                  { value: "expert", label: "Expert (love to cook and experiment)" }
                ]}
              />

              <SelectField
                id="mealPrepTime"
                label="Time available for meal prep"
                value={formData.mealPrepTime || ''}
                onChange={(value) => onFieldChange('mealPrepTime', value)}
                placeholder="Select prep time"
                required
                options={[
                  { value: "minimal", label: "Minimal (15-30 min/day)" },
                  { value: "moderate", label: "Moderate (30-60 min/day)" },
                  { value: "generous", label: "Generous (1-2 hours/day)" },
                  { value: "weekend-batch", label: "Weekend batch cooking" }
                ]}
              />

              <SelectField
                id="budget"
                label="Weekly grocery budget"
                value={formData.budget || ''}
                onChange={(value) => onFieldChange('budget', value)}
                placeholder="Select budget range"
                options={[
                  { value: "under-100", label: "Under $100/week" },
                  { value: "100-150", label: "$100-150/week" },
                  { value: "150-200", label: "$150-200/week" },
                  { value: "over-200", label: "Over $200/week" }
                ]}
              />

              <div className="space-y-3">
                <Label htmlFor="supplementUse" className="text-sm font-medium text-gray-700">
                  Current supplement use
                </Label>
                <Input
                  id="supplementUse"
                  type="text"
                  placeholder="List any supplements you currently take"
                  value={formData.supplementUse || ''}
                  onChange={(e) => onFieldChange('supplementUse', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
                <p className="text-sm text-gray-500">
                  Include protein powder, vitamins, creatine, etc.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};