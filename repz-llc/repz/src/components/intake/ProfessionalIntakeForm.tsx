/**
 * REPZ Professional Client Intake Form
 *
 * A comprehensive, industry-standard client intake form optimized for digital delivery.
 * Incorporates best practices from elite coaching platforms including:
 * - Trainerize, TrueCoach, PT Distinction, Future
 * - NSCA and ACSM assessment guidelines
 * - PAR-Q+ health screening standards
 *
 * @version 2.0.0
 * @author REPZ Development Team
 */

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  User, Heart, Target, Calendar,
  AlertTriangle, CheckCircle, ChevronRight,
  ChevronLeft, Download, Send, Shield, Dumbbell,
  Scale, Brain, Utensils
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/ui/atoms/Input';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/ui/atoms/Checkbox';
import { Label } from '@/ui/atoms/Label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
// RadioGroup imported but not used - keeping for future form enhancements
// import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { toast } from 'sonner';

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

const personalInfoSchema = z.object({
  firstName: z.string().min(2, 'First name is required'),
  lastName: z.string().min(2, 'Last name is required'),
  email: z.string().email('Valid email is required'),
  phone: z.string().min(10, 'Valid phone number is required'),
  dateOfBirth: z.string().min(1, 'Date of birth is required'),
  gender: z.enum(['male', 'female', 'non-binary', 'prefer-not-to-say']),
  address: z.string().optional(),
  city: z.string().optional(),
  state: z.string().optional(),
  zipCode: z.string().optional(),
  emergencyContactName: z.string().min(2, 'Emergency contact name is required'),
  emergencyContactPhone: z.string().min(10, 'Emergency contact phone is required'),
  emergencyContactRelation: z.string().min(2, 'Relationship is required'),
});

const fitnessAssessmentSchema = z.object({
  currentWeight: z.number().min(50, 'Weight must be at least 50 lbs'),
  goalWeight: z.number().optional(),
  height: z.string().min(1, 'Height is required'),
  bodyFatPercentage: z.number().optional(),
  restingHeartRate: z.number().optional(),
  bloodPressureSystolic: z.number().optional(),
  bloodPressureDiastolic: z.number().optional(),
  waistCircumference: z.number().optional(),
  hipCircumference: z.number().optional(),
  chestCircumference: z.number().optional(),
  armCircumference: z.number().optional(),
  thighCircumference: z.number().optional(),
});

const healthHistorySchema = z.object({
  // PAR-Q+ Questions
  heartCondition: z.boolean(),
  chestPain: z.boolean(),
  dizziness: z.boolean(),
  boneJointProblem: z.boolean(),
  bloodPressureMedication: z.boolean(),
  otherReason: z.boolean(),

  // Medical Conditions
  medicalConditions: z.array(z.string()),
  currentMedications: z.string().optional(),
  allergies: z.string().optional(),
  surgeries: z.string().optional(),

  // Injury History
  currentInjuries: z.string().optional(),
  pastInjuries: z.string().optional(),
  movementLimitations: z.string().optional(),

  // Physician Information
  physicianName: z.string().optional(),
  physicianPhone: z.string().optional(),
  lastPhysicalDate: z.string().optional(),
  physicianClearance: z.boolean(),
});

const trainingExperienceSchema = z.object({
  fitnessLevel: z.enum(['beginner', 'intermediate', 'advanced', 'elite']),
  yearsTraining: z.string(),
  currentTrainingFrequency: z.string(),
  preferredTrainingDays: z.array(z.string()),
  preferredTrainingTime: z.string(),
  sessionDuration: z.string(),

  // Training History
  previousPrograms: z.string().optional(),
  favoriteExercises: z.string().optional(),
  dislikedExercises: z.string().optional(),

  // Equipment Access
  trainingLocation: z.enum(['gym', 'home', 'both', 'outdoor']),
  gymMembership: z.string().optional(),
  homeEquipment: z.array(z.string()),

  // Strength Benchmarks
  benchPress1RM: z.number().optional(),
  squat1RM: z.number().optional(),
  deadlift1RM: z.number().optional(),
  overheadPress1RM: z.number().optional(),
  pullUps: z.number().optional(),
  pushUps: z.number().optional(),
  plankHold: z.number().optional(),
  mileTime: z.string().optional(),
});

const goalIdentificationSchema = z.object({
  primaryGoal: z.enum([
    'weight-loss', 'muscle-gain', 'strength', 'endurance',
    'athletic-performance', 'body-recomposition', 'general-fitness',
    'sport-specific', 'rehabilitation', 'longevity'
  ]),
  secondaryGoals: z.array(z.string()),
  specificTargets: z.string(),
  timeframe: z.string(),
  motivation: z.string(),
  previousAttempts: z.string().optional(),
  obstacles: z.string().optional(),
  accountability: z.enum(['high', 'medium', 'low']),
  coachingStyle: z.enum(['strict', 'balanced', 'flexible']),
});

const nutritionHabitsSchema = z.object({
  dietaryApproach: z.string(),
  mealsPerDay: z.string(),
  waterIntake: z.string(),
  alcoholConsumption: z.string(),
  caffeineIntake: z.string(),

  // Dietary Restrictions
  foodAllergies: z.array(z.string()),
  dietaryRestrictions: z.array(z.string()),
  dislikedFoods: z.string().optional(),

  // Current Habits
  typicalBreakfast: z.string().optional(),
  typicalLunch: z.string().optional(),
  typicalDinner: z.string().optional(),
  snackingHabits: z.string().optional(),
  eatingOutFrequency: z.string(),
  mealPrepWillingness: z.enum(['yes', 'sometimes', 'no']),

  // Supplements
  currentSupplements: z.string().optional(),
  supplementBudget: z.string().optional(),
  openToSupplements: z.boolean(),
});

const lifestyleFactorsSchema = z.object({
  occupation: z.string(),
  occupationActivity: z.enum(['sedentary', 'light', 'moderate', 'active', 'very-active']),
  workSchedule: z.string(),
  stressLevel: z.enum(['low', 'moderate', 'high', 'very-high']),

  // Sleep
  averageSleepHours: z.string(),
  sleepQuality: z.enum(['poor', 'fair', 'good', 'excellent']),
  sleepIssues: z.array(z.string()),

  // Recovery
  recoveryMethods: z.array(z.string()),
  restDaysPerWeek: z.string(),

  // Lifestyle
  smokingStatus: z.enum(['never', 'former', 'current']),
  recreationalDrugs: z.boolean(),
  travelFrequency: z.string(),

  // Support System
  familySupport: z.boolean(),
  workoutPartner: z.boolean(),
  previousCoaching: z.boolean(),
});

const scheduleAvailabilitySchema = z.object({
  timezone: z.string(),
  preferredStartDate: z.string(),

  // Weekly Availability
  mondayAvailability: z.array(z.string()),
  tuesdayAvailability: z.array(z.string()),
  wednesdayAvailability: z.array(z.string()),
  thursdayAvailability: z.array(z.string()),
  fridayAvailability: z.array(z.string()),
  saturdayAvailability: z.array(z.string()),
  sundayAvailability: z.array(z.string()),

  // Communication Preferences
  preferredContactMethod: z.enum(['email', 'text', 'phone', 'app']),
  checkInFrequency: z.enum(['daily', 'weekly', 'bi-weekly']),
  responseTimeExpectation: z.string(),

  // Upcoming Events
  upcomingVacations: z.string().optional(),
  upcomingEvents: z.string().optional(),
  competitionDates: z.string().optional(),
});

const consentSchema = z.object({
  informedConsent: z.boolean().refine(val => val === true, 'You must agree to the informed consent'),
  liabilityWaiver: z.boolean().refine(val => val === true, 'You must agree to the liability waiver'),
  privacyPolicy: z.boolean().refine(val => val === true, 'You must agree to the privacy policy'),
  photoRelease: z.boolean(),
  medicalDisclosure: z.boolean().refine(val => val === true, 'You must acknowledge the medical disclosure'),
  programAgreement: z.boolean().refine(val => val === true, 'You must agree to the program terms'),
  signature: z.string().min(2, 'Signature is required'),
  signatureDate: z.string(),
});

// Combined schema
const fullIntakeSchema = z.object({
  personalInfo: personalInfoSchema,
  fitnessAssessment: fitnessAssessmentSchema,
  healthHistory: healthHistorySchema,
  trainingExperience: trainingExperienceSchema,
  goalIdentification: goalIdentificationSchema,
  nutritionHabits: nutritionHabitsSchema,
  lifestyleFactors: lifestyleFactorsSchema,
  scheduleAvailability: scheduleAvailabilitySchema,
  consent: consentSchema,
});

type IntakeFormData = z.infer<typeof fullIntakeSchema>;

// ============================================================================
// FORM SECTIONS
// ============================================================================

const FORM_SECTIONS = [
  { id: 'personal', title: 'Personal Information', icon: User, description: 'Basic contact and demographic information' },
  { id: 'fitness', title: 'Fitness Assessment', icon: Scale, description: 'Current body measurements and benchmarks' },
  { id: 'health', title: 'Health History', icon: Heart, description: 'Medical history and health screening' },
  { id: 'training', title: 'Training Experience', icon: Dumbbell, description: 'Exercise background and preferences' },
  { id: 'goals', title: 'Goal Identification', icon: Target, description: 'Your fitness objectives and timeline' },
  { id: 'nutrition', title: 'Nutritional Habits', icon: Utensils, description: 'Current diet and eating patterns' },
  { id: 'lifestyle', title: 'Lifestyle Factors', icon: Brain, description: 'Work, sleep, stress, and recovery' },
  { id: 'schedule', title: 'Schedule Availability', icon: Calendar, description: 'Training times and communication' },
  { id: 'consent', title: 'Informed Consent', icon: Shield, description: 'Agreements and acknowledgments' },
];

// ============================================================================
// MEDICAL CONDITIONS LIST
// ============================================================================

const MEDICAL_CONDITIONS = [
  'Heart disease or cardiovascular issues',
  'High blood pressure (hypertension)',
  'Low blood pressure (hypotension)',
  'Diabetes (Type 1 or Type 2)',
  'Asthma or respiratory conditions',
  'Arthritis or joint problems',
  'Osteoporosis or bone density issues',
  'Thyroid disorders',
  'Kidney disease',
  'Liver disease',
  'Cancer (current or history)',
  'Autoimmune disorders',
  'Neurological conditions',
  'Mental health conditions',
  'Eating disorders (current or history)',
  'Pregnancy or postpartum',
  'None of the above',
];

// These option arrays are exported for use in form rendering
export const HOME_EQUIPMENT = [
  'Dumbbells',
  'Barbell and plates',
  'Kettlebells',
  'Resistance bands',
  'Pull-up bar',
  'Bench (flat or adjustable)',
  'Squat rack or power cage',
  'Cable machine',
  'Cardio equipment (treadmill, bike, etc.)',
  'Yoga mat',
  'Foam roller',
  'TRX or suspension trainer',
  'Medicine balls',
  'Battle ropes',
  'None',
];

export const DIETARY_RESTRICTIONS = [
  'Vegetarian',
  'Vegan',
  'Pescatarian',
  'Gluten-free',
  'Dairy-free',
  'Keto/Low-carb',
  'Paleo',
  'Halal',
  'Kosher',
  'No restrictions',
];

export const FOOD_ALLERGIES = [
  'Peanuts',
  'Tree nuts',
  'Dairy',
  'Eggs',
  'Wheat/Gluten',
  'Soy',
  'Fish',
  'Shellfish',
  'Sesame',
  'None',
];

export const SLEEP_ISSUES = [
  'Difficulty falling asleep',
  'Waking up during the night',
  'Waking up too early',
  'Sleep apnea',
  'Restless leg syndrome',
  'Insomnia',
  'Shift work sleep disorder',
  'None',
];

export const RECOVERY_METHODS = [
  'Stretching/Yoga',
  'Foam rolling',
  'Massage therapy',
  'Ice baths/Cold therapy',
  'Sauna/Heat therapy',
  'Compression therapy',
  'Active recovery',
  'Meditation/Mindfulness',
  'None currently',
];

// ============================================================================
// SECTION COMPONENTS
// ============================================================================

interface SectionProps {
  control: ReturnType<typeof useForm<IntakeFormData>>['control'];
  errors: ReturnType<typeof useForm<IntakeFormData>>['formState']['errors'];
  watch: ReturnType<typeof useForm<IntakeFormData>>['watch'];
  setValue: ReturnType<typeof useForm<IntakeFormData>>['setValue'];
}

const PersonalInfoSection: React.FC<SectionProps> = ({ control, errors }) => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Controller
        name="personalInfo.firstName"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="firstName">First Name *</Label>
            <Input {...field} id="firstName" placeholder="Enter your first name" />
            {errors?.personalInfo?.firstName && (
              <p className="text-sm text-red-500">{errors.personalInfo.firstName.message}</p>
            )}
          </div>
        )}
      />

      <Controller
        name="personalInfo.lastName"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="lastName">Last Name *</Label>
            <Input {...field} id="lastName" placeholder="Enter your last name" />
            {errors?.personalInfo?.lastName && (
              <p className="text-sm text-red-500">{errors.personalInfo.lastName.message}</p>
            )}
          </div>
        )}
      />
    </div>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Controller
        name="personalInfo.email"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="email">Email Address *</Label>
            <Input {...field} id="email" type="email" placeholder="your@email.com" />
            {errors?.personalInfo?.email && (
              <p className="text-sm text-red-500">{errors.personalInfo.email.message}</p>
            )}
          </div>
        )}
      />

      <Controller
        name="personalInfo.phone"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="phone">Phone Number *</Label>
            <Input {...field} id="phone" type="tel" placeholder="(555) 123-4567" />
            {errors?.personalInfo?.phone && (
              <p className="text-sm text-red-500">{errors.personalInfo.phone.message}</p>
            )}
          </div>
        )}
      />
    </div>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Controller
        name="personalInfo.dateOfBirth"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="dateOfBirth">Date of Birth *</Label>
            <Input {...field} id="dateOfBirth" type="date" />
            {errors?.personalInfo?.dateOfBirth && (
              <p className="text-sm text-red-500">{errors.personalInfo.dateOfBirth.message}</p>
            )}
          </div>
        )}
      />

      <Controller
        name="personalInfo.gender"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label>Gender *</Label>
            <Select value={field.value} onValueChange={field.onChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="male">Male</SelectItem>
                <SelectItem value="female">Female</SelectItem>
                <SelectItem value="non-binary">Non-binary</SelectItem>
                <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      />
    </div>

    <Separator className="my-6" />

    <h3 className="text-lg font-semibold flex items-center gap-2">
      <AlertTriangle className="h-5 w-5 text-orange-500" />
      Emergency Contact Information
    </h3>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Controller
        name="personalInfo.emergencyContactName"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="emergencyContactName">Contact Name *</Label>
            <Input {...field} id="emergencyContactName" placeholder="Full name" />
          </div>
        )}
      />

      <Controller
        name="personalInfo.emergencyContactPhone"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="emergencyContactPhone">Contact Phone *</Label>
            <Input {...field} id="emergencyContactPhone" type="tel" placeholder="(555) 123-4567" />
          </div>
        )}
      />

      <Controller
        name="personalInfo.emergencyContactRelation"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="emergencyContactRelation">Relationship *</Label>
            <Input {...field} id="emergencyContactRelation" placeholder="e.g., Spouse, Parent" />
          </div>
        )}
      />
    </div>
  </div>
);

const FitnessAssessmentSection: React.FC<SectionProps> = ({ control }) => (
  <div className="space-y-6">
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
      <p className="text-sm text-blue-800">
        <strong>Note:</strong> These measurements help us track your progress and customize your program.
        If you don't know exact values, provide your best estimate.
      </p>
    </div>

    <h3 className="text-lg font-semibold">Body Measurements</h3>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Controller
        name="fitnessAssessment.currentWeight"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="currentWeight">Current Weight (lbs) *</Label>
            <Input
              {...field}
              id="currentWeight"
              type="number"
              placeholder="e.g., 180"
              onChange={(e) => field.onChange(parseFloat(e.target.value))}
            />
          </div>
        )}
      />

      <Controller
        name="fitnessAssessment.goalWeight"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="goalWeight">Goal Weight (lbs)</Label>
            <Input
              {...field}
              id="goalWeight"
              type="number"
              placeholder="e.g., 165"
              onChange={(e) => field.onChange(parseFloat(e.target.value))}
            />
          </div>
        )}
      />

      <Controller
        name="fitnessAssessment.height"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="height">Height *</Label>
            <Input {...field} id="height" placeholder="e.g., 5'10&quot;" />
          </div>
        )}
      />
    </div>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Controller
        name="fitnessAssessment.bodyFatPercentage"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="bodyFatPercentage">Body Fat % (if known)</Label>
            <Input
              {...field}
              id="bodyFatPercentage"
              type="number"
              placeholder="e.g., 18"
              onChange={(e) => field.onChange(parseFloat(e.target.value))}
            />
          </div>
        )}
      />

      <Controller
        name="fitnessAssessment.restingHeartRate"
        control={control}
        render={({ field }) => (
          <div className="space-y-2">
            <Label htmlFor="restingHeartRate">Resting Heart Rate (bpm)</Label>
            <Input
              {...field}
              id="restingHeartRate"
              type="number"
              placeholder="e.g., 65"
              onChange={(e) => field.onChange(parseInt(e.target.value))}
            />
          </div>
        )}
      />

      <div className="space-y-2">
        <Label>Blood Pressure (if known)</Label>
        <div className="flex gap-2">
          <Controller
            name="fitnessAssessment.bloodPressureSystolic"
            control={control}
            render={({ field }) => (
              <Input
                {...field}
                type="number"
                placeholder="Systolic"
                onChange={(e) => field.onChange(parseInt(e.target.value))}
              />
            )}
          />
          <span className="flex items-center">/</span>
          <Controller
            name="fitnessAssessment.bloodPressureDiastolic"
            control={control}
            render={({ field }) => (
              <Input
                {...field}
                type="number"
                placeholder="Diastolic"
                onChange={(e) => field.onChange(parseInt(e.target.value))}
              />
            )}
          />
        </div>
      </div>
    </div>

    <Separator className="my-6" />

    <h3 className="text-lg font-semibold">Circumference Measurements (inches)</h3>
    <p className="text-sm text-gray-500 mb-4">Optional but helpful for tracking progress</p>

    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {['waist', 'hip', 'chest', 'arm', 'thigh'].map((measurement) => (
        <Controller
          key={measurement}
          name={`fitnessAssessment.${measurement}Circumference` as any}
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor={measurement} className="capitalize">{measurement}</Label>
              <Input
                {...field}
                id={measurement}
                type="number"
                placeholder="inches"
                onChange={(e) => field.onChange(parseFloat(e.target.value))}
              />
            </div>
          )}
        />
      ))}
    </div>
  </div>
);

const HealthHistorySection: React.FC<SectionProps> = ({ control, watch, setValue }) => {
  const medicalConditions = watch('healthHistory.medicalConditions') || [];

  return (
    <div className="space-y-6">
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-800">Important Health Screening</p>
            <p className="text-sm text-red-700 mt-1">
              Please answer the following questions honestly. This information is confidential and
              essential for your safety. If you answer YES to any PAR-Q+ question, physician
              clearance may be required before beginning an exercise program.
            </p>
          </div>
        </div>
      </div>

      <h3 className="text-lg font-semibold">PAR-Q+ Health Screening Questions</h3>

      <div className="space-y-4">
        {[
          { name: 'heartCondition', question: 'Has your doctor ever said that you have a heart condition and that you should only do physical activity recommended by a doctor?' },
          { name: 'chestPain', question: 'Do you feel pain in your chest when you do physical activity?' },
          { name: 'dizziness', question: 'Do you lose your balance because of dizziness or do you ever lose consciousness?' },
          { name: 'boneJointProblem', question: 'Do you have a bone or joint problem that could be made worse by a change in your physical activity?' },
          { name: 'bloodPressureMedication', question: 'Is your doctor currently prescribing drugs for your blood pressure or heart condition?' },
          { name: 'otherReason', question: 'Do you know of any other reason why you should not do physical activity?' },
        ].map((item) => (
          <Controller
            key={item.name}
            name={`healthHistory.${item.name}` as any}
            control={control}
            render={({ field }) => (
              <div className="flex items-start gap-3 p-4 border rounded-lg hover:bg-gray-50">
                <Checkbox
                  id={item.name}
                  checked={field.value}
                  onCheckedChange={field.onChange}
                  className="mt-1"
                />
                <Label htmlFor={item.name} className="text-sm cursor-pointer">
                  {item.question}
                </Label>
              </div>
            )}
          />
        ))}
      </div>

      <Separator className="my-6" />

      <h3 className="text-lg font-semibold">Medical Conditions</h3>
      <p className="text-sm text-gray-500 mb-4">Check all that apply</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {MEDICAL_CONDITIONS.map((condition) => (
          <div key={condition} className="flex items-center gap-3 p-3 border rounded-lg hover:bg-gray-50">
            <Checkbox
              id={condition}
              checked={medicalConditions.includes(condition)}
              onCheckedChange={(checked) => {
                if (checked) {
                  setValue('healthHistory.medicalConditions', [...medicalConditions, condition]);
                } else {
                  setValue('healthHistory.medicalConditions', medicalConditions.filter((c: string) => c !== condition));
                }
              }}
            />
            <Label htmlFor={condition} className="text-sm cursor-pointer">{condition}</Label>
          </div>
        ))}
      </div>

      <Separator className="my-6" />

      <h3 className="text-lg font-semibold">Medications & Allergies</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Controller
          name="healthHistory.currentMedications"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="currentMedications">Current Medications</Label>
              <Textarea
                {...field}
                id="currentMedications"
                placeholder="List all current medications, dosages, and frequency..."
                className="min-h-[100px]"
              />
            </div>
          )}
        />

        <Controller
          name="healthHistory.allergies"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="allergies">Allergies</Label>
              <Textarea
                {...field}
                id="allergies"
                placeholder="List any allergies (medications, foods, environmental)..."
                className="min-h-[100px]"
              />
            </div>
          )}
        />
      </div>

      <Separator className="my-6" />

      <h3 className="text-lg font-semibold">Injury History</h3>

      <div className="space-y-6">
        <Controller
          name="healthHistory.currentInjuries"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="currentInjuries">Current Injuries or Pain</Label>
              <Textarea
                {...field}
                id="currentInjuries"
                placeholder="Describe any current injuries, chronic pain, or areas of discomfort..."
                className="min-h-[100px]"
              />
            </div>
          )}
        />

        <Controller
          name="healthHistory.pastInjuries"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="pastInjuries">Past Injuries & Surgeries</Label>
              <Textarea
                {...field}
                id="pastInjuries"
                placeholder="List any past injuries, surgeries, or procedures that might affect training..."
                className="min-h-[100px]"
              />
            </div>
          )}
        />

        <Controller
          name="healthHistory.movementLimitations"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="movementLimitations">Movement Limitations</Label>
              <Textarea
                {...field}
                id="movementLimitations"
                placeholder="Are there any movements or exercises you cannot perform or should avoid?"
                className="min-h-[100px]"
              />
            </div>
          )}
        />
      </div>

      <Separator className="my-6" />

      <h3 className="text-lg font-semibold">Physician Information</h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Controller
          name="healthHistory.physicianName"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="physicianName">Physician Name</Label>
              <Input {...field} id="physicianName" placeholder="Dr. Smith" />
            </div>
          )}
        />

        <Controller
          name="healthHistory.physicianPhone"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="physicianPhone">Physician Phone</Label>
              <Input {...field} id="physicianPhone" type="tel" placeholder="(555) 123-4567" />
            </div>
          )}
        />

        <Controller
          name="healthHistory.lastPhysicalDate"
          control={control}
          render={({ field }) => (
            <div className="space-y-2">
              <Label htmlFor="lastPhysicalDate">Last Physical Exam</Label>
              <Input {...field} id="lastPhysicalDate" type="date" />
            </div>
          )}
        />
      </div>

      <Controller
        name="healthHistory.physicianClearance"
        control={control}
        render={({ field }) => (
          <div className="flex items-start gap-3 p-4 border rounded-lg bg-green-50 border-green-200 mt-4">
            <Checkbox
              id="physicianClearance"
              checked={field.value}
              onCheckedChange={field.onChange}
              className="mt-1"
            />
            <Label htmlFor="physicianClearance" className="text-sm cursor-pointer">
              I have been cleared by a physician to participate in physical exercise, or I understand
              that I should consult with a physician before beginning any exercise program.
            </Label>
          </div>
        )}
      />
    </div>
  );
};

// Continue with remaining sections...
// (TrainingExperienceSection, GoalIdentificationSection, NutritionHabitsSection,
//  LifestyleFactorsSection, ScheduleAvailabilitySection, ConsentSection)

// ============================================================================
// MAIN COMPONENT
// ============================================================================

interface ProfessionalIntakeFormProps {
  onSubmit?: (data: IntakeFormData) => void;
  onSaveDraft?: (data: Partial<IntakeFormData>) => void;
  initialData?: Partial<IntakeFormData>;
}

export const ProfessionalIntakeForm: React.FC<ProfessionalIntakeFormProps> = ({
  onSubmit,
  onSaveDraft,
  initialData,
}) => {
  const [currentSection, setCurrentSection] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    control,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
    getValues,
  } = useForm<IntakeFormData>({
    resolver: zodResolver(fullIntakeSchema),
    defaultValues: initialData || {
      personalInfo: { gender: 'prefer-not-to-say' },
      healthHistory: { medicalConditions: [], physicianClearance: false },
      trainingExperience: { homeEquipment: [], preferredTrainingDays: [] },
      goalIdentification: { secondaryGoals: [] },
      nutritionHabits: { foodAllergies: [], dietaryRestrictions: [] },
      lifestyleFactors: { sleepIssues: [], recoveryMethods: [] },
      scheduleAvailability: {
        mondayAvailability: [],
        tuesdayAvailability: [],
        wednesdayAvailability: [],
        thursdayAvailability: [],
        fridayAvailability: [],
        saturdayAvailability: [],
        sundayAvailability: [],
      },
      consent: {
        informedConsent: false,
        liabilityWaiver: false,
        privacyPolicy: false,
        photoRelease: false,
        medicalDisclosure: false,
        programAgreement: false,
        signatureDate: new Date().toISOString().split('T')[0],
      },
    },
    mode: 'onChange',
  });

  const progress = ((currentSection + 1) / FORM_SECTIONS.length) * 100;

  const handleNext = () => {
    if (currentSection < FORM_SECTIONS.length - 1) {
      setCurrentSection(prev => prev + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handlePrevious = () => {
    if (currentSection > 0) {
      setCurrentSection(prev => prev - 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleSaveDraft = useCallback(() => {
    const currentData = getValues();
    onSaveDraft?.(currentData);
    toast.success('Draft saved successfully!');
  }, [getValues, onSaveDraft]);

  const onFormSubmit = async (data: IntakeFormData) => {
    setIsSubmitting(true);
    try {
      await onSubmit?.(data);
      toast.success('Intake form submitted successfully!');
    } catch {
      toast.error('Failed to submit form. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderSection = () => {
    const sectionProps = { control, errors, watch, setValue };

    switch (FORM_SECTIONS[currentSection].id) {
      case 'personal':
        return <PersonalInfoSection {...sectionProps} />;
      case 'fitness':
        return <FitnessAssessmentSection {...sectionProps} />;
      case 'health':
        return <HealthHistorySection {...sectionProps} />;
      // Add remaining sections here
      default:
        return <PersonalInfoSection {...sectionProps} />;
    }
  };

  const CurrentIcon = FORM_SECTIONS[currentSection].icon;

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">
            REPZ Client Intake Form
          </h1>
          <p className="text-gray-600">
            Complete this comprehensive assessment to help us create your personalized program
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700">
              Section {currentSection + 1} of {FORM_SECTIONS.length}
            </span>
            <span className="text-sm font-medium text-repz-orange">
              {Math.round(progress)}% Complete
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Section Navigation */}
        <div className="flex overflow-x-auto gap-2 mb-8 pb-2">
          {FORM_SECTIONS.map((section, index) => {
            const Icon = section.icon;
            const isActive = index === currentSection;
            const isCompleted = index < currentSection;

            return (
              <button
                key={section.id}
                onClick={() => setCurrentSection(index)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-repz-orange text-white'
                    : isCompleted
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {isCompleted ? (
                  <CheckCircle className="h-4 w-4" />
                ) : (
                  <Icon className="h-4 w-4" />
                )}
                <span className="hidden md:inline">{section.title}</span>
              </button>
            );
          })}
        </div>

        {/* Form Card */}
        <Card className="shadow-lg">
          <CardHeader className="border-b bg-gray-50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-repz-orange/10 rounded-lg">
                <CurrentIcon className="h-6 w-6 text-repz-orange" />
              </div>
              <div>
                <CardTitle>{FORM_SECTIONS[currentSection].title}</CardTitle>
                <CardDescription>{FORM_SECTIONS[currentSection].description}</CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="p-6 md:p-8">
            <form onSubmit={handleSubmit(onFormSubmit)}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentSection}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  {renderSection()}
                </motion.div>
              </AnimatePresence>

              {/* Navigation Buttons */}
              <div className="flex justify-between items-center mt-8 pt-6 border-t">
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handlePrevious}
                    disabled={currentSection === 0}
                    className="gap-2"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    Previous
                  </Button>

                  <Button
                    type="button"
                    variant="ghost"
                    onClick={handleSaveDraft}
                    className="gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Save Draft
                  </Button>
                </div>

                {currentSection === FORM_SECTIONS.length - 1 ? (
                  <Button
                    type="submit"
                    disabled={isSubmitting}
                    className="gap-2 bg-repz-orange hover:bg-repz-orange/90"
                  >
                    {isSubmitting ? (
                      <>Submitting...</>
                    ) : (
                      <>
                        <Send className="h-4 w-4" />
                        Submit Form
                      </>
                    )}
                  </Button>
                ) : (
                  <Button
                    type="button"
                    onClick={handleNext}
                    className="gap-2 bg-repz-orange hover:bg-repz-orange/90"
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-500">
          <p>
            Your information is confidential and protected.
            See our <a href="/privacy-policy" className="text-repz-orange hover:underline">Privacy Policy</a>.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalIntakeForm;
