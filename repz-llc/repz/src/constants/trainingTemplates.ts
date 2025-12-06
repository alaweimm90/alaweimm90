/**
 * REPZ Comprehensive Training Program Templates
 *
 * Evidence-based programming templates synthesizing methodologies from:
 * - Leading platforms: Trainerize, TrueCoach, PT Distinction, Future
 * - Research: NSCA, ACSM, peer-reviewed exercise science
 * - Elite coaches: Chris Bumstead's team, Derek Lunsford's protocols, Hany Rambod, etc.
 *
 * Features:
 * - Progressive overload structures
 * - Deload protocols
 * - Volume landmarks (MEV, MAV, MRV)
 * - RPE and RIR tracking
 * - Nutrition periodization frameworks
 * - Recovery metrics
 * - Adaptive algorithm-driven progression
 *
 * @version 2.0.0
 * @author REPZ Development Team
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export type TrainingGoal =
  | 'hypertrophy'
  | 'strength'
  | 'fat-loss'
  | 'powerbuilding'
  | 'athletic-performance'
  | 'general-fitness'
  | 'bodybuilding-prep'
  | 'powerlifting'
  | 'endurance'
  | 'rehabilitation';

export type ExperienceLevel = 'beginner' | 'intermediate' | 'advanced' | 'elite';

export type MuscleGroup =
  | 'chest'
  | 'back'
  | 'shoulders'
  | 'biceps'
  | 'triceps'
  | 'quadriceps'
  | 'hamstrings'
  | 'glutes'
  | 'calves'
  | 'abs'
  | 'forearms'
  | 'traps';

export type MovementPattern =
  | 'horizontal-push'
  | 'horizontal-pull'
  | 'vertical-push'
  | 'vertical-pull'
  | 'hip-hinge'
  | 'squat'
  | 'lunge'
  | 'carry'
  | 'rotation'
  | 'isolation';

export type IntensityTechnique =
  | 'straight-sets'
  | 'drop-sets'
  | 'rest-pause'
  | 'myo-reps'
  | 'cluster-sets'
  | 'supersets'
  | 'giant-sets'
  | 'pre-exhaust'
  | 'post-exhaust'
  | 'mechanical-drop'
  | 'lengthened-partials'
  | 'pause-reps'
  | 'tempo-manipulation'
  | 'blood-flow-restriction';

export interface Exercise {
  id: string;
  name: string;
  muscleGroups: MuscleGroup[];
  movementPattern: MovementPattern;
  equipment: string[];
  difficulty: ExperienceLevel;
  videoUrl?: string;
  cues: string[];
  alternatives: string[];
}

export interface SetPrescription {
  sets: number;
  reps: string; // e.g., "8-12" or "5"
  rpe?: number; // 1-10 scale
  rir?: number; // Reps in Reserve
  restSeconds: number;
  tempo?: string; // e.g., "3-1-2-0" (eccentric-pause-concentric-pause)
  technique?: IntensityTechnique;
  notes?: string;
}

export interface ExercisePrescription {
  exercise: Exercise;
  prescription: SetPrescription;
  warmupSets?: number;
  progressionScheme: ProgressionScheme;
}

export interface WorkoutSession {
  id: string;
  name: string;
  dayOfWeek?: number; // 0-6
  focus: MuscleGroup[];
  duration: number; // minutes
  exercises: ExercisePrescription[];
  warmup: WarmupProtocol;
  cooldown: CooldownProtocol;
  notes?: string;
}

export interface WarmupProtocol {
  duration: number;
  components: {
    type: 'cardio' | 'dynamic-stretch' | 'activation' | 'movement-prep';
    exercises: string[];
    duration: number;
  }[];
}

export interface CooldownProtocol {
  duration: number;
  components: {
    type: 'static-stretch' | 'foam-roll' | 'breathing' | 'mobility';
    exercises: string[];
    duration: number;
  }[];
}

export interface ProgressionScheme {
  type: 'linear' | 'double-progression' | 'wave' | 'percentage-based' | 'rpe-based' | 'autoregulated';
  parameters: {
    incrementWeight?: number; // lbs per successful session
    repTarget?: number;
    setTarget?: number;
    rpeTarget?: number;
    deloadFrequency?: number; // weeks
    deloadPercentage?: number;
  };
}

export interface MesocyclePhase {
  name: string;
  weeks: number;
  focus: string;
  volumeMultiplier: number; // 1.0 = baseline
  intensityMultiplier: number;
  rpeRange: [number, number];
  deloadWeek: boolean;
}

export interface PeriodizationModel {
  name: string;
  phases: MesocyclePhase[];
  totalWeeks: number;
  description: string;
}

export interface VolumeGuidelines {
  muscleGroup: MuscleGroup;
  mev: number; // Minimum Effective Volume (sets/week)
  mav: number; // Maximum Adaptive Volume (sets/week)
  mrv: number; // Maximum Recoverable Volume (sets/week)
  frequency: number; // times per week
  notes: string;
}

export interface NutritionPeriodization {
  phase: string;
  calorieMultiplier: number; // relative to maintenance
  proteinGramsPerKg: number;
  carbsPercentage: number;
  fatsPercentage: number;
  mealTiming: {
    preworkout: string;
    postworkout: string;
    general: string;
  };
  supplements: string[];
}

export interface RecoveryMetrics {
  sleepHoursTarget: number;
  restDaysPerWeek: number;
  activeRecoveryProtocol: string[];
  stressManagement: string[];
  hydrationTarget: string;
  mobilityWork: string[];
}

export interface TrainingProgram {
  id: string;
  name: string;
  description: string;
  goal: TrainingGoal;
  experienceLevel: ExperienceLevel;
  durationWeeks: number;
  daysPerWeek: number;
  sessionDuration: number; // minutes
  periodization: PeriodizationModel;
  weeks: WeeklyPlan[];
  volumeGuidelines: VolumeGuidelines[];
  nutritionPeriodization: NutritionPeriodization[];
  recoveryMetrics: RecoveryMetrics;
  adaptiveRules: AdaptiveRule[];
  tags: string[];
}

export interface WeeklyPlan {
  weekNumber: number;
  phase: string;
  sessions: WorkoutSession[];
  weeklyVolume: Record<MuscleGroup, number>;
  notes: string;
  checkInPrompts: string[];
}

export interface AdaptiveRule {
  trigger: string;
  condition: string;
  action: string;
  priority: number;
}

// ============================================================================
// VOLUME LANDMARKS BY MUSCLE GROUP
// Based on Dr. Mike Israetel's research and Renaissance Periodization guidelines
// ============================================================================

export const VOLUME_LANDMARKS: VolumeGuidelines[] = [
  {
    muscleGroup: 'chest',
    mev: 10,
    mav: 18,
    mrv: 22,
    frequency: 2,
    notes: 'Most respond well to 12-20 sets/week. Emphasize stretch position exercises.'
  },
  {
    muscleGroup: 'back',
    mev: 10,
    mav: 20,
    mrv: 25,
    frequency: 2,
    notes: 'Can handle high volume. Split between vertical and horizontal pulls.'
  },
  {
    muscleGroup: 'shoulders',
    mev: 8,
    mav: 16,
    mrv: 22,
    frequency: 2,
    notes: 'Front delts get indirect work from pressing. Focus on side and rear delts.'
  },
  {
    muscleGroup: 'biceps',
    mev: 8,
    mav: 14,
    mrv: 20,
    frequency: 2,
    notes: 'Get indirect work from pulling. Emphasize stretch and peak contraction.'
  },
  {
    muscleGroup: 'triceps',
    mev: 6,
    mav: 12,
    mrv: 18,
    frequency: 2,
    notes: 'Get significant indirect work from pressing movements.'
  },
  {
    muscleGroup: 'quadriceps',
    mev: 8,
    mav: 16,
    mrv: 20,
    frequency: 2,
    notes: 'High systemic fatigue. Prioritize compound movements.'
  },
  {
    muscleGroup: 'hamstrings',
    mev: 6,
    mav: 12,
    mrv: 16,
    frequency: 2,
    notes: 'Include both hip hinge and leg curl variations.'
  },
  {
    muscleGroup: 'glutes',
    mev: 4,
    mav: 12,
    mrv: 16,
    frequency: 2,
    notes: 'Get indirect work from squats and hip hinges.'
  },
  {
    muscleGroup: 'calves',
    mev: 8,
    mav: 16,
    mrv: 20,
    frequency: 3,
    notes: 'Stubborn muscle group. Higher frequency often beneficial.'
  },
  {
    muscleGroup: 'abs',
    mev: 0,
    mav: 16,
    mrv: 20,
    frequency: 3,
    notes: 'Get indirect work from compounds. Direct work for aesthetics.'
  },
  {
    muscleGroup: 'traps',
    mev: 0,
    mav: 12,
    mrv: 20,
    frequency: 2,
    notes: 'Get significant indirect work from pulling and deadlifts.'
  },
  {
    muscleGroup: 'forearms',
    mev: 0,
    mav: 8,
    mrv: 14,
    frequency: 2,
    notes: 'Get indirect work from gripping. Direct work for weak points.'
  },
];

// ============================================================================
// PERIODIZATION MODELS
// ============================================================================

export const PERIODIZATION_MODELS: Record<string, PeriodizationModel> = {
  linearProgression: {
    name: 'Linear Progression',
    description: 'Simple progressive overload model ideal for beginners. Add weight each session.',
    totalWeeks: 12,
    phases: [
      { name: 'Foundation', weeks: 4, focus: 'Technique and base building', volumeMultiplier: 0.8, intensityMultiplier: 0.7, rpeRange: [6, 7], deloadWeek: false },
      { name: 'Accumulation', weeks: 4, focus: 'Volume building', volumeMultiplier: 1.0, intensityMultiplier: 0.8, rpeRange: [7, 8], deloadWeek: false },
      { name: 'Intensification', weeks: 3, focus: 'Intensity increase', volumeMultiplier: 0.9, intensityMultiplier: 0.9, rpeRange: [8, 9], deloadWeek: false },
      { name: 'Deload', weeks: 1, focus: 'Recovery', volumeMultiplier: 0.5, intensityMultiplier: 0.6, rpeRange: [5, 6], deloadWeek: true },
    ],
  },

  undulatingPeriodization: {
    name: 'Daily Undulating Periodization (DUP)',
    description: 'Varies intensity and volume within each week. Great for intermediate lifters.',
    totalWeeks: 8,
    phases: [
      { name: 'Hypertrophy Focus', weeks: 2, focus: 'High volume, moderate intensity', volumeMultiplier: 1.1, intensityMultiplier: 0.75, rpeRange: [7, 8], deloadWeek: false },
      { name: 'Strength Focus', weeks: 2, focus: 'Moderate volume, high intensity', volumeMultiplier: 0.9, intensityMultiplier: 0.85, rpeRange: [8, 9], deloadWeek: false },
      { name: 'Power Focus', weeks: 2, focus: 'Low volume, max intensity', volumeMultiplier: 0.7, intensityMultiplier: 0.95, rpeRange: [8, 9], deloadWeek: false },
      { name: 'Deload', weeks: 1, focus: 'Recovery and supercompensation', volumeMultiplier: 0.5, intensityMultiplier: 0.6, rpeRange: [5, 6], deloadWeek: true },
      { name: 'Peak', weeks: 1, focus: 'Test new maxes', volumeMultiplier: 0.6, intensityMultiplier: 1.0, rpeRange: [9, 10], deloadWeek: false },
    ],
  },

  blockPeriodization: {
    name: 'Block Periodization',
    description: 'Concentrated loading blocks. Ideal for advanced athletes with specific peaking goals.',
    totalWeeks: 12,
    phases: [
      { name: 'Accumulation Block', weeks: 4, focus: 'High volume, technique work', volumeMultiplier: 1.2, intensityMultiplier: 0.7, rpeRange: [6, 8], deloadWeek: false },
      { name: 'Transmutation Block', weeks: 4, focus: 'Sport-specific strength', volumeMultiplier: 0.9, intensityMultiplier: 0.85, rpeRange: [8, 9], deloadWeek: false },
      { name: 'Realization Block', weeks: 3, focus: 'Peaking for competition', volumeMultiplier: 0.6, intensityMultiplier: 0.95, rpeRange: [9, 10], deloadWeek: false },
      { name: 'Taper', weeks: 1, focus: 'Competition readiness', volumeMultiplier: 0.4, intensityMultiplier: 0.9, rpeRange: [7, 8], deloadWeek: true },
    ],
  },

  conjugateMethod: {
    name: 'Conjugate/Westside Method',
    description: 'Max effort and dynamic effort rotation. For advanced powerlifters.',
    totalWeeks: 4,
    phases: [
      { name: 'Max Effort Upper', weeks: 1, focus: 'Work up to 1-3RM', volumeMultiplier: 0.7, intensityMultiplier: 1.0, rpeRange: [9, 10], deloadWeek: false },
      { name: 'Dynamic Effort Lower', weeks: 1, focus: 'Speed work 50-60% 1RM', volumeMultiplier: 0.8, intensityMultiplier: 0.55, rpeRange: [6, 7], deloadWeek: false },
      { name: 'Max Effort Lower', weeks: 1, focus: 'Work up to 1-3RM', volumeMultiplier: 0.7, intensityMultiplier: 1.0, rpeRange: [9, 10], deloadWeek: false },
      { name: 'Dynamic Effort Upper', weeks: 1, focus: 'Speed work 50-60% 1RM', volumeMultiplier: 0.8, intensityMultiplier: 0.55, rpeRange: [6, 7], deloadWeek: false },
    ],
  },

  renaissancePeriodization: {
    name: 'Renaissance Periodization (RP) Style',
    description: 'MEV to MRV progression with systematic deloads. Science-based hypertrophy.',
    totalWeeks: 6,
    phases: [
      { name: 'MEV Week', weeks: 1, focus: 'Minimum effective volume', volumeMultiplier: 0.7, intensityMultiplier: 0.75, rpeRange: [6, 7], deloadWeek: false },
      { name: 'MAV Week 1', weeks: 1, focus: 'Building toward MAV', volumeMultiplier: 0.85, intensityMultiplier: 0.8, rpeRange: [7, 8], deloadWeek: false },
      { name: 'MAV Week 2', weeks: 1, focus: 'At MAV', volumeMultiplier: 1.0, intensityMultiplier: 0.82, rpeRange: [8, 8], deloadWeek: false },
      { name: 'Overreaching', weeks: 1, focus: 'Approaching MRV', volumeMultiplier: 1.15, intensityMultiplier: 0.85, rpeRange: [8, 9], deloadWeek: false },
      { name: 'MRV Week', weeks: 1, focus: 'Maximum recoverable volume', volumeMultiplier: 1.25, intensityMultiplier: 0.85, rpeRange: [9, 10], deloadWeek: false },
      { name: 'Deload', weeks: 1, focus: 'Recovery and adaptation', volumeMultiplier: 0.5, intensityMultiplier: 0.6, rpeRange: [5, 6], deloadWeek: true },
    ],
  },
};

// ============================================================================
// TRAINING SPLIT TEMPLATES
// ============================================================================

export const TRAINING_SPLITS = {
  pushPullLegs: {
    name: 'Push/Pull/Legs (PPL)',
    daysPerWeek: 6,
    description: 'Classic bodybuilding split. Each muscle hit 2x/week.',
    schedule: [
      { day: 'Monday', focus: ['chest', 'shoulders', 'triceps'], name: 'Push A' },
      { day: 'Tuesday', focus: ['back', 'biceps', 'forearms'], name: 'Pull A' },
      { day: 'Wednesday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Legs A' },
      { day: 'Thursday', focus: ['chest', 'shoulders', 'triceps'], name: 'Push B' },
      { day: 'Friday', focus: ['back', 'biceps', 'forearms'], name: 'Pull B' },
      { day: 'Saturday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Legs B' },
      { day: 'Sunday', focus: [], name: 'Rest' },
    ],
  },

  upperLower: {
    name: 'Upper/Lower Split',
    daysPerWeek: 4,
    description: 'Balanced split for intermediates. Good recovery.',
    schedule: [
      { day: 'Monday', focus: ['chest', 'back', 'shoulders', 'biceps', 'triceps'], name: 'Upper A' },
      { day: 'Tuesday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Lower A' },
      { day: 'Wednesday', focus: [], name: 'Rest' },
      { day: 'Thursday', focus: ['chest', 'back', 'shoulders', 'biceps', 'triceps'], name: 'Upper B' },
      { day: 'Friday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Lower B' },
      { day: 'Saturday', focus: [], name: 'Rest' },
      { day: 'Sunday', focus: [], name: 'Rest' },
    ],
  },

  fullBody: {
    name: 'Full Body',
    daysPerWeek: 3,
    description: 'Hit everything each session. Great for beginners or busy schedules.',
    schedule: [
      { day: 'Monday', focus: ['chest', 'back', 'shoulders', 'quadriceps', 'hamstrings', 'biceps', 'triceps'], name: 'Full Body A' },
      { day: 'Tuesday', focus: [], name: 'Rest' },
      { day: 'Wednesday', focus: ['chest', 'back', 'shoulders', 'quadriceps', 'hamstrings', 'biceps', 'triceps'], name: 'Full Body B' },
      { day: 'Thursday', focus: [], name: 'Rest' },
      { day: 'Friday', focus: ['chest', 'back', 'shoulders', 'quadriceps', 'hamstrings', 'biceps', 'triceps'], name: 'Full Body C' },
      { day: 'Saturday', focus: [], name: 'Rest' },
      { day: 'Sunday', focus: [], name: 'Rest' },
    ],
  },

  arnoldSplit: {
    name: 'Arnold Split',
    daysPerWeek: 6,
    description: 'Classic bodybuilding split used by Arnold Schwarzenegger.',
    schedule: [
      { day: 'Monday', focus: ['chest', 'back'], name: 'Chest & Back' },
      { day: 'Tuesday', focus: ['shoulders', 'biceps', 'triceps'], name: 'Shoulders & Arms' },
      { day: 'Wednesday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Legs' },
      { day: 'Thursday', focus: ['chest', 'back'], name: 'Chest & Back' },
      { day: 'Friday', focus: ['shoulders', 'biceps', 'triceps'], name: 'Shoulders & Arms' },
      { day: 'Saturday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Legs' },
      { day: 'Sunday', focus: [], name: 'Rest' },
    ],
  },

  broSplit: {
    name: 'Bro Split (Body Part)',
    daysPerWeek: 5,
    description: 'One muscle group per day. High volume per session.',
    schedule: [
      { day: 'Monday', focus: ['chest'], name: 'Chest Day' },
      { day: 'Tuesday', focus: ['back'], name: 'Back Day' },
      { day: 'Wednesday', focus: ['shoulders'], name: 'Shoulder Day' },
      { day: 'Thursday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Leg Day' },
      { day: 'Friday', focus: ['biceps', 'triceps'], name: 'Arm Day' },
      { day: 'Saturday', focus: [], name: 'Rest' },
      { day: 'Sunday', focus: [], name: 'Rest' },
    ],
  },

  powerbuilding: {
    name: 'Powerbuilding (PHUL Style)',
    daysPerWeek: 4,
    description: 'Combines powerlifting and bodybuilding. Strength + hypertrophy.',
    schedule: [
      { day: 'Monday', focus: ['chest', 'back', 'shoulders'], name: 'Upper Power' },
      { day: 'Tuesday', focus: ['quadriceps', 'hamstrings', 'glutes'], name: 'Lower Power' },
      { day: 'Wednesday', focus: [], name: 'Rest' },
      { day: 'Thursday', focus: ['chest', 'back', 'shoulders', 'biceps', 'triceps'], name: 'Upper Hypertrophy' },
      { day: 'Friday', focus: ['quadriceps', 'hamstrings', 'glutes', 'calves'], name: 'Lower Hypertrophy' },
      { day: 'Saturday', focus: [], name: 'Rest' },
      { day: 'Sunday', focus: [], name: 'Rest' },
    ],
  },
};

// ============================================================================
// NUTRITION PERIODIZATION TEMPLATES
// ============================================================================

export const NUTRITION_PHASES: Record<string, NutritionPeriodization> = {
  bulking: {
    phase: 'Muscle Building (Bulk)',
    calorieMultiplier: 1.15, // 15% surplus
    proteinGramsPerKg: 2.2,
    carbsPercentage: 50,
    fatsPercentage: 25,
    mealTiming: {
      preworkout: '30-60g carbs, 20-30g protein 1-2 hours before',
      postworkout: '40-60g carbs, 30-40g protein within 1 hour',
      general: '4-6 meals spread evenly throughout the day',
    },
    supplements: ['Creatine 5g/day', 'Protein powder as needed', 'Vitamin D3', 'Omega-3s'],
  },

  cutting: {
    phase: 'Fat Loss (Cut)',
    calorieMultiplier: 0.80, // 20% deficit
    proteinGramsPerKg: 2.4, // Higher protein to preserve muscle
    carbsPercentage: 35,
    fatsPercentage: 30,
    mealTiming: {
      preworkout: '20-30g carbs, 20-30g protein 1-2 hours before',
      postworkout: '30-40g carbs, 30-40g protein within 1 hour',
      general: 'Prioritize protein at each meal, carbs around training',
    },
    supplements: ['Creatine 5g/day', 'Protein powder', 'Caffeine pre-workout', 'Multivitamin'],
  },

  maintenance: {
    phase: 'Maintenance',
    calorieMultiplier: 1.0,
    proteinGramsPerKg: 2.0,
    carbsPercentage: 45,
    fatsPercentage: 30,
    mealTiming: {
      preworkout: '30-40g carbs, 20-30g protein 1-2 hours before',
      postworkout: '30-50g carbs, 30-40g protein within 1 hour',
      general: 'Balanced meals throughout the day',
    },
    supplements: ['Creatine 5g/day', 'Protein powder as needed', 'Vitamin D3'],
  },

  recomp: {
    phase: 'Body Recomposition',
    calorieMultiplier: 1.0, // Slight surplus on training days, deficit on rest
    proteinGramsPerKg: 2.4,
    carbsPercentage: 40,
    fatsPercentage: 30,
    mealTiming: {
      preworkout: '40-50g carbs, 25-35g protein 1-2 hours before',
      postworkout: '50-60g carbs, 35-45g protein within 1 hour',
      general: 'Carb cycling: higher on training days, lower on rest days',
    },
    supplements: ['Creatine 5g/day', 'Protein powder', 'Omega-3s', 'Vitamin D3'],
  },

  competitionPrep: {
    phase: 'Competition Prep',
    calorieMultiplier: 0.70, // Aggressive deficit
    proteinGramsPerKg: 2.8, // Very high protein
    carbsPercentage: 30,
    fatsPercentage: 25,
    mealTiming: {
      preworkout: '20-30g carbs, 30g protein 1-2 hours before',
      postworkout: '30-40g carbs, 40g protein within 30 minutes',
      general: 'Frequent small meals, carbs primarily around training',
    },
    supplements: ['Creatine 5g/day', 'Protein powder', 'BCAAs intra-workout', 'Electrolytes', 'Caffeine'],
  },
};

// ============================================================================
// ADAPTIVE PROGRESSION RULES
// ============================================================================

export const ADAPTIVE_RULES: AdaptiveRule[] = [
  {
    trigger: 'All sets completed at target RPE',
    condition: 'RPE <= target for all working sets',
    action: 'Increase weight by 2.5-5 lbs next session',
    priority: 1,
  },
  {
    trigger: 'Exceeded rep target',
    condition: 'Reps > target range on final set',
    action: 'Increase weight by 5-10 lbs next session',
    priority: 2,
  },
  {
    trigger: 'Missed rep target',
    condition: 'Reps < minimum target range',
    action: 'Maintain weight, focus on technique',
    priority: 3,
  },
  {
    trigger: 'Consecutive session failures',
    condition: 'Missed target 2+ sessions in a row',
    action: 'Reduce weight by 10%, rebuild',
    priority: 4,
  },
  {
    trigger: 'High fatigue indicators',
    condition: 'Sleep < 6 hours OR stress rating > 8',
    action: 'Reduce volume by 20-30% for session',
    priority: 5,
  },
  {
    trigger: 'Recovery deficit',
    condition: 'Soreness rating > 7 for 48+ hours',
    action: 'Add extra rest day or deload session',
    priority: 6,
  },
  {
    trigger: 'Plateau detected',
    condition: 'No progress for 3+ weeks',
    action: 'Change exercise variation or rep scheme',
    priority: 7,
  },
  {
    trigger: 'Approaching MRV',
    condition: 'Weekly volume > 90% of MRV',
    action: 'Schedule deload within 1-2 weeks',
    priority: 8,
  },
];

// ============================================================================
// DELOAD PROTOCOLS
// ============================================================================

export const DELOAD_PROTOCOLS = {
  volumeDeload: {
    name: 'Volume Deload',
    description: 'Reduce sets by 40-50%, maintain intensity',
    volumeReduction: 0.5,
    intensityReduction: 0,
    duration: 1,
    bestFor: ['Accumulated fatigue', 'Joint stress'],
  },

  intensityDeload: {
    name: 'Intensity Deload',
    description: 'Reduce weight by 40-50%, maintain volume',
    volumeReduction: 0,
    intensityReduction: 0.5,
    duration: 1,
    bestFor: ['CNS fatigue', 'Strength plateau'],
  },

  combinedDeload: {
    name: 'Combined Deload',
    description: 'Reduce both volume and intensity by 30-40%',
    volumeReduction: 0.35,
    intensityReduction: 0.35,
    duration: 1,
    bestFor: ['General fatigue', 'End of mesocycle'],
  },

  activeRecovery: {
    name: 'Active Recovery Week',
    description: 'Light movement, mobility work, no resistance training',
    volumeReduction: 1.0,
    intensityReduction: 1.0,
    duration: 1,
    bestFor: ['Severe fatigue', 'Injury prevention', 'Mental break'],
  },
};

// ============================================================================
// RPE/RIR CONVERSION TABLE
// ============================================================================

export const RPE_RIR_TABLE = [
  { rpe: 10, rir: 0, description: 'Maximum effort, no reps left', percentage: 100 },
  { rpe: 9.5, rir: 0.5, description: 'Could maybe do 1 more', percentage: 97 },
  { rpe: 9, rir: 1, description: '1 rep in reserve', percentage: 95 },
  { rpe: 8.5, rir: 1.5, description: '1-2 reps in reserve', percentage: 92 },
  { rpe: 8, rir: 2, description: '2 reps in reserve', percentage: 90 },
  { rpe: 7.5, rir: 2.5, description: '2-3 reps in reserve', percentage: 87 },
  { rpe: 7, rir: 3, description: '3 reps in reserve', percentage: 85 },
  { rpe: 6.5, rir: 3.5, description: '3-4 reps in reserve', percentage: 82 },
  { rpe: 6, rir: 4, description: '4 reps in reserve', percentage: 80 },
  { rpe: 5, rir: 5, description: '5+ reps in reserve', percentage: 75 },
];

// ============================================================================
// SAMPLE PROGRAM TEMPLATES
// ============================================================================

export const PROGRAM_TEMPLATES: Partial<TrainingProgram>[] = [
  {
    id: 'beginner-strength',
    name: 'Beginner Strength Foundation',
    description: 'A 12-week program designed to build foundational strength using linear progression. Perfect for those new to resistance training.',
    goal: 'strength',
    experienceLevel: 'beginner',
    durationWeeks: 12,
    daysPerWeek: 3,
    sessionDuration: 60,
    tags: ['beginner', 'strength', 'full-body', 'linear-progression'],
  },
  {
    id: 'intermediate-hypertrophy',
    name: 'Intermediate Hypertrophy Program',
    description: 'An 8-week muscle-building program using undulating periodization. Designed for those with 1-3 years of training experience.',
    goal: 'hypertrophy',
    experienceLevel: 'intermediate',
    durationWeeks: 8,
    daysPerWeek: 5,
    sessionDuration: 75,
    tags: ['intermediate', 'hypertrophy', 'ppl', 'muscle-building'],
  },
  {
    id: 'advanced-powerbuilding',
    name: 'Advanced Powerbuilding Protocol',
    description: 'A 12-week program combining powerlifting and bodybuilding principles. For experienced lifters seeking both strength and size.',
    goal: 'powerbuilding',
    experienceLevel: 'advanced',
    durationWeeks: 12,
    daysPerWeek: 4,
    sessionDuration: 90,
    tags: ['advanced', 'powerbuilding', 'strength', 'hypertrophy'],
  },
  {
    id: 'fat-loss-preservation',
    name: 'Fat Loss with Muscle Preservation',
    description: 'An 8-week cutting program designed to maximize fat loss while preserving muscle mass. Includes nutrition periodization.',
    goal: 'fat-loss',
    experienceLevel: 'intermediate',
    durationWeeks: 8,
    daysPerWeek: 4,
    sessionDuration: 60,
    tags: ['fat-loss', 'cutting', 'muscle-preservation', 'nutrition'],
  },
  {
    id: 'competition-prep',
    name: 'Bodybuilding Competition Prep',
    description: 'A 16-week competition preparation program based on protocols used by elite bodybuilders. Includes peak week strategies.',
    goal: 'bodybuilding-prep',
    experienceLevel: 'elite',
    durationWeeks: 16,
    daysPerWeek: 6,
    sessionDuration: 90,
    tags: ['competition', 'bodybuilding', 'elite', 'peak-week'],
  },
];

// ============================================================================
// RECOVERY PROTOCOLS
// ============================================================================

export const RECOVERY_PROTOCOLS: RecoveryMetrics[] = [
  {
    sleepHoursTarget: 8,
    restDaysPerWeek: 2,
    activeRecoveryProtocol: [
      'Light walking 20-30 minutes',
      'Swimming or cycling at low intensity',
      'Yoga or stretching session',
    ],
    stressManagement: [
      'Meditation 10-15 minutes daily',
      'Deep breathing exercises',
      'Limit screen time before bed',
      'Nature walks',
    ],
    hydrationTarget: '1 gallon (3.8L) minimum daily',
    mobilityWork: [
      'Hip flexor stretches',
      'Thoracic spine mobility',
      'Shoulder dislocates',
      'Ankle mobility drills',
      'Foam rolling major muscle groups',
    ],
  },
];

// ============================================================================
// EXPORT DEFAULT
// ============================================================================

export default {
  VOLUME_LANDMARKS,
  PERIODIZATION_MODELS,
  TRAINING_SPLITS,
  NUTRITION_PHASES,
  ADAPTIVE_RULES,
  DELOAD_PROTOCOLS,
  RPE_RIR_TABLE,
  PROGRAM_TEMPLATES,
  RECOVERY_PROTOCOLS,
};
