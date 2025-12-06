/**
 * Intake Form Service
 * Handles all intake form operations including submissions, drafts, and processing
 */

import { supabase } from '@/integrations/supabase/client';

// ============================================================================
// TYPES
// ============================================================================

export interface PersonalInfo {
  full_name: string;
  preferred_name?: string;
  date_of_birth: string;
  age?: number;
  gender: 'male' | 'female' | 'non-binary' | 'prefer_not_to_say';
  email: string;
  phone: string;
  address?: string;
  city?: string;
  state?: string;
  zip?: string;
  emergency_contact?: {
    name: string;
    relationship: string;
    phone: string;
    alternate_phone?: string;
  };
}

export interface FitnessAssessment {
  current_weight_lbs?: number;
  goal_weight_lbs?: number;
  height_feet?: number;
  height_inches?: number;
  body_fat_percent?: number;
  goal_body_fat_percent?: number;
  bmi?: number;
  measurements?: {
    neck?: number;
    chest?: number;
    waist?: number;
    hips?: number;
    right_arm?: number;
    left_arm?: number;
    right_thigh?: number;
    left_thigh?: number;
    right_calf?: number;
    left_calf?: number;
  };
  vitals?: {
    resting_heart_rate?: number;
    blood_pressure_systolic?: number;
    blood_pressure_diastolic?: number;
  };
}

export interface HealthHistory {
  parq_responses?: {
    heart_condition: boolean;
    chest_pain_activity: boolean;
    chest_pain_rest: boolean;
    dizziness: boolean;
    bone_joint_problem: boolean;
    blood_pressure_meds: boolean;
    other_reason: boolean;
    explanation?: string;
  };
  medical_conditions?: string[];
  condition_details?: string;
  current_medications?: Array<{
    name: string;
    dosage: string;
    frequency: string;
    purpose: string;
  }>;
  allergies?: string;
  current_injuries?: string;
  past_injuries?: string;
  movements_to_avoid?: string;
  physician?: {
    name?: string;
    phone?: string;
    last_physical_date?: string;
  };
  physician_clearance?: boolean;
}

export interface TrainingExperience {
  fitness_level: 'beginner' | 'intermediate' | 'advanced' | 'elite';
  exercise_frequency: 'none' | '1-2' | '3-4' | '5-6' | 'daily';
  training_duration: string;
  training_background?: string;
  strength_benchmarks?: Record<string, number | string>;
  training_location: 'commercial_gym' | 'home_gym' | 'both' | 'outdoors';
  gym_name?: string;
  home_equipment?: string[];
}

export interface Goals {
  primary_goal: string;
  secondary_goals?: string[];
  specific_targets?: string;
  timeframe: string;
  motivation?: string;
  past_obstacles?: string;
  accountability_level: 'high' | 'medium' | 'low';
  coaching_style: 'strict' | 'balanced' | 'flexible';
}

export interface Nutrition {
  meals_per_day: string;
  current_diet: string;
  diet_other?: string;
  dietary_restrictions?: string[];
  food_allergies?: string[];
  foods_disliked?: string;
  typical_meals?: {
    breakfast?: string;
    lunch?: string;
    dinner?: string;
    snacks?: string;
  };
  hydration?: {
    daily_water_intake: string;
    alcohol_consumption: string;
    caffeine_intake: string;
    eating_out_frequency: string;
  };
  current_supplements?: string;
  open_to_supplements?: boolean;
  willing_to_meal_prep?: 'yes' | 'sometimes' | 'no';
}

export interface Lifestyle {
  occupation?: string;
  work_schedule?: string;
  occupational_activity: string;
  daily_steps?: number;
  sleep?: {
    average_hours: string;
    quality: string;
    bedtime?: string;
    wake_time?: string;
    issues?: string[];
  };
  stress_level: string;
  stress_sources?: string;
  recovery_methods?: string[];
  smoking_status: 'never' | 'former' | 'current';
  recreational_drugs?: boolean;
  travel_frequency: string;
}

export interface Schedule {
  preferred_training_days: string[];
  preferred_training_time: string;
  session_duration: string;
  communication?: {
    preferred_method: string;
    check_in_frequency: string;
    timezone?: string;
    preferred_start_date?: string;
  };
  upcoming_events?: string;
}

export interface Consent {
  informed_consent: boolean;
  liability_waiver: boolean;
  privacy_policy: boolean;
  photo_release_coaching?: boolean;
  photo_release_marketing?: boolean;
  medical_disclosure: boolean;
  program_agreement: boolean;
  signature?: string;
  signature_date?: string;
}

export interface IntakeFormData {
  personal_info: PersonalInfo;
  fitness_assessment: FitnessAssessment;
  health_history: HealthHistory;
  training_experience: TrainingExperience;
  goals: Goals;
  nutrition: Nutrition;
  lifestyle: Lifestyle;
  schedule: Schedule;
  consent: Consent;
}

export interface IntakeSubmission {
  id: string;
  user_id?: string;
  client_profile_id?: string;
  submission_date: string;
  form_version: string;
  submission_source: string;
  personal_info: PersonalInfo;
  fitness_assessment: FitnessAssessment;
  health_history: HealthHistory;
  training_experience: TrainingExperience;
  goals: Goals;
  nutrition: Nutrition;
  lifestyle: Lifestyle;
  schedule: Schedule;
  consent: Consent;
  status: string;
  assigned_coach_id?: string;
  assigned_tier?: string;
  coach_notes?: string;
  admin_notes?: string;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// INTAKE FORM SERVICE
// ============================================================================

export const intakeService = {
  /**
   * Submit a complete intake form
   */
  async submitIntakeForm(data: IntakeFormData): Promise<{ id: string; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();

      const { data: submission, error } = await supabase
        .from('intake_form_submissions')
        .insert({
          user_id: userData?.user?.id,
          personal_info: data.personal_info,
          fitness_assessment: data.fitness_assessment,
          health_history: data.health_history,
          training_experience: data.training_experience,
          goals: data.goals,
          nutrition: data.nutrition,
          lifestyle: data.lifestyle,
          schedule: data.schedule,
          consent: data.consent,
          status: 'submitted',
          form_version: '2.0',
          submission_source: 'web',
        })
        .select('id')
        .single();

      if (error) throw error;

      // Log activity
      if (userData?.user?.id) {
        await supabase.rpc('log_activity', {
          p_user_id: userData.user.id,
          p_action_type: 'intake_submitted',
          p_action_description: 'Client submitted intake form',
          p_target_type: 'intake_form_submissions',
          p_target_id: submission.id,
        });
      }

      return { id: submission.id };
    } catch (error) {
      console.error('Error submitting intake form:', error);
      return { id: '', error: error instanceof Error ? error.message : 'Failed to submit form' };
    }
  },

  /**
   * Save intake form draft (auto-save)
   */
  async saveDraft(data: Partial<IntakeFormData>, currentStep: number): Promise<{ success: boolean; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();
      const userId = userData?.user?.id;

      if (!userId) {
        // For non-logged-in users, use session storage
        const sessionId = this.getOrCreateSessionId();

        const { error } = await supabase
          .from('intake_form_drafts')
          .upsert({
            session_id: sessionId,
            form_data: data,
            current_step: currentStep,
            updated_at: new Date().toISOString(),
          }, {
            onConflict: 'session_id',
          });

        if (error) throw error;
      } else {
        const { error } = await supabase
          .from('intake_form_drafts')
          .upsert({
            user_id: userId,
            form_data: data,
            current_step: currentStep,
            updated_at: new Date().toISOString(),
          }, {
            onConflict: 'user_id',
          });

        if (error) throw error;
      }

      return { success: true };
    } catch (error) {
      console.error('Error saving draft:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to save draft' };
    }
  },

  /**
   * Load intake form draft
   */
  async loadDraft(): Promise<{ data?: Partial<IntakeFormData>; currentStep?: number; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();
      const userId = userData?.user?.id;

      let query;
      if (userId) {
        query = supabase
          .from('intake_form_drafts')
          .select('form_data, current_step')
          .eq('user_id', userId)
          .single();
      } else {
        const sessionId = this.getSessionId();
        if (!sessionId) return { data: undefined };

        query = supabase
          .from('intake_form_drafts')
          .select('form_data, current_step')
          .eq('session_id', sessionId)
          .single();
      }

      const { data, error } = await query;

      if (error && error.code !== 'PGRST116') throw error; // PGRST116 = no rows

      return {
        data: data?.form_data as Partial<IntakeFormData>,
        currentStep: data?.current_step,
      };
    } catch (error) {
      console.error('Error loading draft:', error);
      return { error: error instanceof Error ? error.message : 'Failed to load draft' };
    }
  },

  /**
   * Delete intake form draft
   */
  async deleteDraft(): Promise<{ success: boolean; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();
      const userId = userData?.user?.id;

      if (userId) {
        await supabase
          .from('intake_form_drafts')
          .delete()
          .eq('user_id', userId);
      } else {
        const sessionId = this.getSessionId();
        if (sessionId) {
          await supabase
            .from('intake_form_drafts')
            .delete()
            .eq('session_id', sessionId);
        }
      }

      return { success: true };
    } catch (error) {
      console.error('Error deleting draft:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to delete draft' };
    }
  },

  /**
   * Get submission by ID
   */
  async getSubmission(id: string): Promise<{ data?: IntakeSubmission; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('intake_form_submissions')
        .select('*')
        .eq('id', id)
        .single();

      if (error) throw error;

      return { data: data as IntakeSubmission };
    } catch (error) {
      console.error('Error getting submission:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get submission' };
    }
  },

  /**
   * Get all submissions (admin/coach)
   */
  async getAllSubmissions(status?: string): Promise<{ data?: IntakeSubmission[]; error?: string }> {
    try {
      let query = supabase
        .from('intake_form_submissions')
        .select('*')
        .order('submission_date', { ascending: false });

      if (status) {
        query = query.eq('status', status);
      }

      const { data, error } = await query;

      if (error) throw error;

      return { data: data as IntakeSubmission[] };
    } catch (error) {
      console.error('Error getting submissions:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get submissions' };
    }
  },

  /**
   * Update submission status (admin/coach)
   */
  async updateSubmissionStatus(
    id: string,
    status: string,
    notes?: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { data: userData } = await supabase.auth.getUser();

      const { error } = await supabase
        .from('intake_form_submissions')
        .update({
          status,
          coach_notes: notes,
          reviewed_at: new Date().toISOString(),
          reviewed_by: userData?.user?.id,
          updated_at: new Date().toISOString(),
        })
        .eq('id', id);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error updating submission:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update submission' };
    }
  },

  /**
   * Assign coach to submission
   */
  async assignCoach(
    submissionId: string,
    coachId: string,
    tier: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('intake_form_submissions')
        .update({
          assigned_coach_id: coachId,
          assigned_tier: tier,
          status: 'approved',
          updated_at: new Date().toISOString(),
        })
        .eq('id', submissionId);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error assigning coach:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to assign coach' };
    }
  },

  /**
   * Create client from approved submission
   */
  async createClientFromSubmission(
    submissionId: string,
    tier: string = 'core'
  ): Promise<{ clientId?: string; error?: string }> {
    try {
      const { data, error } = await supabase.rpc('create_client_from_intake', {
        p_submission_id: submissionId,
        p_tier: tier,
      });

      if (error) throw error;

      return { clientId: data };
    } catch (error) {
      console.error('Error creating client:', error);
      return { error: error instanceof Error ? error.message : 'Failed to create client' };
    }
  },

  // Session ID helpers for non-logged-in users
  getSessionId(): string | null {
    if (typeof window === 'undefined') return null;
    return sessionStorage.getItem('intake_session_id');
  },

  getOrCreateSessionId(): string {
    let sessionId = this.getSessionId();
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      sessionStorage.setItem('intake_session_id', sessionId);
    }
    return sessionId;
  },
};

export default intakeService;
