/**
 * Onboarding Service
 * Handles client onboarding workflow and progress tracking
 */

import { supabase } from '@/integrations/supabase/client';

// ============================================================================
// TYPES
// ============================================================================

export type OnboardingStatus =
  | 'pending'
  | 'intake_submitted'
  | 'payment_pending'
  | 'coach_assignment'
  | 'program_creation'
  | 'active'
  | 'paused'
  | 'cancelled';

export interface OnboardingProgress {
  id: string;
  client_id: string;
  intake_submission_id?: string;
  intake_completed: boolean;
  intake_completed_at?: string;
  payment_completed: boolean;
  payment_completed_at?: string;
  stripe_subscription_id?: string;
  coach_assigned: boolean;
  coach_assigned_at?: string;
  welcome_email_sent: boolean;
  welcome_email_sent_at?: string;
  initial_consultation_scheduled: boolean;
  initial_consultation_date?: string;
  program_created: boolean;
  program_created_at?: string;
  first_checkin_completed: boolean;
  first_checkin_at?: string;
  onboarding_status: OnboardingStatus;
  progress_percent: number;
  created_at: string;
  updated_at: string;
}

export interface OnboardingStep {
  id: string;
  name: string;
  description: string;
  completed: boolean;
  completedAt?: string;
  order: number;
}

// ============================================================================
// ONBOARDING SERVICE
// ============================================================================

export const onboardingService = {
  /**
   * Get current user's onboarding progress
   */
  async getMyOnboarding(): Promise<{ data?: OnboardingProgress; error?: string }> {
    try {
      const { data: authData } = await supabase.auth.getUser();
      if (!authData?.user) return { error: 'Not authenticated' };

      // Get client profile first
      const { data: clientProfile } = await supabase
        .from('client_profiles')
        .select('id')
        .eq('auth_user_id', authData.user.id)
        .single();

      if (!clientProfile) return { error: 'No client profile found' };

      const { data, error } = await supabase
        .from('client_onboarding')
        .select('*')
        .eq('client_id', clientProfile.id)
        .single();

      if (error && error.code !== 'PGRST116') throw error;

      return { data: data as OnboardingProgress };
    } catch (error) {
      console.error('Error getting onboarding:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get onboarding' };
    }
  },

  /**
   * Get onboarding progress for a specific client
   */
  async getClientOnboarding(clientId: string): Promise<{ data?: OnboardingProgress; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('client_onboarding')
        .select('*')
        .eq('client_id', clientId)
        .single();

      if (error && error.code !== 'PGRST116') throw error;

      return { data: data as OnboardingProgress };
    } catch (error) {
      console.error('Error getting client onboarding:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get onboarding' };
    }
  },

  /**
   * Create onboarding record for new client
   */
  async createOnboarding(
    clientId: string,
    intakeSubmissionId?: string
  ): Promise<{ id?: string; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('client_onboarding')
        .insert({
          client_id: clientId,
          intake_submission_id: intakeSubmissionId,
          intake_completed: !!intakeSubmissionId,
          intake_completed_at: intakeSubmissionId ? new Date().toISOString() : null,
          onboarding_status: intakeSubmissionId ? 'intake_submitted' : 'pending',
        })
        .select('id')
        .single();

      if (error) throw error;

      return { id: data.id };
    } catch (error) {
      console.error('Error creating onboarding:', error);
      return { error: error instanceof Error ? error.message : 'Failed to create onboarding' };
    }
  },

  /**
   * Update onboarding step
   */
  async updateStep(
    clientId: string,
    step: keyof Omit<OnboardingProgress, 'id' | 'client_id' | 'created_at' | 'updated_at' | 'progress_percent' | 'onboarding_status'>,
    value: boolean | string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const updates: Record<string, unknown> = {
        [step]: value,
      };

      // Add timestamp for boolean steps
      if (typeof value === 'boolean' && value === true) {
        const timestampKey = `${step}_at`;
        if (step !== 'intake_completed' && step !== 'payment_completed' && step !== 'coach_assigned' &&
            step !== 'welcome_email_sent' && step !== 'program_created' && step !== 'first_checkin_completed') {
          // Skip for non-timestamped fields
        } else {
          updates[timestampKey] = new Date().toISOString();
        }
      }

      const { error } = await supabase
        .from('client_onboarding')
        .update(updates)
        .eq('client_id', clientId);

      if (error) throw error;

      // Update status based on progress
      await this.updateOnboardingStatus(clientId);

      return { success: true };
    } catch (error) {
      console.error('Error updating step:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update step' };
    }
  },

  /**
   * Mark intake as completed
   */
  async markIntakeCompleted(
    clientId: string,
    submissionId: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('client_onboarding')
        .update({
          intake_completed: true,
          intake_completed_at: new Date().toISOString(),
          intake_submission_id: submissionId,
          onboarding_status: 'intake_submitted',
        })
        .eq('client_id', clientId);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error marking intake completed:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update' };
    }
  },

  /**
   * Mark payment as completed
   */
  async markPaymentCompleted(
    clientId: string,
    stripeSubscriptionId: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('client_onboarding')
        .update({
          payment_completed: true,
          payment_completed_at: new Date().toISOString(),
          stripe_subscription_id: stripeSubscriptionId,
          onboarding_status: 'coach_assignment',
        })
        .eq('client_id', clientId);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error marking payment completed:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update' };
    }
  },

  /**
   * Mark coach as assigned
   */
  async markCoachAssigned(clientId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('client_onboarding')
        .update({
          coach_assigned: true,
          coach_assigned_at: new Date().toISOString(),
          onboarding_status: 'program_creation',
        })
        .eq('client_id', clientId);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error marking coach assigned:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update' };
    }
  },

  /**
   * Mark program as created and activate client
   */
  async markProgramCreated(clientId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('client_onboarding')
        .update({
          program_created: true,
          program_created_at: new Date().toISOString(),
          onboarding_status: 'active',
        })
        .eq('client_id', clientId);

      if (error) throw error;

      // Update client profile
      await supabase
        .from('client_profiles')
        .update({
          onboarding_completed: true,
          updated_at: new Date().toISOString(),
        })
        .eq('id', clientId);

      return { success: true };
    } catch (error) {
      console.error('Error marking program created:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update' };
    }
  },

  /**
   * Update onboarding status based on progress
   */
  async updateOnboardingStatus(clientId: string): Promise<void> {
    try {
      const { data } = await supabase
        .from('client_onboarding')
        .select('*')
        .eq('client_id', clientId)
        .single();

      if (!data) return;

      let newStatus: OnboardingStatus = 'pending';

      if (data.first_checkin_completed) {
        newStatus = 'active';
      } else if (data.program_created) {
        newStatus = 'active';
      } else if (data.coach_assigned) {
        newStatus = 'program_creation';
      } else if (data.payment_completed) {
        newStatus = 'coach_assignment';
      } else if (data.intake_completed) {
        newStatus = 'payment_pending';
      }

      await supabase
        .from('client_onboarding')
        .update({ onboarding_status: newStatus })
        .eq('client_id', clientId);
    } catch (error) {
      console.error('Error updating status:', error);
    }
  },

  /**
   * Get onboarding steps as structured list
   */
  getOnboardingSteps(progress: OnboardingProgress): OnboardingStep[] {
    return [
      {
        id: 'intake',
        name: 'Complete Intake Form',
        description: 'Fill out your comprehensive fitness assessment',
        completed: progress.intake_completed,
        completedAt: progress.intake_completed_at,
        order: 1,
      },
      {
        id: 'payment',
        name: 'Choose Your Plan',
        description: 'Select and subscribe to your coaching tier',
        completed: progress.payment_completed,
        completedAt: progress.payment_completed_at,
        order: 2,
      },
      {
        id: 'coach',
        name: 'Coach Assignment',
        description: 'Get matched with your personal coach',
        completed: progress.coach_assigned,
        completedAt: progress.coach_assigned_at,
        order: 3,
      },
      {
        id: 'welcome',
        name: 'Welcome Package',
        description: 'Receive your welcome email and resources',
        completed: progress.welcome_email_sent,
        completedAt: progress.welcome_email_sent_at,
        order: 4,
      },
      {
        id: 'consultation',
        name: 'Initial Consultation',
        description: 'Schedule your first call with your coach',
        completed: progress.initial_consultation_scheduled,
        completedAt: progress.initial_consultation_date,
        order: 5,
      },
      {
        id: 'program',
        name: 'Program Creation',
        description: 'Your personalized program is being created',
        completed: progress.program_created,
        completedAt: progress.program_created_at,
        order: 6,
      },
      {
        id: 'checkin',
        name: 'First Check-in',
        description: 'Complete your first weekly check-in',
        completed: progress.first_checkin_completed,
        completedAt: progress.first_checkin_at,
        order: 7,
      },
    ];
  },

  /**
   * Get all onboarding records (admin/coach)
   */
  async getAllOnboarding(status?: OnboardingStatus): Promise<{ data?: OnboardingProgress[]; error?: string }> {
    try {
      let query = supabase
        .from('client_onboarding')
        .select('*')
        .order('created_at', { ascending: false });

      if (status) {
        query = query.eq('onboarding_status', status);
      }

      const { data, error } = await query;

      if (error) throw error;

      return { data: data as OnboardingProgress[] };
    } catch (error) {
      console.error('Error getting all onboarding:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get onboarding' };
    }
  },

  /**
   * Get onboarding stats (admin)
   */
  async getOnboardingStats(): Promise<{
    data?: {
      total: number;
      byStatus: Record<OnboardingStatus, number>;
      averageProgress: number;
      completedThisMonth: number;
    };
    error?: string;
  }> {
    try {
      const { data, error } = await supabase
        .from('client_onboarding')
        .select('onboarding_status, progress_percent, created_at');

      if (error) throw error;

      const now = new Date();
      const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);

      const stats = {
        total: data?.length || 0,
        byStatus: {
          pending: 0,
          intake_submitted: 0,
          payment_pending: 0,
          coach_assignment: 0,
          program_creation: 0,
          active: 0,
          paused: 0,
          cancelled: 0,
        } as Record<OnboardingStatus, number>,
        averageProgress: 0,
        completedThisMonth: 0,
      };

      let totalProgress = 0;
      data?.forEach(record => {
        stats.byStatus[record.onboarding_status as OnboardingStatus]++;
        totalProgress += record.progress_percent || 0;

        if (record.onboarding_status === 'active' && new Date(record.created_at) >= monthStart) {
          stats.completedThisMonth++;
        }
      });

      stats.averageProgress = data?.length ? Math.round(totalProgress / data.length) : 0;

      return { data: stats };
    } catch (error) {
      console.error('Error getting stats:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get stats' };
    }
  },
};

export default onboardingService;
