/**
 * REPZ Database Types
 * Auto-generated from Supabase schema - Updated for clean backend
 */

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  public: {
    Tables: {
      // ========================================
      // PROFILES
      // ========================================
      profiles: {
        Row: {
          id: string
          email: string | null
          full_name: string | null
          avatar_url: string | null
          phone: string | null
          date_of_birth: string | null
          gender: 'male' | 'female' | 'other' | 'prefer_not_to_say' | null
          timezone: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          email?: string | null
          full_name?: string | null
          avatar_url?: string | null
          phone?: string | null
          date_of_birth?: string | null
          gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say' | null
          timezone?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          email?: string | null
          full_name?: string | null
          avatar_url?: string | null
          phone?: string | null
          date_of_birth?: string | null
          gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say' | null
          timezone?: string
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }

      // ========================================
      // USER ROLES
      // ========================================
      user_roles: {
        Row: {
          id: string
          user_id: string
          role: 'admin' | 'coach' | 'client'
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          role: 'admin' | 'coach' | 'client'
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          role?: 'admin' | 'coach' | 'client'
          created_at?: string
        }
        Relationships: []
      }

      // ========================================
      // SUBSCRIPTION TIERS
      // ========================================
      subscription_tiers: {
        Row: {
          id: string
          name: string
          display_name: string
          description: string | null
          price_cents: number
          billing_period: 'monthly' | 'quarterly' | 'annual'
          stripe_price_id: string | null
          features: Json
          is_active: boolean
          is_popular: boolean
          is_limited: boolean
          max_clients: number | null
          sort_order: number
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          name: string
          display_name: string
          description?: string | null
          price_cents: number
          billing_period?: 'monthly' | 'quarterly' | 'annual'
          stripe_price_id?: string | null
          features?: Json
          is_active?: boolean
          is_popular?: boolean
          is_limited?: boolean
          max_clients?: number | null
          sort_order?: number
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          name?: string
          display_name?: string
          description?: string | null
          price_cents?: number
          billing_period?: 'monthly' | 'quarterly' | 'annual'
          stripe_price_id?: string | null
          features?: Json
          is_active?: boolean
          is_popular?: boolean
          is_limited?: boolean
          max_clients?: number | null
          sort_order?: number
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }

      // ========================================
      // COACH PROFILES
      // ========================================
      coach_profiles: {
        Row: {
          id: string
          auth_user_id: string
          coach_name: string
          bio: string | null
          specializations: string[] | null
          certifications: string[] | null
          avatar_url: string | null
          is_active: boolean
          max_clients: number
          current_client_count: number
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          auth_user_id: string
          coach_name: string
          bio?: string | null
          specializations?: string[] | null
          certifications?: string[] | null
          avatar_url?: string | null
          is_active?: boolean
          max_clients?: number
          current_client_count?: number
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          auth_user_id?: string
          coach_name?: string
          bio?: string | null
          specializations?: string[] | null
          certifications?: string[] | null
          avatar_url?: string | null
          is_active?: boolean
          max_clients?: number
          current_client_count?: number
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }

      // ========================================
      // CLIENT PROFILES
      // ========================================
      client_profiles: {
        Row: {
          id: string
          auth_user_id: string
          client_name: string
          subscription_tier_id: string | null
          subscription_tier: string | null
          assigned_coach_id: string | null
          onboarding_completed: boolean
          onboarding_completed_at: string | null
          status: 'pending' | 'active' | 'paused' | 'cancelled'
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          auth_user_id: string
          client_name: string
          subscription_tier_id?: string | null
          subscription_tier?: string | null
          assigned_coach_id?: string | null
          onboarding_completed?: boolean
          onboarding_completed_at?: string | null
          status?: 'pending' | 'active' | 'paused' | 'cancelled'
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          auth_user_id?: string
          client_name?: string
          subscription_tier_id?: string | null
          subscription_tier?: string | null
          assigned_coach_id?: string | null
          onboarding_completed?: boolean
          onboarding_completed_at?: string | null
          status?: 'pending' | 'active' | 'paused' | 'cancelled'
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "client_profiles_subscription_tier_id_fkey"
            columns: ["subscription_tier_id"]
            referencedRelation: "subscription_tiers"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "client_profiles_assigned_coach_id_fkey"
            columns: ["assigned_coach_id"]
            referencedRelation: "coach_profiles"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // SUBSCRIPTIONS
      // ========================================
      subscriptions: {
        Row: {
          id: string
          user_id: string
          client_profile_id: string | null
          tier_id: string | null
          stripe_customer_id: string | null
          stripe_subscription_id: string | null
          stripe_price_id: string | null
          status: 'trialing' | 'active' | 'past_due' | 'canceled' | 'unpaid' | 'paused'
          current_period_start: string | null
          current_period_end: string | null
          cancel_at_period_end: boolean
          canceled_at: string | null
          trial_start: string | null
          trial_end: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          client_profile_id?: string | null
          tier_id?: string | null
          stripe_customer_id?: string | null
          stripe_subscription_id?: string | null
          stripe_price_id?: string | null
          status?: 'trialing' | 'active' | 'past_due' | 'canceled' | 'unpaid' | 'paused'
          current_period_start?: string | null
          current_period_end?: string | null
          cancel_at_period_end?: boolean
          canceled_at?: string | null
          trial_start?: string | null
          trial_end?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          client_profile_id?: string | null
          tier_id?: string | null
          stripe_customer_id?: string | null
          stripe_subscription_id?: string | null
          stripe_price_id?: string | null
          status?: 'trialing' | 'active' | 'past_due' | 'canceled' | 'unpaid' | 'paused'
          current_period_start?: string | null
          current_period_end?: string | null
          cancel_at_period_end?: boolean
          canceled_at?: string | null
          trial_start?: string | null
          trial_end?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "subscriptions_tier_id_fkey"
            columns: ["tier_id"]
            referencedRelation: "subscription_tiers"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "subscriptions_client_profile_id_fkey"
            columns: ["client_profile_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // INTAKE FORM SUBMISSIONS
      // ========================================
      intake_form_submissions: {
        Row: {
          id: string
          user_id: string | null
          client_profile_id: string | null
          personal_info: Json
          fitness_assessment: Json
          health_history: Json
          training_experience: Json
          goals: Json
          nutrition: Json
          lifestyle: Json
          schedule: Json
          consent: Json
          status: 'draft' | 'submitted' | 'under_review' | 'approved' | 'needs_info' | 'rejected'
          assigned_coach_id: string | null
          assigned_tier_id: string | null
          coach_notes: string | null
          admin_notes: string | null
          submitted_at: string
          reviewed_at: string | null
          reviewed_by: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id?: string | null
          client_profile_id?: string | null
          personal_info?: Json
          fitness_assessment?: Json
          health_history?: Json
          training_experience?: Json
          goals?: Json
          nutrition?: Json
          lifestyle?: Json
          schedule?: Json
          consent?: Json
          status?: 'draft' | 'submitted' | 'under_review' | 'approved' | 'needs_info' | 'rejected'
          assigned_coach_id?: string | null
          assigned_tier_id?: string | null
          coach_notes?: string | null
          admin_notes?: string | null
          submitted_at?: string
          reviewed_at?: string | null
          reviewed_by?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string | null
          client_profile_id?: string | null
          personal_info?: Json
          fitness_assessment?: Json
          health_history?: Json
          training_experience?: Json
          goals?: Json
          nutrition?: Json
          lifestyle?: Json
          schedule?: Json
          consent?: Json
          status?: 'draft' | 'submitted' | 'under_review' | 'approved' | 'needs_info' | 'rejected'
          assigned_coach_id?: string | null
          assigned_tier_id?: string | null
          coach_notes?: string | null
          admin_notes?: string | null
          submitted_at?: string
          reviewed_at?: string | null
          reviewed_by?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "intake_form_submissions_client_profile_id_fkey"
            columns: ["client_profile_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "intake_form_submissions_assigned_coach_id_fkey"
            columns: ["assigned_coach_id"]
            referencedRelation: "coach_profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "intake_form_submissions_assigned_tier_id_fkey"
            columns: ["assigned_tier_id"]
            referencedRelation: "subscription_tiers"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // INTAKE FORM DRAFTS
      // ========================================
      intake_form_drafts: {
        Row: {
          id: string
          user_id: string
          session_id: string | null
          form_data: Json
          current_step: number
          expires_at: string
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          session_id?: string | null
          form_data?: Json
          current_step?: number
          expires_at?: string
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          session_id?: string | null
          form_data?: Json
          current_step?: number
          expires_at?: string
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }

      // ========================================
      // CLIENT ONBOARDING
      // ========================================
      client_onboarding: {
        Row: {
          id: string
          client_id: string
          intake_submission_id: string | null
          intake_completed: boolean
          intake_completed_at: string | null
          payment_completed: boolean
          payment_completed_at: string | null
          coach_assigned: boolean
          coach_assigned_at: string | null
          welcome_sent: boolean
          welcome_sent_at: string | null
          consultation_scheduled: boolean
          consultation_date: string | null
          program_created: boolean
          program_created_at: string | null
          status: 'pending' | 'in_progress' | 'completed' | 'paused'
          progress_percent: number
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          client_id: string
          intake_submission_id?: string | null
          intake_completed?: boolean
          intake_completed_at?: string | null
          payment_completed?: boolean
          payment_completed_at?: string | null
          coach_assigned?: boolean
          coach_assigned_at?: string | null
          welcome_sent?: boolean
          welcome_sent_at?: string | null
          consultation_scheduled?: boolean
          consultation_date?: string | null
          program_created?: boolean
          program_created_at?: string | null
          status?: 'pending' | 'in_progress' | 'completed' | 'paused'
          progress_percent?: number
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          client_id?: string
          intake_submission_id?: string | null
          intake_completed?: boolean
          intake_completed_at?: string | null
          payment_completed?: boolean
          payment_completed_at?: string | null
          coach_assigned?: boolean
          coach_assigned_at?: string | null
          welcome_sent?: boolean
          welcome_sent_at?: string | null
          consultation_scheduled?: boolean
          consultation_date?: string | null
          program_created?: boolean
          program_created_at?: string | null
          status?: 'pending' | 'in_progress' | 'completed' | 'paused'
          progress_percent?: number
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "client_onboarding_client_id_fkey"
            columns: ["client_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "client_onboarding_intake_submission_id_fkey"
            columns: ["intake_submission_id"]
            referencedRelation: "intake_form_submissions"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // COACH-CLIENT ASSIGNMENTS
      // ========================================
      coach_client_assignments: {
        Row: {
          id: string
          coach_id: string
          client_id: string
          is_primary: boolean
          status: 'active' | 'paused' | 'ended'
          assigned_at: string
          assigned_by: string | null
          ended_at: string | null
          notes: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          coach_id: string
          client_id: string
          is_primary?: boolean
          status?: 'active' | 'paused' | 'ended'
          assigned_at?: string
          assigned_by?: string | null
          ended_at?: string | null
          notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          coach_id?: string
          client_id?: string
          is_primary?: boolean
          status?: 'active' | 'paused' | 'ended'
          assigned_at?: string
          assigned_by?: string | null
          ended_at?: string | null
          notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "coach_client_assignments_coach_id_fkey"
            columns: ["coach_id"]
            referencedRelation: "coach_profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "coach_client_assignments_client_id_fkey"
            columns: ["client_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // PROGRAMS
      // ========================================
      programs: {
        Row: {
          id: string
          client_id: string
          coach_id: string | null
          name: string
          description: string | null
          program_type: 'strength' | 'hypertrophy' | 'endurance' | 'weight_loss' | 'general_fitness' | 'sport_specific' | 'rehabilitation' | null
          status: 'draft' | 'active' | 'completed' | 'archived'
          start_date: string | null
          end_date: string | null
          program_data: Json
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          client_id: string
          coach_id?: string | null
          name: string
          description?: string | null
          program_type?: 'strength' | 'hypertrophy' | 'endurance' | 'weight_loss' | 'general_fitness' | 'sport_specific' | 'rehabilitation' | null
          status?: 'draft' | 'active' | 'completed' | 'archived'
          start_date?: string | null
          end_date?: string | null
          program_data?: Json
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          client_id?: string
          coach_id?: string | null
          name?: string
          description?: string | null
          program_type?: 'strength' | 'hypertrophy' | 'endurance' | 'weight_loss' | 'general_fitness' | 'sport_specific' | 'rehabilitation' | null
          status?: 'draft' | 'active' | 'completed' | 'archived'
          start_date?: string | null
          end_date?: string | null
          program_data?: Json
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "programs_client_id_fkey"
            columns: ["client_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "programs_coach_id_fkey"
            columns: ["coach_id"]
            referencedRelation: "coach_profiles"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // WEEKLY CHECKINS
      // ========================================
      weekly_checkins: {
        Row: {
          id: string
          client_id: string
          coach_id: string | null
          week_start: string
          weight_lbs: number | null
          body_fat_percent: number | null
          sleep_quality: number | null
          energy_level: number | null
          stress_level: number | null
          adherence_training: number | null
          adherence_nutrition: number | null
          wins: string | null
          challenges: string | null
          questions: string | null
          coach_feedback: string | null
          status: 'pending' | 'submitted' | 'reviewed'
          submitted_at: string | null
          reviewed_at: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          client_id: string
          coach_id?: string | null
          week_start: string
          weight_lbs?: number | null
          body_fat_percent?: number | null
          sleep_quality?: number | null
          energy_level?: number | null
          stress_level?: number | null
          adherence_training?: number | null
          adherence_nutrition?: number | null
          wins?: string | null
          challenges?: string | null
          questions?: string | null
          coach_feedback?: string | null
          status?: 'pending' | 'submitted' | 'reviewed'
          submitted_at?: string | null
          reviewed_at?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          client_id?: string
          coach_id?: string | null
          week_start?: string
          weight_lbs?: number | null
          body_fat_percent?: number | null
          sleep_quality?: number | null
          energy_level?: number | null
          stress_level?: number | null
          adherence_training?: number | null
          adherence_nutrition?: number | null
          wins?: string | null
          challenges?: string | null
          questions?: string | null
          coach_feedback?: string | null
          status?: 'pending' | 'submitted' | 'reviewed'
          submitted_at?: string | null
          reviewed_at?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "weekly_checkins_client_id_fkey"
            columns: ["client_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "weekly_checkins_coach_id_fkey"
            columns: ["coach_id"]
            referencedRelation: "coach_profiles"
            referencedColumns: ["id"]
          }
        ]
      }

      // ========================================
      // MESSAGES
      // ========================================
      messages: {
        Row: {
          id: string
          sender_id: string | null
          recipient_id: string | null
          thread_id: string | null
          content: string
          message_type: 'text' | 'image' | 'file' | 'system'
          is_read: boolean
          read_at: string | null
          created_at: string
        }
        Insert: {
          id?: string
          sender_id?: string | null
          recipient_id?: string | null
          thread_id?: string | null
          content: string
          message_type?: 'text' | 'image' | 'file' | 'system'
          is_read?: boolean
          read_at?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          sender_id?: string | null
          recipient_id?: string | null
          thread_id?: string | null
          content?: string
          message_type?: 'text' | 'image' | 'file' | 'system'
          is_read?: boolean
          read_at?: string | null
          created_at?: string
        }
        Relationships: []
      }

      // ========================================
      // ACTIVITY LOG
      // ========================================
      activity_log: {
        Row: {
          id: string
          user_id: string | null
          action: string
          entity_type: string | null
          entity_id: string | null
          metadata: Json
          ip_address: string | null
          user_agent: string | null
          created_at: string
        }
        Insert: {
          id?: string
          user_id?: string | null
          action: string
          entity_type?: string | null
          entity_id?: string | null
          metadata?: Json
          ip_address?: string | null
          user_agent?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string | null
          action?: string
          entity_type?: string | null
          entity_id?: string | null
          metadata?: Json
          ip_address?: string | null
          user_agent?: string | null
          created_at?: string
        }
        Relationships: []
      }

      // ========================================
      // PROGRESS PHOTOS
      // ========================================
      progress_photos: {
        Row: {
          id: string
          client_id: string
          photo_url: string
          photo_type: 'front' | 'side' | 'back' | 'other' | null
          taken_at: string
          notes: string | null
          is_private: boolean
          created_at: string
        }
        Insert: {
          id?: string
          client_id: string
          photo_url: string
          photo_type?: 'front' | 'side' | 'back' | 'other' | null
          taken_at?: string
          notes?: string | null
          is_private?: boolean
          created_at?: string
        }
        Update: {
          id?: string
          client_id?: string
          photo_url?: string
          photo_type?: 'front' | 'side' | 'back' | 'other' | null
          taken_at?: string
          notes?: string | null
          is_private?: boolean
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "progress_photos_client_id_fkey"
            columns: ["client_id"]
            referencedRelation: "client_profiles"
            referencedColumns: ["id"]
          }
        ]
      }
    }
    Views: Record<string, never>
    Functions: {
      has_role: {
        Args: { check_role: string }
        Returns: boolean
      }
      get_user_role: {
        Args: Record<string, never>
        Returns: string
      }
    }
    Enums: Record<string, never>
    CompositeTypes: Record<string, never>
  }
}

// ========================================
// HELPER TYPES
// ========================================

export type Tables<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Row']
export type Insertable<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Insert']
export type Updatable<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Update']

// Convenience aliases
export type Profile = Tables<'profiles'>
export type UserRole = Tables<'user_roles'>
export type SubscriptionTier = Tables<'subscription_tiers'>
export type CoachProfile = Tables<'coach_profiles'>
export type ClientProfile = Tables<'client_profiles'>
export type Subscription = Tables<'subscriptions'>
export type IntakeFormSubmission = Tables<'intake_form_submissions'>
export type IntakeFormDraft = Tables<'intake_form_drafts'>
export type ClientOnboarding = Tables<'client_onboarding'>
export type CoachClientAssignment = Tables<'coach_client_assignments'>
export type Program = Tables<'programs'>
export type WeeklyCheckin = Tables<'weekly_checkins'>
export type Message = Tables<'messages'>
export type ActivityLog = Tables<'activity_log'>
export type ProgressPhoto = Tables<'progress_photos'>
