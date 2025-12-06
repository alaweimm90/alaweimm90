/**
 * User Service
 * Handles user management, profiles, roles, and authentication helpers
 */

import { supabase } from '@/integrations/supabase/client';

// ============================================================================
// TYPES
// ============================================================================

export type UserRole = 'admin' | 'coach' | 'client' | 'member';

export interface UserProfile {
  id: string;
  email: string;
  subscription_tier: string;
  subscription_status: string;
  created_at: string;
  updated_at: string;
}

export interface ClientProfile {
  id: string;
  auth_user_id: string;
  coach_id?: string;
  client_name: string;
  subscription_tier: string;
  onboarding_completed: boolean;
  created_at: string;
  updated_at: string;
}

export interface CoachProfile {
  id: string;
  auth_user_id: string;
  coach_name: string;
  specializations?: string[];
  bio?: string;
  certifications?: string[];
  created_at: string;
  updated_at: string;
}

export interface User {
  id: string;
  email: string;
  role: UserRole;
  profile?: UserProfile;
  clientProfile?: ClientProfile;
  coachProfile?: CoachProfile;
}

// ============================================================================
// USER SERVICE
// ============================================================================

export const userService = {
  /**
   * Get current user with all profiles
   */
  async getCurrentUser(): Promise<{ user?: User; error?: string }> {
    try {
      const { data: authData, error: authError } = await supabase.auth.getUser();
      if (authError) throw authError;
      if (!authData?.user) return { error: 'Not authenticated' };

      const userId = authData.user.id;

      // Get profile
      const { data: profile } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single();

      // Get role
      const { data: roleData } = await supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      // Get client profile if exists
      const { data: clientProfile } = await supabase
        .from('client_profiles')
        .select('*')
        .eq('auth_user_id', userId)
        .single();

      // Get coach profile if exists
      const { data: coachProfile } = await supabase
        .from('coach_profiles')
        .select('*')
        .eq('auth_user_id', userId)
        .single();

      const user: User = {
        id: userId,
        email: authData.user.email || '',
        role: (roleData?.role as UserRole) || 'member',
        profile: profile as UserProfile,
        clientProfile: clientProfile as ClientProfile,
        coachProfile: coachProfile as CoachProfile,
      };

      return { user };
    } catch (error) {
      console.error('Error getting current user:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get user' };
    }
  },

  /**
   * Check if current user has a specific role
   */
  async hasRole(role: UserRole): Promise<boolean> {
    try {
      const { data: authData } = await supabase.auth.getUser();
      if (!authData?.user) return false;

      const { data } = await supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', authData.user.id)
        .eq('role', role)
        .single();

      return !!data;
    } catch (error) {
      return false;
    }
  },

  /**
   * Check if current user is admin
   */
  async isAdmin(): Promise<boolean> {
    return this.hasRole('admin');
  },

  /**
   * Check if current user is coach
   */
  async isCoach(): Promise<boolean> {
    return this.hasRole('coach');
  },

  /**
   * Get user's primary role
   */
  async getPrimaryRole(): Promise<UserRole | null> {
    try {
      const { data: authData } = await supabase.auth.getUser();
      if (!authData?.user) return null;

      const { data } = await supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', authData.user.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      return (data?.role as UserRole) || null;
    } catch (error) {
      return null;
    }
  },

  /**
   * Update user profile
   */
  async updateProfile(updates: Partial<UserProfile>): Promise<{ success: boolean; error?: string }> {
    try {
      const { data: authData } = await supabase.auth.getUser();
      if (!authData?.user) return { success: false, error: 'Not authenticated' };

      const { error } = await supabase
        .from('profiles')
        .update({
          ...updates,
          updated_at: new Date().toISOString(),
        })
        .eq('id', authData.user.id);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error updating profile:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update' };
    }
  },

  /**
   * Update client profile
   */
  async updateClientProfile(updates: Partial<ClientProfile>): Promise<{ success: boolean; error?: string }> {
    try {
      const { data: authData } = await supabase.auth.getUser();
      if (!authData?.user) return { success: false, error: 'Not authenticated' };

      const { error } = await supabase
        .from('client_profiles')
        .update({
          ...updates,
          updated_at: new Date().toISOString(),
        })
        .eq('auth_user_id', authData.user.id);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error updating client profile:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update' };
    }
  },

  /**
   * Get all users (admin only)
   */
  async getAllUsers(): Promise<{ data?: User[]; error?: string }> {
    try {
      const { data: profiles, error } = await supabase
        .from('profiles')
        .select(`
          *,
          user_roles (role),
          client_profiles (*),
          coach_profiles (*)
        `)
        .order('created_at', { ascending: false });

      if (error) throw error;

      const users: User[] = profiles?.map(p => ({
        id: p.id,
        email: p.email,
        role: p.user_roles?.[0]?.role || 'member',
        profile: p as UserProfile,
        clientProfile: p.client_profiles?.[0] as ClientProfile,
        coachProfile: p.coach_profiles?.[0] as CoachProfile,
      })) || [];

      return { data: users };
    } catch (error) {
      console.error('Error getting users:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get users' };
    }
  },

  /**
   * Get all clients (coach/admin)
   */
  async getAllClients(): Promise<{ data?: ClientProfile[]; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('client_profiles')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      return { data: data as ClientProfile[] };
    } catch (error) {
      console.error('Error getting clients:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get clients' };
    }
  },

  /**
   * Get all coaches (admin)
   */
  async getAllCoaches(): Promise<{ data?: CoachProfile[]; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('coach_profiles')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      return { data: data as CoachProfile[] };
    } catch (error) {
      console.error('Error getting coaches:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get coaches' };
    }
  },

  /**
   * Assign role to user (admin only)
   */
  async assignRole(userId: string, role: UserRole): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('user_roles')
        .upsert({
          user_id: userId,
          role,
          created_at: new Date().toISOString(),
        }, {
          onConflict: 'user_id,role',
        });

      if (error) throw error;

      // Log activity
      const { data: authData } = await supabase.auth.getUser();
      if (authData?.user) {
        await supabase.rpc('log_activity', {
          p_user_id: authData.user.id,
          p_action_type: 'role_assigned',
          p_action_description: `Assigned role ${role} to user`,
          p_target_type: 'user_roles',
          p_target_id: userId,
          p_metadata: { role },
        });
      }

      return { success: true };
    } catch (error) {
      console.error('Error assigning role:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to assign role' };
    }
  },

  /**
   * Remove role from user (admin only)
   */
  async removeRole(userId: string, role: UserRole): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('user_roles')
        .delete()
        .eq('user_id', userId)
        .eq('role', role);

      if (error) throw error;

      return { success: true };
    } catch (error) {
      console.error('Error removing role:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to remove role' };
    }
  },

  /**
   * Update user tier (admin only)
   */
  async updateUserTier(userId: string, tier: string): Promise<{ success: boolean; error?: string }> {
    try {
      // Update profile
      const { error: profileError } = await supabase
        .from('profiles')
        .update({
          subscription_tier: tier,
          subscription_status: 'active',
          updated_at: new Date().toISOString(),
        })
        .eq('id', userId);

      if (profileError) throw profileError;

      // Update client profile if exists
      await supabase
        .from('client_profiles')
        .update({
          subscription_tier: tier,
          updated_at: new Date().toISOString(),
        })
        .eq('auth_user_id', userId);

      return { success: true };
    } catch (error) {
      console.error('Error updating tier:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to update tier' };
    }
  },

  /**
   * Create coach profile (admin only)
   */
  async createCoachProfile(
    userId: string,
    coachName: string,
    specializations?: string[]
  ): Promise<{ id?: string; error?: string }> {
    try {
      // Assign coach role
      await this.assignRole(userId, 'coach');

      // Create coach profile
      const { data, error } = await supabase
        .from('coach_profiles')
        .insert({
          auth_user_id: userId,
          coach_name: coachName,
          specializations,
        })
        .select('id')
        .single();

      if (error) throw error;

      return { id: data.id };
    } catch (error) {
      console.error('Error creating coach:', error);
      return { error: error instanceof Error ? error.message : 'Failed to create coach' };
    }
  },

  /**
   * Assign client to coach
   */
  async assignClientToCoach(
    clientId: string,
    coachId: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { data: authData } = await supabase.auth.getUser();

      const { error } = await supabase
        .from('coach_client_assignments')
        .upsert({
          coach_id: coachId,
          client_id: clientId,
          assigned_by: authData?.user?.id,
          status: 'active',
          is_primary: true,
        }, {
          onConflict: 'coach_id,client_id',
        });

      if (error) throw error;

      // Update client profile
      await supabase
        .from('client_profiles')
        .update({
          coach_id: coachId,
          updated_at: new Date().toISOString(),
        })
        .eq('id', clientId);

      return { success: true };
    } catch (error) {
      console.error('Error assigning client:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Failed to assign client' };
    }
  },

  /**
   * Get coach's clients
   */
  async getCoachClients(coachId: string): Promise<{ data?: ClientProfile[]; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('coach_client_assignments')
        .select(`
          client_id,
          client_profiles (*)
        `)
        .eq('coach_id', coachId)
        .eq('status', 'active');

      if (error) throw error;

      const clients = data?.map(d => d.client_profiles).filter(Boolean) as ClientProfile[];

      return { data: clients };
    } catch (error) {
      console.error('Error getting coach clients:', error);
      return { error: error instanceof Error ? error.message : 'Failed to get clients' };
    }
  },
};

export default userService;
