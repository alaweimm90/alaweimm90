export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      non_portal_clients: {
        Row: {
          id: string
          email: string
          full_name: string
          phone: string | null
          intake_data: Json
          payment_type: Database["public"]["Enums"]["subscription_tier"]
          payment_status: Database["public"]["Enums"]["payment_status"]
          stripe_customer_id: string | null
          stripe_session_id: string | null
          stripe_subscription_id: string | null
          plan_generated: boolean
          plan_sent_at: string | null
          plan_pdf_url: string | null
          coach_notes: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          email: string
          full_name: string
          phone?: string | null
          intake_data?: Json
          payment_type?: Database["public"]["Enums"]["subscription_tier"]
          payment_status?: Database["public"]["Enums"]["payment_status"]
          stripe_customer_id?: string | null
          stripe_session_id?: string | null
          stripe_subscription_id?: string | null
          plan_generated?: boolean
          plan_sent_at?: string | null
          plan_pdf_url?: string | null
          coach_notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          email?: string
          full_name?: string
          phone?: string | null
          intake_data?: Json
          payment_type?: Database["public"]["Enums"]["subscription_tier"]
          payment_status?: Database["public"]["Enums"]["payment_status"]
          stripe_customer_id?: string | null
          stripe_session_id?: string | null
          stripe_subscription_id?: string | null
          plan_generated?: boolean
          plan_sent_at?: string | null
          plan_pdf_url?: string | null
          coach_notes?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }
      content: {
        Row: {
          content: string
          content_type: Database["public"]["Enums"]["content_type"]
          created_at: string
          id: string
          is_published: boolean
          published_at: string | null
          required_tier: Database["public"]["Enums"]["subscription_tier"]
          slug: string
          title: string
          updated_at: string
        }
        Insert: {
          content: string
          content_type: Database["public"]["Enums"]["content_type"]
          created_at?: string
          id?: string
          is_published?: boolean
          published_at?: string | null
          required_tier?: Database["public"]["Enums"]["subscription_tier"]
          slug: string
          title: string
          updated_at?: string
        }
        Update: {
          content?: string
          content_type?: Database["public"]["Enums"]["content_type"]
          created_at?: string
          id?: string
          is_published?: boolean
          published_at?: string | null
          required_tier?: Database["public"]["Enums"]["subscription_tier"]
          slug?: string
          title?: string
          updated_at?: string
        }
        Relationships: []
      }
      profiles: {
        Row: {
          created_at: string
          email: string
          id: string
          last_login: string | null
          stripe_customer_id: string | null
          subscription_status: Database["public"]["Enums"]["subscription_status"]
          subscription_tier: Database["public"]["Enums"]["subscription_tier"]
          updated_at: string
        }
        Insert: {
          created_at?: string
          email: string
          id: string
          last_login?: string | null
          stripe_customer_id?: string | null
          subscription_status?: Database["public"]["Enums"]["subscription_status"]
          subscription_tier?: Database["public"]["Enums"]["subscription_tier"]
          updated_at?: string
        }
        Update: {
          created_at?: string
          email?: string
          id?: string
          last_login?: string | null
          stripe_customer_id?: string | null
          subscription_status?: Database["public"]["Enums"]["subscription_status"]
          subscription_tier?: Database["public"]["Enums"]["subscription_tier"]
          updated_at?: string
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      can_access_tier: {
        Args: {
          _required_tier: Database["public"]["Enums"]["subscription_tier"]
          _user_tier: Database["public"]["Enums"]["subscription_tier"]
        }
        Returns: boolean
      }
      get_user_tier: {
        Args: { _user_id: string }
        Returns: Database["public"]["Enums"]["subscription_tier"]
      }
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
    }
    Enums: {
      app_role: "member" | "admin"
      content_type: "article" | "video" | "guide" | "protocol"
      payment_status: "pending" | "completed" | "failed" | "refunded"
      subscription_status: "active" | "canceled" | "paused" | "expired"
      subscription_tier: "core" | "adaptive" | "performance" | "longevity"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["member", "admin"],
      content_type: ["article", "video", "guide", "protocol"],
      payment_status: ["pending", "completed", "failed", "refunded"],
      subscription_status: ["active", "canceled", "paused", "expired"],
      subscription_tier: ["core", "adaptive", "performance", "longevity"],
    },
  },
} as const
