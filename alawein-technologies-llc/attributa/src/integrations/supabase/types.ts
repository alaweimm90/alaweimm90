export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instanciate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.4"
  }
  public: {
    Tables: {
      artifacts: {
        Row: {
          analysis_results: Json | null
          artifact_type: string
          char_length: number
          content: string
          content_hash: string
          created_at: string
          id: string
          project_id: string
          source_id: string | null
          title: string | null
        }
        Insert: {
          analysis_results?: Json | null
          artifact_type: string
          char_length: number
          content: string
          content_hash: string
          created_at?: string
          id?: string
          project_id: string
          source_id?: string | null
          title?: string | null
        }
        Update: {
          analysis_results?: Json | null
          artifact_type?: string
          char_length?: number
          content?: string
          content_hash?: string
          created_at?: string
          id?: string
          project_id?: string
          source_id?: string | null
          title?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "artifacts_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "artifacts_source_id_fkey"
            columns: ["source_id"]
            isOneToOne: false
            referencedRelation: "sources"
            referencedColumns: ["id"]
          },
        ]
      }
      attributions: {
        Row: {
          artifact_id: string
          attribution_type: string
          confidence_level: string
          confidence_score: number
          created_at: string
          id: string
          rationale: string[] | null
          signals: Json | null
          similarity_score: number | null
          source_id: string
        }
        Insert: {
          artifact_id: string
          attribution_type: string
          confidence_level: string
          confidence_score: number
          created_at?: string
          id?: string
          rationale?: string[] | null
          signals?: Json | null
          similarity_score?: number | null
          source_id: string
        }
        Update: {
          artifact_id?: string
          attribution_type?: string
          confidence_level?: string
          confidence_score?: number
          created_at?: string
          id?: string
          rationale?: string[] | null
          signals?: Json | null
          similarity_score?: number | null
          source_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "attributions_artifact_id_fkey"
            columns: ["artifact_id"]
            isOneToOne: false
            referencedRelation: "artifacts"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "attributions_source_id_fkey"
            columns: ["source_id"]
            isOneToOne: false
            referencedRelation: "sources"
            referencedColumns: ["id"]
          },
        ]
      }
      citations: {
        Row: {
          citation_format: string | null
          created_at: string
          doi: string | null
          id: string
          parsed_citation: Json | null
          project_id: string
          raw_citation: string
          resolves: boolean | null
          suggestions: Json | null
          validation_status: string | null
        }
        Insert: {
          citation_format?: string | null
          created_at?: string
          doi?: string | null
          id?: string
          parsed_citation?: Json | null
          project_id: string
          raw_citation: string
          resolves?: boolean | null
          suggestions?: Json | null
          validation_status?: string | null
        }
        Update: {
          citation_format?: string | null
          created_at?: string
          doi?: string | null
          id?: string
          parsed_citation?: Json | null
          project_id?: string
          raw_citation?: string
          resolves?: boolean | null
          suggestions?: Json | null
          validation_status?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "citations_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      projects: {
        Row: {
          created_at: string
          description: string | null
          id: string
          name: string
          settings: Json | null
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          description?: string | null
          id?: string
          name: string
          settings?: Json | null
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          description?: string | null
          id?: string
          name?: string
          settings?: Json | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      reports: {
        Row: {
          completed_at: string | null
          created_at: string
          export_format: string | null
          id: string
          project_id: string
          report_type: string
          results: Json | null
          status: string
        }
        Insert: {
          completed_at?: string | null
          created_at?: string
          export_format?: string | null
          id?: string
          project_id: string
          report_type: string
          results?: Json | null
          status?: string
        }
        Update: {
          completed_at?: string | null
          created_at?: string
          export_format?: string | null
          id?: string
          project_id?: string
          report_type?: string
          results?: Json | null
          status?: string
        }
        Relationships: [
          {
            foreignKeyName: "reports_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      sources: {
        Row: {
          content_hash: string
          created_at: string
          id: string
          metadata: Json | null
          original_content: string | null
          project_id: string
          source_type: string
          title: string | null
        }
        Insert: {
          content_hash: string
          created_at?: string
          id?: string
          metadata?: Json | null
          original_content?: string | null
          project_id: string
          source_type: string
          title?: string | null
        }
        Update: {
          content_hash?: string
          created_at?: string
          id?: string
          metadata?: Json | null
          original_content?: string | null
          project_id?: string
          source_type?: string
          title?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "sources_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      user_api_keys: {
        Row: {
          anthropic_key: string | null
          created_at: string
          openai_key: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          anthropic_key?: string | null
          created_at?: string
          openai_key?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          anthropic_key?: string | null
          created_at?: string
          openai_key?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
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
    Enums: {},
  },
} as const
