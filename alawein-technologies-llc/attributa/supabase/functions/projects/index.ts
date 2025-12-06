import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface ProjectRequest {
  name: string;
  description?: string;
  settings?: Record<string, unknown>;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    );

    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (req.method === 'GET') {
      // Get all projects for user
      const { data: projects, error } = await supabase
        .from('projects')
        .select(`
          *,
          sources(count),
          artifacts(count),
          reports(count)
        `)
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Projects fetch error:', error);
        return new Response(
          JSON.stringify({ error: 'Failed to fetch projects' }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ projects }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (req.method === 'POST') {
      // Create new project
      const { name, description, settings }: ProjectRequest = await req.json();

      if (!name || name.trim().length === 0) {
        return new Response(
          JSON.stringify({ error: 'Project name is required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      const { data: project, error } = await supabase
        .from('projects')
        .insert({
          user_id: user.id,
          name: name.trim(),
          description: description?.trim() || null,
          settings: settings || {}
        })
        .select()
        .single();

      if (error) {
        console.error('Project creation error:', error);
        return new Response(
          JSON.stringify({ error: 'Failed to create project' }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ project }),
        { status: 201, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (req.method === 'PUT') {
      // Update project
      const url = new URL(req.url);
      const projectId = url.searchParams.get('id');

      if (!projectId) {
        return new Response(
          JSON.stringify({ error: 'Project ID required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      const { name, description, settings }: ProjectRequest = await req.json();

      const { data: project, error } = await supabase
        .from('projects')
        .update({
          name: name?.trim(),
          description: description?.trim() || null,
          settings: settings || {}
        })
        .eq('id', projectId)
        .eq('user_id', user.id)
        .select()
        .single();

      if (error) {
        console.error('Project update error:', error);
        return new Response(
          JSON.stringify({ error: 'Failed to update project' }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ project }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (req.method === 'DELETE') {
      // Delete project
      const url = new URL(req.url);
      const projectId = url.searchParams.get('id');

      if (!projectId) {
        return new Response(
          JSON.stringify({ error: 'Project ID required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      const { error } = await supabase
        .from('projects')
        .delete()
        .eq('id', projectId)
        .eq('user_id', user.id);

      if (error) {
        console.error('Project deletion error:', error);
        return new Response(
          JSON.stringify({ error: 'Failed to delete project' }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ message: 'Project deleted successfully' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Projects API error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});