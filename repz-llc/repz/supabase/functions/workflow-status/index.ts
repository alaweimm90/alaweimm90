import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const { trace_id } = await req.json();
    if (!trace_id) return new Response(JSON.stringify({ error: "trace_id required" }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 400 });

    const wf = await supabase
      .from('workflows')
      .select('id, trace_id, type, status, updated_at')
      .eq('trace_id', trace_id)
      .order('updated_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (!wf.data) return new Response(JSON.stringify({ data: null }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });

    const steps = await supabase
      .from('workflow_steps')
      .select('step, status, error_code, error_message, created_at, updated_at')
      .eq('workflow_id', wf.data.id)
      .order('created_at', { ascending: true });

    return new Response(JSON.stringify({ data: { workflow: wf.data, steps: steps.data || [] } }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return new Response(JSON.stringify({ error: message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
  }
});

