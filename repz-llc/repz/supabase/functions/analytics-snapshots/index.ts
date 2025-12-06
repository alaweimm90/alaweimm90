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

    const { trace_id, timeframe, action } = await req.json();
    if (!trace_id) return new Response(JSON.stringify({ error: "trace_id required" }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 400 });

    if (action === 'refresh') {
      const tf = timeframe || '30d';
      const res = await supabase.functions.invoke('business-intelligence', {
        body: { timeframe: tf, metrics: ['revenue','users','conversions','churn','growth'], segmentation: {} }
      });
      if (res.error) return new Response(JSON.stringify({ error: res.error.message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
      const ins = await supabase
        .from('analytics_snapshots')
        .upsert({ trace_id, timeframe: tf, payload: res.data }, { onConflict: 'trace_id,timeframe' })
        .select('*')
        .single();
      if (ins.error) return new Response(JSON.stringify({ error: ins.error.message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
      return new Response(JSON.stringify({ data: ins.data }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });
    }

    if (action === 'list') {
      let query = supabase
        .from('analytics_snapshots')
        .select('id, created_at, timeframe')
        .eq('trace_id', trace_id);

      if (timeframe) query = query.eq('timeframe', timeframe);

      const { data, error } = await query
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) return new Response(JSON.stringify({ error: error.message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
      return new Response(JSON.stringify({ data }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });
    }

    let query = supabase
      .from('analytics_snapshots')
      .select('*')
      .eq('trace_id', trace_id);

    if (timeframe) {
      query = query.eq('timeframe', timeframe);
    }

    const { data } = await query
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (!data) return new Response(JSON.stringify({ data: null }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });

    return new Response(JSON.stringify({ data }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return new Response(JSON.stringify({ error: message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
  }
});
