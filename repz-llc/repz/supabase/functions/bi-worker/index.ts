import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[BI-WORKER] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const { data: items } = await supabase
      .from('outbox').select('*')
      .eq('published', false).eq('event_type', 'bi_publish')
      .or('next_run_at.is.null,next_run_at.lte.' + new Date().toISOString())
      .order('created_at', { ascending: true }).limit(25);

    for (const item of items || []) {
      if ((item.attempts ?? 0) >= 5) {
        const traceId = item.trace_id || crypto.randomUUID();
        const wf = await supabase.from('workflows')
          .upsert({ trace_id: traceId, type: 'checkout', status: 'failed', updated_at: new Date().toISOString() }, { onConflict: 'trace_id' })
          .select('id').single();
        if (!wf.error) {
          await supabase.from('workflow_steps').insert({ workflow_id: wf.data.id, step: 'bi_publish', status: 'failed', error_code: 'max_attempts', error_message: item.last_error || 'max attempts reached', created_at: new Date().toISOString(), updated_at: new Date().toISOString() });
        }
        await supabase.from('outbox').update({ published: true }).eq('id', item.id);
        continue;
      }
      const traceId = item.trace_id || crypto.randomUUID();
      const wf = await supabase.from('workflows')
        .upsert({ trace_id: traceId, type: 'checkout', status: 'running', updated_at: new Date().toISOString() }, { onConflict: 'trace_id' })
        .select('id').single();
      if (wf.error) { logStep('workflow_upsert_error', { error: wf.error.message }); continue; }

      const existingStep = await supabase
        .from('workflow_steps')
        .select('id')
        .eq('workflow_id', wf.data.id)
        .eq('step', 'bi_publish')
        .limit(1)
        .maybeSingle();
      if (!existingStep.data) {
      const timeframes: ('7d'|'30d'|'90d'|'1y')[] = ['7d','30d','90d','1y'];
      for (const tf of timeframes) {
        const analytics = await supabase.functions.invoke('business-intelligence', {
          body: { timeframe: tf, metrics: ['revenue', 'users', 'conversions', 'churn', 'growth'] }
        });
        if (analytics.error) {
          const delayInvoke = Math.min(900000, Math.pow(2, (item.attempts ?? 0)) * 60000);
          await supabase.from('outbox').update({ attempts: (item.attempts ?? 0) + 1, next_run_at: new Date(Date.now() + delayInvoke).toISOString(), last_error: analytics.error.message }).eq('id', item.id);
          logStep('bi_invoke_error', { error: analytics.error.message, timeframe: tf });
          continue;
        }

        const snapshot = await supabase
          .from('analytics_snapshots')
          .upsert({ trace_id: traceId, timeframe: tf, payload: analytics.data }, { onConflict: 'trace_id,timeframe' });
        if (snapshot.error) {
          const delaySnap = Math.min(900000, Math.pow(2, (item.attempts ?? 0)) * 60000);
          await supabase.from('outbox').update({ attempts: (item.attempts ?? 0) + 1, next_run_at: new Date(Date.now() + delaySnap).toISOString(), last_error: snapshot.error.message }).eq('id', item.id);
          logStep('snapshot_insert_error', { error: snapshot.error.message, timeframe: tf });
          continue;
        }
      }

      const step = await supabase.from('workflow_steps')
        .insert({ workflow_id: wf.data.id, step: 'bi_publish', status: 'succeeded', created_at: new Date().toISOString(), updated_at: new Date().toISOString() });
      if (step.error) { logStep('workflow_step_error', { error: step.error.message }); continue; }
      }

      const mark = await supabase.from('outbox').update({ published: true }).eq('id', item.id);
      if (mark.error) {
        await supabase.from('outbox').update({ attempts: (item.attempts ?? 0) + 1, next_run_at: new Date(Date.now() + 60000).toISOString(), last_error: mark.error.message }).eq('id', item.id);
        logStep('outbox_update_error', { error: mark.error.message });
        continue;
      }
    }

    return new Response(JSON.stringify({ processed: (items || []).length }), { headers: { ...corsHeaders, "Content-Type": "application/json" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logStep('ERROR', { message });
    return new Response(JSON.stringify({ error: message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
  }
});
