import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const logStep = (step: string, details?: Record<string, unknown>) => {
  const detailsStr = details ? ` - ${JSON.stringify(details)}` : '';
  console.log(`[OUTBOX-WORKER] ${step}${detailsStr}`);
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      { auth: { persistSession: false } }
    );

    const { data: items } = await supabase
      .from('outbox')
      .select('*')
      .eq('published', false)
      .eq('event_type', 'checkout_completed')
      .or('next_run_at.is.null,next_run_at.lte.' + new Date().toISOString())
      .order('created_at', { ascending: true })
      .limit(25);

    const list = items || [];
    let processed = 0;

    for (const item of list) {
      if ((item.attempts ?? 0) >= 5) {
        const traceId = item.trace_id || crypto.randomUUID();
        const wfUpsert = await supabase
          .from('workflows')
          .upsert({ trace_id: traceId, type: 'checkout', status: 'failed', updated_at: new Date().toISOString() }, { onConflict: 'trace_id' })
          .select('id')
          .single();
        if (!wfUpsert.error) {
          await supabase.from('workflow_steps').insert({ workflow_id: wfUpsert.data.id, step: 'checkout_completed', status: 'failed', error_code: 'max_attempts', error_message: item.last_error || 'max attempts reached', created_at: new Date().toISOString(), updated_at: new Date().toISOString() });
        }
        await supabase.from('outbox').update({ published: true }).eq('id', item.id);
        continue;
      }
      const traceId = item.trace_id || crypto.randomUUID();
      const wfUpsert = await supabase
        .from('workflows')
        .upsert({ trace_id: traceId, type: 'checkout', status: 'running', updated_at: new Date().toISOString() }, { onConflict: 'trace_id' })
        .select('id')
        .single();

      if (wfUpsert.error) {
        logStep('workflow_upsert_error', { error: wfUpsert.error.message });
        continue;
      }

      const workflowId = wfUpsert.data.id;

      const existingStep = await supabase
        .from('workflow_steps')
        .select('id')
        .eq('workflow_id', workflowId)
        .eq('step', 'checkout_completed')
        .limit(1)
        .maybeSingle();

      if (!existingStep.data) {
        const stepInsert = await supabase
          .from('workflow_steps')
          .insert({ workflow_id: workflowId, step: 'checkout_completed', status: 'succeeded', created_at: new Date().toISOString(), updated_at: new Date().toISOString() });
        if (stepInsert.error) {
          await supabase.from('workflows').update({ attempts: (wfUpsert.data.attempts ?? 0) + 1 }).eq('id', workflowId);
          logStep('workflow_step_error', { error: stepInsert.error.message });
          continue;
        }
      }

      if (stepInsert.error) {
        logStep('workflow_step_error', { error: stepInsert.error.message });
        continue;
      }

      const mark = await supabase
        .from('outbox')
        .update({ published: true })
        .eq('id', item.id);

      if (mark.error) {
        const delay = Math.min(900000, Math.pow(2, (item.attempts ?? 0)) * 60000);
        await supabase.from('outbox').update({ attempts: (item.attempts ?? 0) + 1, next_run_at: new Date(Date.now() + delay).toISOString(), last_error: mark.error.message }).eq('id', item.id);
        logStep('outbox_update_error', { error: mark.error.message });
        continue;
      }

      const existingEnt = await supabase.from('outbox').select('id').eq('event_type', 'apply_entitlements').eq('trace_id', traceId).limit(1).maybeSingle();
      if (!existingEnt.data) {
        await supabase.from('outbox').insert({ event_type: 'apply_entitlements', payload: item.payload, trace_id: traceId, published: false });
      }
      const existingBi = await supabase.from('outbox').select('id').eq('event_type', 'bi_publish').eq('trace_id', traceId).limit(1).maybeSingle();
      if (!existingBi.data) {
        await supabase.from('outbox').insert({ event_type: 'bi_publish', payload: item.payload, trace_id: traceId, published: false });
      }

      processed += 1;
    }

    logStep('processed', { count: processed });
    return new Response(JSON.stringify({ processed }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logStep('ERROR', { message });
    return new Response(JSON.stringify({ error: message }), { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 500 });
  }
});
