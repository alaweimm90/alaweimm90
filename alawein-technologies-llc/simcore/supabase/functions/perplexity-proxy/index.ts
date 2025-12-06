import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const PERPLEXITY_API_KEY = Deno.env.get("PERPLEXITY_API_KEY");
    if (!PERPLEXITY_API_KEY) {
      return new Response(
        JSON.stringify({ error: "PERPLEXITY_API_KEY is not set" }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const raw = await req.text();
    if (!raw) {
      return new Response(
        JSON.stringify({ error: "Missing JSON body" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    let body: any;
    try {
      body = JSON.parse(raw);
    } catch {
      return new Response(
        JSON.stringify({ error: "Invalid JSON body" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const {
      model = "llama-3.1-sonar-small-128k-online",
      messages = [],
      temperature = 0.2,
      top_p = 0.9,
      max_tokens = 1000,
      return_images = false,
      return_related_questions = false,
      frequency_penalty = 1,
      presence_penalty = 0,
      search_domain_filter,
      search_recency_filter = "month",
    } = body || {};

    if (!Array.isArray(messages) || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: "messages array is required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${PERPLEXITY_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages,
        temperature,
        top_p,
        max_tokens,
        return_images,
        return_related_questions,
        search_domain_filter,
        search_recency_filter,
        frequency_penalty,
        presence_penalty,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("Perplexity error:", data);
      return new Response(JSON.stringify({ error: data.error || data }), {
        status: response.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("perplexity-proxy unexpected error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : String(error) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
