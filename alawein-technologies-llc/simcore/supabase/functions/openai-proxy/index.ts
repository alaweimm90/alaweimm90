import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { z } from "https://deno.land/x/zod@v3.23.8/mod.ts";

const ALLOWED_ORIGINS = [/^https?:\/\/localhost:\d+$/, /^https?:\/\/.*malawein\.com$/];

function isAllowedOrigin(origin: string | null): boolean {
  if (!origin) return false;
  return ALLOWED_ORIGINS.some((re) => re.test(origin));
}

function corsHeaders(origin: string | null) {
  return {
    "Access-Control-Allow-Origin": isAllowedOrigin(origin) ? origin : "",
    "Vary": "Origin",
    "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  } as const;
}

const BodySchema = z.object({
  model: z.string().default("gpt-4o-mini"),
  messages: z.array(z.any()).min(1, "messages array is required"),
  temperature: z.number().min(0).max(2).default(0.3),
  max_tokens: z.number().int().positive().max(8192).default(1024),
});

serve(async (req) => {
  const origin = req.headers.get("origin");
  // CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders(origin) });
  }

  try {
    const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
    if (!OPENAI_API_KEY) {
      return new Response(
        JSON.stringify({ error: "OPENAI_API_KEY is not set" }),
        { status: 500, headers: { ...corsHeaders(origin), "Content-Type": "application/json" } }
      );
    }

    const raw = await req.text();
    if (!raw) {
      return new Response(
        JSON.stringify({ error: "Missing JSON body" }),
        { status: 400, headers: { ...corsHeaders(origin), "Content-Type": "application/json" } }
      );
    }

    let body: any;
    try {
      body = JSON.parse(raw);
    } catch {
      return new Response(
        JSON.stringify({ error: "Invalid JSON body" }),
        { status: 400, headers: { ...corsHeaders(origin), "Content-Type": "application/json" } }
      );
    }

    const parsed = BodySchema.safeParse(body);
    if (!parsed.success) {
      return new Response(
        JSON.stringify({ error: parsed.error.flatten() }),
        { status: 400, headers: { ...corsHeaders(origin), "Content-Type": "application/json" } }
      );
    }
    const { model, messages, temperature, max_tokens } = parsed.data;

    if (!Array.isArray(messages) || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: "messages array is required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, messages, temperature, max_tokens }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("OpenAI error:", data);
      return new Response(JSON.stringify({ error: data.error || data }), {
        status: response.status,
        headers: { ...corsHeaders(origin), "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders(origin), "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("openai-proxy unexpected error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : String(error) }),
      { status: 500, headers: { ...corsHeaders(null), "Content-Type": "application/json" } }
    );
  }
});
