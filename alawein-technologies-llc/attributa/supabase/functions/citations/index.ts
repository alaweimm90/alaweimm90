import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface CitationRequest {
  projectId: string;
  rawCitation: string;
  format?: 'APA' | 'MLA' | 'IEEE' | 'Chicago';
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
      const url = new URL(req.url);
      const projectId = url.searchParams.get('projectId');

      if (!projectId) {
        return new Response(
          JSON.stringify({ error: 'Project ID required' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      // Verify project access
      const { data: project, error: projectError } = await supabase
        .from('projects')
        .select('id')
        .eq('id', projectId)
        .eq('user_id', user.id)
        .single();

      if (projectError || !project) {
        return new Response(
          JSON.stringify({ error: 'Project not found or access denied' }),
          { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      // Get citations for project
      const { data: citations, error } = await supabase
        .from('citations')
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Citations fetch error:', error);
        return new Response(
          JSON.stringify({ error: 'Failed to fetch citations' }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ citations }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (req.method === 'POST') {
      const { projectId, rawCitation, format = 'APA' }: CitationRequest = await req.json();

      // Verify project access
      const { data: project, error: projectError } = await supabase
        .from('projects')
        .select('id')
        .eq('id', projectId)
        .eq('user_id', user.id)
        .single();

      if (projectError || !project) {
        return new Response(
          JSON.stringify({ error: 'Project not found or access denied' }),
          { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      // Parse and validate citation
      const parsedCitation = parseCitation(rawCitation);
      const validationResult = await validateCitation(parsedCitation);

      // Create citation record
      const { data: citation, error: citationError } = await supabase
        .from('citations')
        .insert({
          project_id: projectId,
          raw_citation: rawCitation,
          parsed_citation: parsedCitation,
          citation_format: format,
          doi: parsedCitation.doi,
          resolves: validationResult.resolves,
          validation_status: validationResult.status,
          suggestions: validationResult.suggestions
        })
        .select()
        .single();

      if (citationError) {
        console.error('Citation creation error:', citationError);
        return new Response(
          JSON.stringify({ error: 'Failed to create citation' }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ 
          citation,
          formatted: formatCitation(parsedCitation, format),
          validation: validationResult 
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Citation API error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});

interface ParsedCitation {
  raw?: string;
  type?: string;
  doi?: string;
  url?: string;
  authors?: string;
  year?: string;
  title?: string;
  journal?: string;
}

function parseCitation(rawCitation: string): ParsedCitation {
  // Simple citation parsing - can be enhanced with more sophisticated logic
  const citation: ParsedCitation = {
    raw: rawCitation,
    type: 'unknown'
  };

  // Try to extract DOI
  const doiMatch = rawCitation.match(/doi:?\s*(10\.\d+\/[^\s]+)|https?:\/\/doi\.org\/(10\.\d+\/[^\s]+)/i);
  if (doiMatch) {
    citation.doi = doiMatch[1] || doiMatch[2];
  }

  // Try to extract URL
  const urlMatch = rawCitation.match(/https?:\/\/[^\s]+/);
  if (urlMatch) {
    citation.url = urlMatch[0];
  }

  // Simple author extraction (names before year)
  const authorMatch = rawCitation.match(/^([^(]+)\s*\(/);
  if (authorMatch) {
    citation.authors = authorMatch[1].trim();
  }

  // Year extraction
  const yearMatch = rawCitation.match(/\((\d{4})\)/);
  if (yearMatch) {
    citation.year = yearMatch[1];
  }

  // Title extraction (text in quotes or after year)
  const titleMatch = rawCitation.match(/"([^"]+)"|'([^']+)'/) || 
                    rawCitation.match(/\d{4}\)\.\s*([^.]+)\./);
  if (titleMatch) {
    citation.title = titleMatch[1] || titleMatch[2] || titleMatch[3];
  }

  // Journal/venue extraction
  const journalMatch = rawCitation.match(/\.\s*([^,.]+),?\s*\d/);
  if (journalMatch) {
    citation.journal = journalMatch[1].trim();
  }

  return citation;
}

interface SimilarWork {
  title: string;
  doi: string;
  confidence: number;
  authors?: string;
  year?: string;
}

async function validateCitation(parsedCitation: ParsedCitation) {
  const result = {
    resolves: false,
    status: 'pending' as 'pending' | 'valid' | 'invalid' | 'suggestions',
    suggestions: [] as SimilarWork[]
  };

  try {
    // If we have a DOI, try to resolve it
    if (parsedCitation.doi) {
      // Mock DOI resolution - in production, this would call CrossRef API
      const doiValid = await mockResolveDoI(parsedCitation.doi);
      if (doiValid) {
        result.resolves = true;
        result.status = 'valid';
        return result;
      }
    }

    // If we have enough metadata, check for potential matches
    if (parsedCitation.title && parsedCitation.authors) {
      const suggestions = await findSimilarWorks(parsedCitation);
      if (suggestions.length > 0) {
        result.status = 'suggestions';
        result.suggestions = suggestions;
      } else {
        result.status = 'invalid';
      }
    } else {
      result.status = 'invalid';
    }

    return result;
  } catch (error) {
    console.error('Citation validation error:', error);
    result.status = 'invalid';
    return result;
  }
}

async function mockResolveDoI(doi: string): Promise<boolean> {
  // Mock DOI resolution - returns true for valid-looking DOIs
  return doi.startsWith('10.') && doi.includes('/');
}

async function findSimilarWorks(citation: ParsedCitation): Promise<SimilarWork[]> {
  // Mock similar works finder - would use CrossRef/PubMed APIs in production
  if (citation.title?.toLowerCase().includes('ai') || 
      citation.title?.toLowerCase().includes('attribution')) {
    return [
      {
        title: `Enhanced ${citation.title}`,
        doi: '10.1234/example.doi',
        confidence: 0.85,
        authors: citation.authors,
        year: citation.year
      }
    ];
  }
  return [];
}

function formatCitation(parsedCitation: ParsedCitation, format: string): string {
  const { authors = 'Unknown', year = 'n.d.', title = 'Untitled', journal = '', doi = '' } = parsedCitation;

  switch (format) {
    case 'APA':
      return `${authors} (${year}). ${title}. ${journal ? journal + '. ' : ''}${doi ? `https://doi.org/${doi}` : ''}`.trim();
    
    case 'MLA':
      return `${authors}. "${title}" ${journal ? journal + ', ' : ''}${year}${doi ? `, doi:${doi}` : ''}.`.trim();
    
    case 'IEEE':
      return `${authors}, "${title}," ${journal ? journal + ', ' : ''}${year}${doi ? `, doi: ${doi}` : ''}.`.trim();
    
    case 'Chicago':
      return `${authors}. "${title}" ${journal ? journal + ' ' : ''}(${year})${doi ? `. https://doi.org/${doi}` : ''}.`.trim();
    
    default:
      return parsedCitation.raw;
  }
}