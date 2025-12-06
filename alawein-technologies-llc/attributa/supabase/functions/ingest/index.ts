import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface IngestRequest {
  projectId: string;
  source: "paste" | "file" | "url" | "github";
  content?: string;
  files?: { name: string; content: string }[];
  githubUrl?: string;
  title?: string;
  metadata?: Record<string, unknown>;
}

serve(async (req) => {
  // Handle CORS preflight requests
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

    // Get the user from the request
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    if (userError || !user) {
      console.error('Auth error:', userError);
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const { projectId, source, content, files, githubUrl, title, metadata }: IngestRequest = await req.json();

    console.log('Ingesting content:', { projectId, source, title });

    // Verify project ownership
    const { data: project, error: projectError } = await supabase
      .from('projects')
      .select('id')
      .eq('id', projectId)
      .eq('user_id', user.id)
      .single();

    if (projectError || !project) {
      console.error('Project access error:', projectError);
      return new Response(
        JSON.stringify({ error: 'Project not found or access denied' }),
        { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Process content based on source type
    let processedContent = '';
    const sourceMetadata = metadata || {};

    switch (source) {
      case 'paste':
        processedContent = content || '';
        break;
      case 'file':
        if (files && files.length > 0) {
          processedContent = files.map(f => `File: ${f.name}\n\n${f.content}`).join('\n\n---\n\n');
          sourceMetadata.files = files.map(f => ({ name: f.name, size: f.content.length }));
        }
        break;
      case 'url':
        // For now, store the URL in metadata - could be enhanced to fetch content
        sourceMetadata.url = content;
        processedContent = content || '';
        break;
      case 'github':
        sourceMetadata.githubUrl = githubUrl;
        processedContent = content || '';
        break;
    }

    // Generate content hash
    const contentHash = await generateHash(processedContent);

    // Check if source already exists
    const { data: existingSource } = await supabase
      .from('sources')
      .select('id')
      .eq('project_id', projectId)
      .eq('content_hash', contentHash)
      .maybeSingle();

    if (existingSource) {
      return new Response(
        JSON.stringify({
          sourceId: existingSource.id,
          message: 'Source already exists',
          duplicate: true
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Create new source
    const { data: newSource, error: sourceError } = await supabase
      .from('sources')
      .insert({
        project_id: projectId,
        source_type: source,
        title: title || `${source} source`,
        content_hash: contentHash,
        original_content: processedContent,
        metadata: sourceMetadata
      })
      .select()
      .single();

    if (sourceError) {
      console.error('Source creation error:', sourceError);
      return new Response(
        JSON.stringify({ error: 'Failed to create source' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Segment the content into artifacts
    const segments = await segmentContent(processedContent);
    const artifacts = [];

    for (const segment of segments) {
      const segmentHash = await generateHash(segment.content);
      
      const { data: artifact, error: artifactError } = await supabase
        .from('artifacts')
        .insert({
          project_id: projectId,
          source_id: newSource.id,
          artifact_type: segment.type,
          title: segment.title,
          content: segment.content,
          content_hash: segmentHash,
          char_length: segment.content.length
        })
        .select()
        .single();

      if (!artifactError && artifact) {
        artifacts.push(artifact);
      }
    }

    console.log(`Created source with ${artifacts.length} artifacts`);

    return new Response(
      JSON.stringify({
        sourceId: newSource.id,
        artifactCount: artifacts.length,
        message: 'Content ingested successfully'
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Ingest error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});

async function generateHash(content: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

interface ContentSegment {
  type: 'prose' | 'code' | 'latex' | 'mixed';
  title: string;
  content: string;
}

async function segmentContent(content: string): Promise<ContentSegment[]> {
  // Simple segmentation logic - can be enhanced later
  const lines = content.split('\n');
  const segments: ContentSegment[] = [];
  let currentSegment = '';
  let currentType: ContentSegment['type'] = 'prose';
  let segmentCount = 1;

  for (const line of lines) {
    // Detect code blocks
    if (line.trim().startsWith('```') || line.trim().startsWith('~~~')) {
      if (currentSegment.trim()) {
        segments.push({
          type: currentType,
          title: `Segment ${segmentCount++}`,
          content: currentSegment.trim()
        });
      }
      currentType = currentType === 'code' ? 'prose' : 'code';
      currentSegment = '';
      continue;
    }

    // Detect LaTeX
    if (line.includes('\\documentclass') || line.includes('\\begin{') || line.includes('\\end{')) {
      if (currentType !== 'latex') {
        if (currentSegment.trim()) {
          segments.push({
            type: currentType,
            title: `Segment ${segmentCount++}`,
            content: currentSegment.trim()
          });
        }
        currentType = 'latex';
        currentSegment = '';
      }
    }

    currentSegment += line + '\n';

    // Create segments at paragraph breaks for prose
    if (currentType === 'prose' && line.trim() === '' && currentSegment.trim().length > 500) {
      segments.push({
        type: currentType,
        title: `Segment ${segmentCount++}`,
        content: currentSegment.trim()
      });
      currentSegment = '';
    }
  }

  // Add final segment
  if (currentSegment.trim()) {
    segments.push({
      type: currentType,
      title: `Segment ${segmentCount}`,
      content: currentSegment.trim()
    });
  }

  return segments.length > 0 ? segments : [{
    type: 'prose',
    title: 'Full Content',
    content: content.trim()
  }];
}