import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Search categories and their configurations
const SEARCH_CATEGORIES = {
  workouts: {
    table: 'workout_plans',
    fields: ['name', 'description', 'muscle_groups', 'equipment_needed'],
    weights: { name: 3, description: 2, muscle_groups: 2, equipment_needed: 1 }
  },
  exercises: {
    table: 'exercises',
    fields: ['name', 'description', 'muscle_groups', 'instructions'],
    weights: { name: 3, description: 2, muscle_groups: 2, instructions: 1 }
  },
  nutrition: {
    table: 'nutrition_plans',
    fields: ['name', 'description', 'goals', 'food_items'],
    weights: { name: 3, description: 2, goals: 2, food_items: 1 }
  },
  progress: {
    table: 'daily_tracking',
    fields: ['daily_notes', 'workout_notes'],
    weights: { daily_notes: 2, workout_notes: 2 }
  },
  clients: {
    table: 'client_profiles',
    fields: ['client_name', 'primary_goal', 'activity_level'],
    weights: { client_name: 3, primary_goal: 2, activity_level: 1 }
  }
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { query, categories, limit = 20, userId } = await req.json();

    if (!query || query.trim().length === 0) {
      throw new Error('Search query is required');
    }

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 2);
    const searchCategories = categories || Object.keys(SEARCH_CATEGORIES);
    
    console.log(`Searching for "${query}" in categories: ${searchCategories.join(', ')}`);

    const searchResults: any[] = [];

    // Search each category
    for (const category of searchCategories) {
      const config = SEARCH_CATEGORIES[category as keyof typeof SEARCH_CATEGORIES];
      if (!config) continue;

      try {
        let supabaseQuery = supabase.from(config.table).select('*');

        // Add user-specific filtering where applicable
        if (userId) {
          if (category === 'progress') {
            supabaseQuery = supabaseQuery.eq('client_id', userId);
          } else if (category === 'clients') {
            supabaseQuery = supabaseQuery.eq('auth_user_id', userId);
          }
        }

        const { data, error } = await supabaseQuery.limit(limit);

        if (error) {
          console.error(`Error searching ${category}:`, error);
          continue;
        }

        if (!data || data.length === 0) continue;

        // Score and filter results
        const scoredResults = data
          .map(item => {
            let totalScore = 0;
            const matchDetails: any = {};

            // Calculate relevance score
            for (const field of config.fields) {
              const fieldValue = item[field];
              if (!fieldValue) continue;

              const fieldText = Array.isArray(fieldValue) 
                ? fieldValue.join(' ').toLowerCase()
                : String(fieldValue).toLowerCase();

              let fieldScore = 0;
              const fieldMatches: string[] = [];

              // Check for exact matches
              if (fieldText.includes(query.toLowerCase())) {
                fieldScore += 10;
                fieldMatches.push('exact');
              }

              // Check for term matches
              for (const term of searchTerms) {
                if (fieldText.includes(term)) {
                  fieldScore += 3;
                  fieldMatches.push(term);
                }
              }

              // Apply field weight
              const weight = config.weights[field as keyof typeof config.weights] || 1;
              totalScore += fieldScore * weight;

              if (fieldMatches.length > 0) {
                matchDetails[field] = {
                  score: fieldScore,
                  matches: fieldMatches,
                  text: fieldText.substring(0, 100)
                };
              }
            }

            return {
              ...item,
              category,
              relevanceScore: totalScore,
              matchDetails,
              type: category.slice(0, -1) // Remove 's' from category name
            };
          })
          .filter(item => item.relevanceScore > 0)
          .sort((a, b) => b.relevanceScore - a.relevanceScore);

        searchResults.push(...scoredResults);

      } catch (categoryError) {
        console.error(`Error in category ${category}:`, categoryError);
        continue;
      }
    }

    // Sort all results by relevance and limit
    const finalResults = searchResults
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, limit);

    // Group results by category for better presentation
    const groupedResults = finalResults.reduce((groups, result) => {
      const category = result.category;
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(result);
      return groups;
    }, {} as Record<string, any[]>);

    console.log(`Search completed: ${finalResults.length} results found`);

    return new Response(
      JSON.stringify({
        query,
        totalResults: finalResults.length,
        results: finalResults,
        groupedResults,
        categories: Object.keys(groupedResults),
        searchTime: Date.now(),
        suggestions: generateSearchSuggestions(query, finalResults)
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );

  } catch (error) {
    console.error('Error in universal-search function:', error);
    return new Response(
      JSON.stringify({ 
        error: error.message,
        timestamp: new Date().toISOString()
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    );
  }
});

// Generate search suggestions based on results
function generateSearchSuggestions(query: string, results: any[]): string[] {
  const suggestions = new Set<string>();
  
  // Extract common terms from high-scoring results
  results.slice(0, 5).forEach(result => {
    Object.values(result.matchDetails || {}).forEach((match: any) => {
      if (match.matches) {
        match.matches.forEach((term: string) => {
          if (term !== 'exact' && term.length > 3 && !query.toLowerCase().includes(term)) {
            suggestions.add(term);
          }
        });
      }
    });
  });

  // Add category-based suggestions
  const categories = [...new Set(results.map(r => r.category))];
  categories.forEach(cat => {
    suggestions.add(`${query} in ${cat}`);
  });

  return Array.from(suggestions).slice(0, 5);
}