# REPZ Coach Platform - AI Model Configuration Guide

## Current AI Models in Use

Your platform currently uses:

- **gpt-4.1-2025-04-14** (form analysis, workout recommendations, nutrition AI, predictive
  analytics)
- **gpt-4o** (video analysis)
- **eleven_turbo_v2** (ElevenLabs TTS)

## Available OpenAI Premium Models

### Latest GPT-4o Series (Recommended for Most Tasks)

- **gpt-4o** ([good for] general-purpose AI tasks, fast responses, multimodal input)
- **gpt-4o-mini** ([good for] lightweight tasks, cost-effective, faster inference)
- **gpt-4o-2024-08-06** ([good for] stable version with consistent performance)
- **gpt-4o-2024-05-13** ([good for] previous stable release with proven reliability)

### GPT-4 Turbo Series (High-Performance)

- **gpt-4-turbo** ([good for] complex reasoning, large context windows, advanced analysis)
- **gpt-4-turbo-preview** ([good for] latest features, experimental capabilities)
- **gpt-4-1106-preview** ([good for] extended context up to 128k tokens)
- **gpt-4-0125-preview** ([good for] improved function calling, JSON mode)
- **gpt-4-turbo-2024-04-09** ([good for] stable turbo version with vision capabilities)

### GPT-4 Base Models (Original Series)

- **gpt-4** ([good for] reliable performance, established capabilities)
- **gpt-4-0613** ([good for] stable baseline, function calling)
- **gpt-4-32k** ([good for] longer conversations, document analysis)
- **gpt-4-32k-0613** ([good for] extended context with stable performance)

### GPT-3.5 Turbo Series (Cost-Effective)

- **gpt-3.5-turbo** ([good for] budget-friendly, good performance/cost ratio)
- **gpt-3.5-turbo-16k** ([good for] longer contexts at lower cost)
- **gpt-3.5-turbo-1106** ([good for] improved JSON handling)
- **gpt-3.5-turbo-0125** ([good for] latest 3.5 turbo with better reasoning)

### Specialized Vision Models

- **gpt-4-vision-preview** ([good for] image analysis, visual form assessment)
- **gpt-4o-vision** ([good for] advanced computer vision, workout form analysis)
- **gpt-4-1106-vision-preview** ([good for] vision with extended context)

### Enterprise & Custom Models

- **gpt-4-32k-0314** ([good for] enterprise applications requiring long contexts)
- **gpt-4-0314** ([good for] legacy enterprise deployments)
- **gpt-3.5-turbo-instruct** ([good for] instruction-following tasks, legacy compatibility)

## Model Selection Guide

### For Fitness Coaching Tasks:

- **Form Analysis**: `gpt-4o` or `gpt-4-vision-preview`
- **Workout Recommendations**: `gpt-4-turbo` or `gpt-4o`
- **Nutrition AI**: `gpt-4-turbo` (complex reasoning needed)
- **Video Analysis**: `gpt-4o` or `gpt-4-vision-preview`
- **Predictive Analytics**: `gpt-4-turbo` (large context windows)

### Cost vs Performance Balance:

- **High Performance**: GPT-4 Turbo series
- **Balanced**: GPT-4o series
- **Cost-Effective**: GPT-3.5 Turbo series

## Implementation Example

```typescript
// For form analysis (vision + reasoning)
model: 'gpt-4o', // Good for multimodal tasks

// For complex workout planning (reasoning + context)
model: 'gpt-4-turbo', // Good for complex analysis

// For quick responses (speed + cost)
model: 'gpt-4o-mini', // Good for lightweight tasks
```

## Migration Strategy

1. **Test Environment**: Start with `gpt-4o-mini` for cost monitoring
2. **Staging**: Use `gpt-4o` for feature validation
3. **Production**: Scale to `gpt-4-turbo` for premium features

## Cost Comparison (Approximate per 1K tokens)

- **gpt-4o-mini**: $0.15 input, $0.60 output
- **gpt-4o**: $2.50 input, $10.00 output
- **gpt-4-turbo**: $10.00 input, $30.00 output

## Next Steps

Would you like me to help you:

1. Update specific Edge Functions to use different models?
2. Add model selection configuration?
3. Implement A/B testing between models?
4. Set up cost monitoring and optimization?
