/**
 * System prompts and templates for Attributa AI Assistant
 */

import { AnalysisResult } from '@/types';

export const SYSTEM_PROMPT = `You are an AI assistant for Attributa, a privacy-first attribution intelligence tool. Your role is to help users understand AI content detection analysis results and provide guidance on using the tool effectively.

**Key Responsibilities:**
1. Explain analysis results (GLTR, DetectGPT, watermark detection scores)
2. Help users interpret confidence levels and what they mean
3. Provide guidance on when results are reliable vs. unreliable
4. Explain the limitations of AI detection methods
5. Suggest improvements to analysis workflows
6. Answer questions about the tool's methodology

**Important Context:**
- Attributa uses local ML models (GPT-2 based) for privacy
- GLTR analyzes token rank distributions (AI uses more predictable tokens)
- DetectGPT measures probability curvature through perturbations
- Short texts (<1000 chars) are unreliable and get ×0.5 penalty
- Non-English text may produce inconsistent results
- These are probabilistic indicators, NOT definitive proof
- Human-written formal/technical text can trigger false positives

**Tone:**
- Professional but approachable
- Honest about limitations and uncertainty
- Educational and informative
- Encourage critical thinking
- Never claim absolute certainty in AI detection

**Response Guidelines:**
- Be concise (2-3 paragraphs max unless asked for detail)
- Use markdown formatting for clarity
- Cite specific metrics when discussing results
- Highlight both strengths and weaknesses in analysis
- Suggest actionable next steps when relevant`;

export interface AnalysisContext {
  result?: AnalysisResult;
  currentPage?: string;
  userQuestion?: string;
}

/**
 * Build context-aware system prompt based on current app state
 */
export function buildContextualPrompt(context: AnalysisContext): string {
  let prompt = SYSTEM_PROMPT;

  if (context.currentPage === 'results' && context.result) {
    prompt += `\n\n**Current Analysis Context:**
- Overall Score: ${(context.result.overallScore * 100).toFixed(1)}%
- Confidence: ${context.result.confidence}
- Segments Analyzed: ${context.result.segments?.length || 0}
- Document: ${context.result.documentName}`;

    if (context.result.gltrScore !== undefined) {
      prompt += `\n- GLTR Score: ${(context.result.gltrScore * 100).toFixed(1)}%`;
    }
    if (context.result.detectGptScore !== undefined) {
      prompt += `\n- DetectGPT Score: ${(context.result.detectGptScore * 100).toFixed(1)}%`;
    }
    if (context.result.watermarkScore !== undefined) {
      prompt += `\n- Watermark Score: ${(context.result.watermarkScore * 100).toFixed(1)}%`;
    }
  }

  return prompt;
}

/**
 * Quick response templates for common questions
 */
export const QUICK_RESPONSES = {
  'explain-score': (score: number) => `Your overall attribution score is ${(score * 100).toFixed(1)}%, which indicates ${
    score > 0.8 ? 'strong signals consistent with AI generation' :
    score > 0.5 ? 'moderate signals that warrant further investigation' :
    'weak signals, more consistent with human authorship'
  }. This is based on multiple probabilistic indicators, not definitive proof.`,

  'why-unreliable': (length: number) => `This text segment is ${length} characters, which is ${
    length < 1000 ? 'below the 1000-character threshold for reliable detection' : 'quite short'
  }. AI detection methods need sufficient text to identify patterns. Short segments receive a ×0.5 confidence penalty.`,

  'false-positives': 'False positives can occur with formal/technical writing, heavily edited text, or non-English content. GLTR is trained on English text, so other languages may produce unreliable results. Always use multiple signals and human judgment.',

  'what-is-gltr': 'GLTR (Giant Language Model Test Room) analyzes how predictable each word is. AI models tend to choose more predictable, high-probability words (top-10 tokens). Human writers use more varied vocabulary. A high GLTR score suggests AI-like word predictability.',

  'what-is-detectgpt': 'DetectGPT measures probability curvature by creating slight variations of the text and checking if probabilities decrease. AI-generated text shows negative curvature (perturbations are less likely), while human text shows positive curvature (more natural variation).',

  'what-is-watermark': 'Watermark detection looks for statistical patterns in token choices. Some AI systems use "greenlist" tokens (slightly boosted probability). This is experimental and easily broken by paraphrasing or post-editing.',

  'next-steps-high': 'Given the high attribution score, consider: (1) Verify with multiple tools, (2) Check for properly cited sources, (3) Review document history/drafts, (4) Discuss findings with the author in good faith. Remember: these are indicators, not proof.',

  'next-steps-low': 'The low score suggests human authorship, but remember: (1) AI detection is not perfect, (2) Edited AI text can pass detection, (3) Paraphrasing breaks most detection methods. Use this as one data point among many.',
};

/**
 * Generate a help message for a specific feature
 */
export function getFeatureHelp(feature: string): string {
  const helpText: Record<string, string> = {
    scan: 'Upload a PDF or paste text to analyze. For best results, use documents with at least 1000 characters of continuous text. Non-English text may produce unreliable results.',

    results: 'Review your analysis results here. The overall score combines GLTR, DetectGPT, and optional watermark detection. Click on segments to see detailed breakdowns. Confidence levels indicate reliability: High (>80%), Medium (50-80%), Low (<50%).',

    citations: 'Citation validation checks DOIs against CrossRef to identify potentially fabricated references. This requires opt-in and is rate-limited to respect CrossRef terms of service.',

    code: 'Code security scanning identifies common vulnerabilities (CWEs) often present in AI-generated code. This is pattern-based and should be verified with dedicated security tools.',

    segments: 'Text is automatically split into analyzable segments. Each segment is scored independently, and short segments (<1200 chars) receive a ×0.5 confidence penalty.',

    export: 'Export your results as JSON (for programmatic use), PDF report (for documentation), or BibTeX (for academic citations). All exports include methodology details and limitations.',
  };

  return helpText[feature] || 'Feature help not available. Ask me a specific question about this feature.';
}

/**
 * Generate a summary of analysis limitations
 */
export function getLimitationsMessage(): string {
  return `**Important Limitations:**

⚠️ **Not Definitive Proof** - These are probabilistic indicators based on statistical patterns, not forensic evidence.

⚠️ **Short Text Unreliable** - Segments under 1000 characters may produce false positives/negatives.

⚠️ **Easy to Defeat** - Light paraphrasing or editing breaks most detection methods.

⚠️ **English-Centric** - Trained on English text; other languages may show inconsistent results.

⚠️ **False Positives** - Formal/technical human writing can trigger AI-like patterns.

**Best Practices:**
- Use as one data point among many
- Combine with source verification, citation checks, and human judgment
- Consider document history and context
- Approach findings with good faith and transparency`;
}
