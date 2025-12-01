---
name: "Prompt Optimization Superprompt"
version: "1.0"
category: "project"
tags: ["prompt-engineering", "optimization", "llm", "blackbox", "ide"]
created: "2024-11-30"
---

# Prompt Optimization Superprompt

## Purpose

Comprehensive framework for optimizing prompts for length, formatting, code generation quality, and IDE-specific features including Blackbox AI, Cursor, and other AI coding assistants.

---

## System Prompt

```text
You are a Prompt Engineering Expert with expertise in:
- LLM prompt optimization and token efficiency
- Code generation prompt design
- IDE-specific prompt formatting (Blackbox, Cursor, Copilot)
- Context window management
- Output quality optimization
- Multi-turn conversation design

Your mission is to create prompts that:
1. Maximize output quality within token limits
2. Generate accurate, production-ready code
3. Leverage IDE-specific features effectively
4. Maintain consistency across interactions
5. Enable efficient context utilization
```

---

## Prompt Optimization Principles

### Token Efficiency

```yaml
token_optimization:
  principles:
    - Use concise, precise language
    - Eliminate redundant instructions
    - Leverage structured formats (YAML, JSON)
    - Use abbreviations where clear
    - Prioritize high-value context
    
  techniques:
    compression:
      - Replace verbose phrases with keywords
      - Use bullet points over paragraphs
      - Consolidate similar instructions
      - Remove filler words
      
    prioritization:
      - Most important context first
      - Recent context over historical
      - Specific over general
      - Examples over explanations
      
    chunking:
      - Split large tasks into subtasks
      - Use continuation prompts
      - Implement sliding context windows
      - Cache reusable context
```

### Prompt Structure Templates

```yaml
# Optimal prompt structure for code generation
code_generation_template:
  structure:
    1_context: |
      # Context (10-15% of tokens)
      - Language/framework
      - Project structure
      - Relevant dependencies
      
    2_task: |
      # Task (5-10% of tokens)
      - Clear, specific objective
      - Expected behavior
      - Constraints
      
    3_examples: |
      # Examples (20-30% of tokens)
      - Input/output pairs
      - Edge cases
      - Style reference
      
    4_output_spec: |
      # Output Specification (5% of tokens)
      - Format requirements
      - Naming conventions
      - Documentation needs
      
    5_reserved: |
      # Reserved for response (50-60% of tokens)
      - Leave room for complete response
      - Account for code + explanations
```

---

## IDE-Specific Optimization

### Blackbox AI Optimization

```yaml
blackbox_optimization:
  features:
    code_autocomplete:
      trigger: "Start typing code"
      optimization:
        - Write clear function signatures
        - Use descriptive variable names
        - Add type hints for better suggestions
        - Include docstrings for context
        
    code_chat:
      trigger: "Select code + ask question"
      optimization:
        - Select minimal relevant code
        - Ask specific, focused questions
        - Reference line numbers when helpful
        - Request specific output format
        
    code_generation:
      trigger: "Describe what you need"
      optimization:
        - Start with language/framework
        - Specify function signature
        - Include example usage
        - Define error handling needs

  prompt_templates:
    function_generation: |
      Create a {language} function:
      - Name: {function_name}
      - Input: {parameters with types}
      - Output: {return type and description}
      - Behavior: {specific behavior}
      - Handle: {error cases}
      
    refactoring: |
      Refactor this {language} code:
      - Goal: {optimization goal}
      - Maintain: {preserved behavior}
      - Improve: {specific improvements}
      - Constraints: {limitations}
      
    debugging: |
      Debug this {language} code:
      - Error: {error message}
      - Expected: {expected behavior}
      - Actual: {actual behavior}
      - Context: {relevant context}
```

### Cursor IDE Optimization

```yaml
cursor_optimization:
  features:
    cmd_k:
      description: "Inline code generation"
      optimization:
        - Select context code first
        - Use natural language descriptions
        - Specify output location
        - Reference existing patterns
        
    cmd_l:
      description: "Chat with codebase"
      optimization:
        - Ask about specific files/functions
        - Request explanations with examples
        - Use for architecture decisions
        - Leverage for code review
        
    composer:
      description: "Multi-file editing"
      optimization:
        - Describe full feature scope
        - List all affected files
        - Specify dependencies
        - Define testing requirements

  prompt_templates:
    feature_implementation: |
      Implement {feature_name}:
      
      Files to modify:
      - {file1}: {changes}
      - {file2}: {changes}
      
      Requirements:
      - {requirement1}
      - {requirement2}
      
      Follow patterns from: {reference_file}
      
    codebase_question: |
      In this codebase:
      - How does {component} work?
      - Where is {functionality} implemented?
      - What's the pattern for {pattern_type}?
      
      Focus on: {specific_aspect}
```

### GitHub Copilot Optimization

```yaml
copilot_optimization:
  features:
    inline_suggestions:
      optimization:
        - Write clear comments before code
        - Use consistent naming patterns
        - Provide type annotations
        - Structure code predictably
        
    copilot_chat:
      optimization:
        - Use /commands effectively
        - Reference @workspace for context
        - Ask for specific formats
        - Request step-by-step solutions

  effective_comments:
    function_hint: |
      // Function: {name}
      // Input: {params}
      // Output: {return}
      // Example: {usage}
      
    algorithm_hint: |
      // Algorithm: {name}
      // Time: O({complexity})
      // Space: O({complexity})
      // Steps:
      // 1. {step1}
      // 2. {step2}
      
    pattern_hint: |
      // Pattern: {pattern_name}
      // Use case: {when_to_use}
      // Implementation:
```

---

## Code Generation Quality

### Quality Optimization Techniques

```typescript
// prompts/code-quality.ts

export const codeQualityPrompts = {
  // Production-ready code generation
  productionCode: (spec: CodeSpec) => `
Generate production-ready ${spec.language} code:

Requirements:
- ${spec.requirements.join('\n- ')}

Quality Standards:
- Full error handling with specific error types
- Input validation for all parameters
- Comprehensive logging at appropriate levels
- Type safety (no any types in TypeScript)
- Unit test coverage considerations
- Performance optimized for ${spec.scale}

Code Style:
- Follow ${spec.styleGuide} conventions
- Use meaningful variable/function names
- Add JSDoc/docstring comments
- Maximum function length: 50 lines
- Maximum file length: 300 lines

Output Format:
\`\`\`${spec.language}
// Implementation here
\`\`\`

Include:
- Main implementation
- Type definitions
- Example usage
- Error types`,

  // Refactoring prompt
  refactoring: (code: string, goals: string[]) => `
Refactor the following code:

\`\`\`
${code}
\`\`\`

Refactoring Goals:
${goals.map((g, i) => `${i + 1}. ${g}`).join('\n')}

Constraints:
- Maintain all existing functionality
- Preserve public API signatures
- Keep backward compatibility
- Add no new dependencies

Output:
1. Refactored code
2. List of changes made
3. Potential risks/considerations`,

  // Bug fix prompt
  bugFix: (code: string, error: string, expected: string) => `
Fix the bug in this code:

Code:
\`\`\`
${code}
\`\`\`

Error/Issue: ${error}
Expected Behavior: ${expected}

Provide:
1. Root cause analysis
2. Fixed code
3. Explanation of the fix
4. Test case to verify the fix`,
};
```

### Context Window Management

```typescript
// lib/context-manager.ts

interface ContextItem {
  content: string;
  priority: number;
  tokens: number;
  type: 'system' | 'code' | 'example' | 'history';
}

export class ContextManager {
  private maxTokens: number;
  private items: ContextItem[] = [];
  
  constructor(maxTokens: number = 8000) {
    this.maxTokens = maxTokens;
  }
  
  add(item: Omit<ContextItem, 'tokens'>): void {
    const tokens = this.estimateTokens(item.content);
    this.items.push({ ...item, tokens });
  }
  
  build(): string {
    // Sort by priority (higher first)
    const sorted = [...this.items].sort((a, b) => b.priority - a.priority);
    
    let totalTokens = 0;
    const included: ContextItem[] = [];
    
    for (const item of sorted) {
      if (totalTokens + item.tokens <= this.maxTokens) {
        included.push(item);
        totalTokens += item.tokens;
      }
    }
    
    // Reorder by type for coherent prompt
    const ordered = this.orderByType(included);
    
    return ordered.map(item => item.content).join('\n\n');
  }
  
  private orderByType(items: ContextItem[]): ContextItem[] {
    const typeOrder = ['system', 'code', 'example', 'history'];
    return items.sort((a, b) => 
      typeOrder.indexOf(a.type) - typeOrder.indexOf(b.type)
    );
  }
  
  private estimateTokens(text: string): number {
    // Rough estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
  }
  
  getUsage(): { used: number; max: number; percentage: number } {
    const used = this.items.reduce((sum, item) => sum + item.tokens, 0);
    return {
      used,
      max: this.maxTokens,
      percentage: Math.round((used / this.maxTokens) * 100),
    };
  }
}

// Usage example
const context = new ContextManager(8000);

context.add({
  content: 'You are a senior TypeScript developer...',
  priority: 100,
  type: 'system',
});

context.add({
  content: '// Current file content...',
  priority: 90,
  type: 'code',
});

context.add({
  content: '// Example of similar implementation...',
  priority: 70,
  type: 'example',
});

const prompt = context.build();
```

---

## Prompt Testing Framework

```typescript
// testing/prompt-testing.ts

interface PromptTest {
  name: string;
  prompt: string;
  expectedPatterns: RegExp[];
  forbiddenPatterns: RegExp[];
  qualityChecks: QualityCheck[];
}

interface QualityCheck {
  name: string;
  check: (response: string) => boolean;
}

interface TestResult {
  name: string;
  passed: boolean;
  details: {
    patternsMatched: string[];
    patternsMissed: string[];
    forbiddenFound: string[];
    qualityResults: { name: string; passed: boolean }[];
  };
}

export class PromptTester {
  private llmClient: LLMClient;
  
  constructor(llmClient: LLMClient) {
    this.llmClient = llmClient;
  }
  
  async runTest(test: PromptTest): Promise<TestResult> {
    const response = await this.llmClient.chat([
      { role: 'user', content: test.prompt }
    ]);
    
    const content = response.content;
    
    // Check expected patterns
    const patternsMatched: string[] = [];
    const patternsMissed: string[] = [];
    
    for (const pattern of test.expectedPatterns) {
      if (pattern.test(content)) {
        patternsMatched.push(pattern.source);
      } else {
        patternsMissed.push(pattern.source);
      }
    }
    
    // Check forbidden patterns
    const forbiddenFound: string[] = [];
    for (const pattern of test.forbiddenPatterns) {
      if (pattern.test(content)) {
        forbiddenFound.push(pattern.source);
      }
    }
    
    // Run quality checks
    const qualityResults = test.qualityChecks.map(check => ({
      name: check.name,
      passed: check.check(content),
    }));
    
    const passed = 
      patternsMissed.length === 0 &&
      forbiddenFound.length === 0 &&
      qualityResults.every(r => r.passed);
    
    return {
      name: test.name,
      passed,
      details: {
        patternsMatched,
        patternsMissed,
        forbiddenFound,
        qualityResults,
      },
    };
  }
  
  async runSuite(tests: PromptTest[]): Promise<TestResult[]> {
    const results: TestResult[] = [];
    
    for (const test of tests) {
      const result = await this.runTest(test);
      results.push(result);
    }
    
    return results;
  }
}

// Example test suite
const codeGenTests: PromptTest[] = [
  {
    name: 'TypeScript function generation',
    prompt: `Generate a TypeScript function that validates email addresses.
Include: type annotations, error handling, JSDoc comments.`,
    expectedPatterns: [
      /function\s+\w+.*email/i,
      /:\s*(string|boolean)/,
      /\/\*\*[\s\S]*\*\//,
      /throw|Error/,
    ],
    forbiddenPatterns: [
      /any(?!\w)/,  // No 'any' type
      /console\.log/,  // No console.log in production code
    ],
    qualityChecks: [
      {
        name: 'Has type annotations',
        check: (r) => /:\s*(string|boolean|number|void)/.test(r),
      },
      {
        name: 'Has error handling',
        check: (r) => /try|throw|catch|Error/.test(r),
      },
      {
        name: 'Has documentation',
        check: (r) => /\/\*\*[\s\S]*@param[\s\S]*\*\//.test(r),
      },
    ],
  },
];
```

---

## Optimization Checklist

```yaml
prompt_optimization_checklist:
  before_sending:
    - [ ] Remove unnecessary whitespace
    - [ ] Consolidate redundant instructions
    - [ ] Verify context relevance
    - [ ] Check token count vs limit
    - [ ] Validate output format specification
    
  structure:
    - [ ] Clear task definition first
    - [ ] Relevant context included
    - [ ] Examples provided if complex
    - [ ] Output format specified
    - [ ] Constraints clearly stated
    
  quality:
    - [ ] Specific over vague language
    - [ ] Actionable instructions
    - [ ] Measurable success criteria
    - [ ] Edge cases addressed
    - [ ] Error handling specified
    
  ide_specific:
    - [ ] Leverage IDE features
    - [ ] Use appropriate triggers
    - [ ] Format for tool expectations
    - [ ] Include file context when needed
```

---

## Execution Phases

### Phase 1: Analysis

- [ ] Audit existing prompts
- [ ] Measure token usage
- [ ] Identify optimization opportunities
- [ ] Benchmark current quality

### Phase 2: Optimization

- [ ] Apply compression techniques
- [ ] Restructure for efficiency
- [ ] Add quality specifications
- [ ] Implement context management

### Phase 3: Testing

- [ ] Create test suite
- [ ] Run quality checks
- [ ] Measure improvements
- [ ] Iterate on failures

### Phase 4: Integration

- [ ] Deploy optimized prompts
- [ ] Configure IDE settings
- [ ] Train team on best practices
- [ ] Monitor and iterate

---

*Last updated: 2024-11-30*
