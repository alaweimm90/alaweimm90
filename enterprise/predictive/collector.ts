// ATLAS Enterprise - Metrics Collector

import { CodeMetrics } from './types.js';

/**
 * Collects code quality metrics from source code
 */
export class MetricsCollector {
  /**
   * Collect comprehensive metrics from code
   */
  async collectMetrics(code: string, language: string): Promise<Omit<CodeMetrics, 'predictions' | 'timestamp'>> {
    const lines = code.split('\n');
    const cleanLines = lines.filter(line => line.trim().length > 0);

    return {
      linesOfCode: cleanLines.length,
      cyclomaticComplexity: this.calculateCyclomaticComplexity(code, language),
      maintainabilityIndex: this.calculateMaintainabilityIndex(code, language),
      halsteadVolume: this.calculateHalsteadVolume(code),
      commentRatio: this.calculateCommentRatio(code, language),
      duplicateLines: this.detectDuplicateLines(lines),
      technicalDebt: this.estimateTechnicalDebt(code, language),
      language,
      imports: this.countImports(code, language),
      functions: this.countFunctions(code, language),
      classes: this.countClasses(code, language)
    };
  }

  private calculateCyclomaticComplexity(code: string, language: string): number {
    // Simplified cyclomatic complexity calculation
    let complexity = 1; // Base complexity

    // Count control flow keywords
    const controlFlowKeywords = this.getControlFlowKeywords(language);
    const keywordRegex = new RegExp(`\\b(${controlFlowKeywords.join('|')})\\b`, 'gi');

    const matches = code.match(keywordRegex);
    if (matches) {
      complexity += matches.length;
    }

    // Count logical operators
    const logicalOps = code.match(/\|\||&&/g);
    if (logicalOps) {
      complexity += logicalOps.length;
    }

    return complexity;
  }

  private calculateMaintainabilityIndex(code: string, language: string): number {
    // Simplified maintainability index calculation
    const lines = code.split('\n').length;
    const complexity = this.calculateCyclomaticComplexity(code, language);
    const commentRatio = this.calculateCommentRatio(code, language);

    // MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(L) + 50 * sin(sqrt(2.4 * C))
    // Simplified version
    const volume = Math.log(lines + 1);
    const mi = 171 - 5.2 * volume - 0.23 * complexity + 50 * Math.sin(Math.sqrt(2.4 * commentRatio * 100));

    return Math.max(0, Math.min(171, mi));
  }

  private calculateHalsteadVolume(code: string): number {
    // Simplified Halstead volume calculation
    const operators = code.match(/[+\-*/=<>!&|%^~?:;,.(){}[\]]/g)?.length || 0;
    const operands = code.match(/\b\w+\b/g)?.length || 0;

    const n1 = Math.max(1, new Set(code.match(/[+\-*/=<>!&|%^~?:;,.(){}[\]]/g) || []).size);
    const n2 = Math.max(1, new Set(code.match(/\b\w+\b/g) || []).size);
    const N1 = operators;
    const N2 = operands;

    const vocabulary = n1 + n2;
    const length = N1 + N2;

    if (vocabulary === 0) return 0;

    return length * Math.log2(vocabulary);
  }

  private calculateCommentRatio(code: string, language: string): number {
    const commentPatterns = this.getCommentPatterns(language);
    let commentLines = 0;

    for (const pattern of commentPatterns) {
      const matches = code.match(pattern);
      if (matches) {
        commentLines += matches.length;
      }
    }

    const totalLines = code.split('\n').length;
    return totalLines > 0 ? commentLines / totalLines : 0;
  }

  private detectDuplicateLines(lines: string[]): number {
    const lineMap = new Map<string, number>();
    let duplicates = 0;

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length > 10) { // Only check substantial lines
        const count = lineMap.get(trimmed) || 0;
        lineMap.set(trimmed, count + 1);
        if (count > 0) {
          duplicates++;
        }
      }
    }

    return duplicates;
  }

  private estimateTechnicalDebt(code: string, language: string): number {
    // Simplified technical debt estimation
    let debt = 0;

    // Complexity debt
    const complexity = this.calculateCyclomaticComplexity(code, language);
    if (complexity > 10) {
      debt += (complexity - 10) * 10;
    }

    // Duplicate code debt
    const lines = code.split('\n');
    const duplicates = this.detectDuplicateLines(lines);
    debt += duplicates * 5;

    // Comment ratio debt
    const commentRatio = this.calculateCommentRatio(code, language);
    if (commentRatio < 0.1) {
      debt += (0.1 - commentRatio) * 1000;
    }

    return debt;
  }

  private countImports(code: string, language: string): number {
    const importPatterns = this.getImportPatterns(language);
    let count = 0;

    for (const pattern of importPatterns) {
      const matches = code.match(pattern);
      if (matches) {
        count += matches.length;
      }
    }

    return count;
  }

  private countFunctions(code: string, language: string): number {
    const functionPatterns = this.getFunctionPatterns(language);
    let count = 0;

    for (const pattern of functionPatterns) {
      const matches = code.match(pattern);
      if (matches) {
        count += matches.length;
      }
    }

    return count;
  }

  private countClasses(code: string, language: string): number {
    const classPatterns = this.getClassPatterns(language);
    let count = 0;

    for (const pattern of classPatterns) {
      const matches = code.match(pattern);
      if (matches) {
        count += matches.length;
      }
    }

    return count;
  }

  private getControlFlowKeywords(language: string): string[] {
    const keywords: Record<string, string[]> = {
      javascript: ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch'],
      typescript: ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch'],
      python: ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with'],
      java: ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch'],
      csharp: ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch']
    };

    return keywords[language.toLowerCase()] || [];
  }

  private getCommentPatterns(language: string): RegExp[] {
    const patterns: Record<string, RegExp[]> = {
      javascript: [/\/\/.*$/gm, /\/\*[\s\S]*?\*\//g],
      typescript: [/\/\/.*$/gm, /\/\*[\s\S]*?\*\//g],
      python: [/#.*$/gm, /"""[\s\S]*?"""/g, /'''[\s\S]*?'''/g],
      java: [/\/\/.*$/gm, /\/\*[\s\S]*?\*\//g],
      csharp: [/\/\/.*$/gm, /\/\*[\s\S]*?\*\//g]
    };

    return patterns[language.toLowerCase()] || [];
  }

  private getImportPatterns(language: string): RegExp[] {
    const patterns: Record<string, RegExp[]> = {
      javascript: [/import\s+.*from\s+['"]/g, /const\s+\w+\s*=\s*require\s*\(/g],
      typescript: [/import\s+.*from\s+['"]/g, /const\s+\w+\s*=\s*require\s*\(/g],
      python: [/^(import\s+\w+|from\s+\w+\s+import)/gm],
      java: [/^import\s+[\w.]+;/gm],
      csharp: [/^using\s+[\w.]+;/gm]
    };

    return patterns[language.toLowerCase()] || [];
  }

  private getFunctionPatterns(language: string): RegExp[] {
    const patterns: Record<string, RegExp[]> = {
      javascript: [/function\s+\w+\s*\(/g, /const\s+\w+\s*=\s*\([^)]*\)\s*=>/g, /\w+\s*\([^)]*\)\s*{/g],
      typescript: [/function\s+\w+\s*\(/g, /const\s+\w+\s*=\s*\([^)]*\)\s*=>/g, /\w+\s*\([^)]*\)\s*{/g],
      python: [/^def\s+\w+\s*\(/gm],
      java: [/public\s+.*\s+\w+\s*\(/g, /private\s+.*\s+\w+\s*\(/g, /protected\s+.*\s+\w+\s*\(/g],
      csharp: [/public\s+.*\s+\w+\s*\(/g, /private\s+.*\s+\w+\s*\(/g, /protected\s+.*\s+\w+\s*\(/g]
    };

    return patterns[language.toLowerCase()] || [];
  }

  private getClassPatterns(language: string): RegExp[] {
    const patterns: Record<string, RegExp[]> = {
      javascript: [/class\s+\w+/g],
      typescript: [/class\s+\w+/g],
      python: [/^class\s+\w+/gm],
      java: [/^class\s+\w+/gm, /^public\s+class\s+\w+/gm],
      csharp: [/^class\s+\w+/gm, /^public\s+class\s+\w+/gm]
    };

    return patterns[language.toLowerCase()] || [];
  }
}</content>
</edit_file>