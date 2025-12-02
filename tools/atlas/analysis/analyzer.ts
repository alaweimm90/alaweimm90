// ATLAS Repository Analyzer - AST parsing and complexity metrics

import type {
  RepositoryMetrics,
  CodeAnalysis,
  FunctionInfo,
  ClassInfo,
  ImportInfo,
  CodeIssue,
} from '../types/index';

// Stub class - types preserved for future implementation
export class RepositoryAnalyzer {
  // Types are imported for interface documentation

  private _types:
    | RepositoryMetrics
    | CodeAnalysis
    | FunctionInfo
    | ClassInfo
    | ImportInfo
    | CodeIssue
    | undefined;
}
