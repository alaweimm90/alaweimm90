# Implementation Plan

## Overview

This project aims to simplify the GitHub repository structure by consolidating the duplicate automation systems. Currently, there are two automation frameworks implemented in different languages: the Python-based `automation/` directory and the TypeScript-based `automation-ts/` directory. These systems provide overlapping functionality for managing AI prompts, agents, workflows, orchestration, and deployment, sharing some configuration files but maintaining separate codebases.

The consolidation will merge both systems into a single, unified `automation/` directory. The TypeScript implementation will be chosen as the foundation due to its modernity, type safety, better test coverage, and inclusion of additional deployment features not present in the Python version. Python-specific components will be ported to TypeScript where functionality needs to be preserved, and shared configurations will be centralized.

This consolidation will reduce maintenance overhead, eliminate code duplication, improve developer experience by having one source of truth for automation assets, and simplify the overall repository structure while preserving all key functionality.

## Types

Extends the existing TypeScript interfaces and adds new ones for unified functionality.

| Type | Description | Location |
|------|-------------|----------|
| PromptConfig | Configuration for AI prompt templates with metadata | automation/types/PromptConfig.ts (new) |
| ExecutionContext | Unified context for workflow execution across deployment targets | automation/types/ExecutionContext.ts (new) |
| AgentTemplate | Template structure for agent configuration | automation/types/AgentTemplate.ts (extends existing) |
| ValidationRule | Rules for validating automation assets | automation/types/ValidationRule.ts (extends existing) |
| DeploymentTarget | Enumeration of supported deployment environments | automation/types/DeploymentTarget.ts (new) |

## Files

### New Files
- `automation/types/PromptConfig.ts` - TypeScript interfaces for prompt configuration
- `automation/types/ExecutionContext.ts` - Execution context types
- `automation/types/DeploymentTarget.ts` - Deployment target enumeration
- `automation/core/PythonBridge.ts` - Bridge for any Python functionality that must be preserved
- `automation/core/UnifiedExecutor.ts` - Replaced CLI functionality
- `automation/migration/migrate-py.sh` - Migration script for Python assets
- `automation/migration/migrate-ts.sh` - Migration script for TS assets

### Modified Files
- `automation/README.md` - Updated to reflect unified nature
- `automation/prompts/CATALOG.md` - Updated catalog entries
- `package.json` - Remove automation-ts references, update automation scripts
- `CLAUDE.md` - Update protected directories list

### Deleted Files
- All files in `automation-ts/` directory tree
- `automation/cli.py.old` - Outdated version
- `automation/executor.py.old` - Outdated version
- `automation/executor_final.py` - Temporary development file
- `automation/executor_new.py` - Temporary development file
- `automation/executor_refactored.py` - Temporary development file

### Moved Files
- Import TypeScript test files from `automation-ts/src/__tests__/` to `automation/__tests__/`
- Consolidate deployment registry logic from `automation-ts/` into `automation/deployment/`
- Move crew management from TypeScript to unified orchestration

## Functions

### New Functions
- `unifiedExecutor(routeTask)` in `automation/core/UnifiedExecutor.ts` - Main entry point replacing separate CLIs
- `validateAsset(asset, rules)` in `automation/validation/index.ts` - Unified validation for all automation assets
- `migrateAsset(asset, sourceType)` in `automation/migration/index.ts` - Asset migration helpers
- `deployToTarget(target, asset)` in `automation/deployment/index.ts` - Unified deployment interface

### Modified Functions
- `loadPrompts()` - Updated to merge Python and TS prompt loading logic
- `executeWorkflow()` - Extended to support both Python-style and TS-style execution
- `validateAgentConfig()` - Enhanced validation rules combining both system checks
- `routeTask()` - Unified routing decision logic

### Removed Functions
- All standalone CLI functions from `automation-ts/cli/` (replaced by unified)
- Temporary migration functions in `automation/executor_*.py` files

## Classes

### New Classes
- `AutomationCore` - Central orchestration class replacing separate CLI handlers
- `AssetMigrator` - Handles migration of assets between systems
- `ValidationEngine` - Unified validation engine with extensible rule system
- `DeploymentManager` - Manages deployment across different target environments

### Modified Classes
- `PromptManager` - Extended pattern matching to support Python prompt formats
- `WorkflowExecutor` - Enhanced execution pipeline to support backward compatibility
- `AgentLoader` - Updated configuration loading to merge YAML and JSON formats

### Removed Classes
- `PythonCLI` - Replaced by unified AutomationCore
- `TypescriptCLI` - Replaced by unified AutomationCore
- Legacy execution classes in executor files

## Dependencies

New packages for enhanced functionality:

| Package | Version | Purpose |
|---------|---------|---------|
| `yaml` | 2.3.4+ | Enhanced YAML parsing for migration |
| `@types/yaml` | 2.3.4+ | TypeScript definitions |
| `commander-next` | 0.1.0+ | Advanced CLI building |
| `chalk` | 5.6.2+ | Colorized console output |

Dependencies to be removed:
- None (consolidation doesn't remove working dependencies)

All dependencies managed through existing package managers (npm for TS, pip for any remaining Python bridges).

## Testing

### Testing Strategy
- **Unit Tests**: Migrate Jest tests from automation-ts/ to automation/__tests__/
- **Integration Tests**: Combined test suites testing unified functionality
- **Migration Tests**: Verify that Python assets can be processed by TS system
- **Compatibility Tests**: Ensure existing automation workflows continue working

### Test Files
- `automation/__tests__/unified-executor.test.ts`
- `automation/__tests__/migration.test.ts`
- `automation/__tests__/integration.test.ts`
- Updated `automation/__tests__/types.test.ts` with new interfaces

### Testing Dependencies
- Jest 29+ for unified test framework
- Mock libraries for external service testing
- Asset fixtures for testing migration scenarios

## Implementation Order

1. **Analysis Phase** (Week 1)
   - Complete asset inventory of both systems
   - Identify unique functionality in each system
   - Document breaking changes for consumers

2. **Type System Setup** (Week 2)
   - Create unified TypeScript interfaces
   - Implement validation rules
   - Set up testing framework skeleton

3. **Core Consolidation** (Week 3)
   - Merge executor logic into AutomationCore
   - Implement unified CLI interface
   - Migrate deployment management

4. **Asset Migration** (Week 4)
   - Migrate prompts and agents configurations
   - Update workflow definitions
   - Preserve backward compatibility

5. **Testing and Validation** (Week 5)
   - Implement comprehensive test suite
   - Validate all migration scenarios
   - Performance benchmarking

6. **Cleanup and Documentation** (Week 6)
   - Remove duplicate files (automation-ts/)
   - Update all documentation
   - Update CI/CD pipelines

7. **Deployment and Verification** (Week 7)
   - Deploy consolidated system
   - Verify existing workflows function correctly
   - Monitor for breaking changes
