# AI Prompts Library

Comprehensive prompts for AI-assisted repository management and consolidation.

## Available Prompts

### Repository Consolidation Suite

#### 1. Master Prompt

**File**: `repository-consolidation-master.md`
**Purpose**: Overall consolidation strategy and roadmap
**Use When**: Planning the full consolidation project

#### 2. Phase 1: Infrastructure

**File**: `phase-1-infrastructure.md`
**Purpose**: Consolidate deploy/, templates/ into .metaHub/
**Use When**: Executing infrastructure consolidation

#### 3. Phase 2: Tooling

**File**: `phase-2-tooling.md`
**Purpose**: Consolidate tools/, scripts/ into .metaHub/tools/
**Use When**: Executing tooling consolidation

#### 4. Phase 3: AI Integration

**File**: `phase-3-ai-integration.md`
**Purpose**: Merge ai-tools/ into .ai/tools/
**Use When**: Executing AI system integration

#### 5. Claude Opus Instructions

**File**: `claude-opus-instructions.md`
**Purpose**: Detailed execution protocol for Claude Opus
**Use When**: Delegating consolidation work to Claude Opus

## Usage

### For Human Developers

1. Read `repository-consolidation-master.md` for overview
2. Follow phase-specific prompts in order
3. Execute commands carefully
4. Test after each phase

### For AI Assistants

1. Load `claude-opus-instructions.md`
2. Follow execution protocol exactly
3. Report progress after each phase
4. Ask for clarification when needed

## Prompt Structure

Each prompt includes:

- **Objective**: Clear goal statement
- **Pre-Flight Checks**: Safety measures
- **Execution Steps**: Detailed commands
- **Path Updates**: Required code changes
- **Validation Tests**: Verification procedures
- **Rollback Procedure**: Recovery steps
- **Success Criteria**: Completion checklist
- **Commit Message**: Standardized commit format

## Safety Features

All prompts include:

- Backup branch creation
- Pre-execution validation
- Step-by-step testing
- Rollback procedures
- Git history preservation
- Zero data loss guarantees

## Execution Order

1. **Phase 6**: Cache cleanup (safest)
2. **Phase 5**: Documentation consolidation
3. **Phase 7**: Empty folder removal
4. **Phase 4**: Kubernetes cleanup
5. **Phase 1**: Infrastructure consolidation
6. **Phase 2**: Tooling consolidation
7. **Phase 3**: AI integration

## Contributing

When adding new prompts:

1. Follow existing structure
2. Include all safety measures
3. Provide clear validation steps
4. Document rollback procedures
5. Update this README

## Support

For issues or questions:

1. Check prompt documentation
2. Review execution logs
3. Consult backup branches
4. Ask for human guidance
