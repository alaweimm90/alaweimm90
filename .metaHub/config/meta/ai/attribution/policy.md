# AI Attribution and Credit Policy

**Engineering Excellence Framework - AI Integration Standards**

This policy establishes the requirements for attributing AI-generated content, tools, and assistance throughout the development lifecycle.

## 🎯 Policy Overview

### Purpose
- Ensure transparency in AI-assisted development
- Maintain code quality and accountability
- Track AI contributions for auditing and compliance
- Facilitate proper credit attribution

### Scope
This policy applies to:
- All code repositories in the monorepo
- Documentation and technical writing
- Design and architecture decisions
- Testing and quality assurance
- CI/CD pipeline automation

## 📋 Attribution Requirements

### Code Attribution

#### AI-Generated Code
```typescript
// AI-generated with Claude - Initial implementation [2025-11-XX]
// Reviewed and approved by: @developer-name
// Source: .meta/ai/prompts/code-generation.md#template-name
function exampleFunction(param: string): boolean {
  // Implementation details...
}
```

#### AI-Assisted Code
```typescript
// AI-assisted refactoring with GitHub Copilot
// Original logic by: @developer-name
// AI suggestions integrated by: @reviewer-name
// Prompt context: .meta/ai/prompts/refactoring.md#pattern-name
function refactoredFunction(input: InputType): OutputType {
  // Refactored implementation...
}
```

### Documentation Attribution

#### AI-Generated Documentation
```markdown
<!-- AI-generated documentation using Claude -->
<!-- Template: .meta/ai/prompts/documentation.md#api-docs -->
<!-- Reviewed by: @technical-writer @engineering-lead -->
<!-- Generated: 2025-11-XX -->
```

#### AI-Assisted Documentation
```markdown
<!-- AI-assisted content enhancement -->
<!-- Original draft by: @content-author -->
<!-- AI improvements by Claude -->
<!-- Final review by: @editor-name -->
```

### Architecture Decisions

#### ADRs (Architecture Decision Records)
```markdown
# ADR 042: Implementation Decision Title

## Status
Accepted

## Context
[Decision context...]

## AI Contribution
- **Tool Used**: Claude
- **Prompt Template**: .meta/ai/prompts/architecture.md#decision-analysis
- **AI Analysis Reviewed By**: @architecture-team
- **Final Decision By**: @tech-lead-name

## Decision
[Decision details...]
```

## 🏷️ Attribution Tags

### Code Comment Tags

```typescript
// AI:GENERATED - Claude - [Date] - Template: template-name
// AI:ASSISTED - Copilot - [Date] - Context: context-description
// AI:REVIEWED - Claude - [Date] - Reviewer: @username
// AI:REFACTORED - ChatGPT - [Date] - Pattern: pattern-name
```

### Commit Message Tags

```
feat: add user authentication
AI:GENERATED - Claude/code-generation-template
Reviewed-by: @security-team

refactor: simplify data processing
AI:ASSISTED - Copilot/inline-suggestions
Tested-by: @qa-team
```

### Pull Request Labels

- `🤖 ai-generated` - Primarily AI-generated content
- `🤖 ai-assisted` - Significant AI assistance
- `🤖 ai-reviewed` - AI-assisted code review
- `🤖 ai-tools` - Changes to AI tooling/infrastructure

## 📊 Tracking and Metrics

### Attribution Metadata
All AI contributions must include:
- Tool/version used
- Prompt template reference
- Date of generation/assistance
- Human reviewer identifier
- Approval status

### Quality Metrics
- Acceptance rate of AI suggestions
- Error rates in AI-generated code
- Review time for AI contributions
- Compliance with attribution requirements

## 🔍 Audit and Compliance

### Automated Checks
Pre-commit hooks verify:
- Presence of attribution comments in new code
- Correct formatting of attribution tags
- Reference to valid prompt templates
- Human reviewer identification

### Quarterly Audits
- Review attribution compliance across repositories
- Assess effectiveness of AI integration policies
- Update attribution standards based on findings
- Report on AI contribution quality metrics

## 📝 Responsibilities

### AI Tool Users
- Follow attribution standards for all AI interactions
- Include proper metadata in commits and PRs
- Ensure human review of AI-generated content
- Report issues with AI tool quality

### Code Reviewers
- Verify attribution compliance in PRs
- Assess quality of AI-generated code
- Provide feedback on AI tool effectiveness
- Escalate concerns about AI attribution violations

### Engineering Leadership
- Maintain SSOT documentation
- Review attribution policy effectiveness
- Approve updates to AI tooling and standards
- Ensure organization-wide compliance

## 🚫 Prohibited Practices

### Attribution Violations
- Removing or modifying attribution comments without approval
- False attribution (claiming human work as AI-generated)
- Using AI tools without following attribution requirements
- Merging AI-generated code without human review

### Quality Compromises
- Accepting AI suggestions without understanding the code
- Skipping testing for AI-generated features
- Ignoring linting errors from AI-generated code

## 🛠️ Implementation

### Pre-commit Hook Integration
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: ai-attribution-check
      name: AI Attribution Compliance
      entry: scripts/check-ai-attribution.sh
      language: system
```

### CI/CD Integration
```yaml
# .github/workflows/ci.yml
- name: AI Attribution Audit
  run: npm run audit:ai-attribution
- name: AI Quality Metrics
  run: npm run collect:ai-metrics
```

### Tool-Specific Attribution

#### Claude
```typescript
// @ai-generated claude-3.5-sonnet
// @prompt .meta/ai/prompts/code-generation.md#function-template
// @reviewed @developer-name
// @approved 2025-11-XX
```

#### GitHub Copilot
```typescript
// @ai-assisted copilot-chat
// @context inline-suggestion
// @session vscode-session-123
// @reviewed @developer-name
```

#### ChatGPT
```typescript
// @ai-assisted chatgpt-4
// @conversation https://chat.openai.com/share/abc123
// @reviewed @developer-name
// @approved 2025-11-XX
```

## 📈 Policy Evolution

### Annual Review
- Assess policy effectiveness
- Update attribution standards
- Incorporate new AI tool capabilities
- Adjust requirements based on organizational needs

### Continuous Improvement
- Monitor emerging attribution standards
- Adapt to new AI tool integrations
- Update training materials for developers
- Refine automated compliance checks

---

**Enforcement**: Violations of this policy may result in code revert, additional review requirements, or temporary suspension of AI tool access. This policy is mandatory for all Engineering Excellence Framework repositories.
