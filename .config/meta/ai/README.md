# AI Single Source of Truth (SSOT)

**Engineering Excellence Framework - AI Integration Standards**

This directory serves as the centralized hub for all AI-related instructions, prompts, tools, and context used across the monorepo. All AI interactions must reference this SSOT to ensure consistency, quality, and maintainability.

## 📁 Directory Structure

```
.meta/ai/
├── README.md                    # This overview document
├── instructions/                # Core AI instructions and guidelines
│   ├── development.md           # Development workflow guidelines
│   ├── code-review.md           # AI-assisted code review standards
│   ├── testing.md              # AI testing and QA guidelines
│   └── documentation.md        # Documentation generation standards
├── prompts/                     # Standardized prompts for common tasks
│   ├── code-generation.md       # Code generation templates
│   ├── refactoring.md           # Code refactoring prompts
│   ├── debugging.md             # Debug assistance prompts
│   └── architecture.md          # Architecture design prompts
├── tools/                       # Tool-specific configurations and adapters
│   ├── claude/                  # Claude-specific settings
│   ├── chatgpt/                 # ChatGPT-specific configurations
│   ├── copilot/                 # GitHub Copilot rules
│   └── custom/                  # Custom AI tool integrations
├── context/                     # Project context and knowledge base
│   ├── architecture.md          # System architecture overview
│   ├── patterns.md              # Code patterns and conventions
│   ├── glossary.md              # Technical terminology
│   └── constraints.md           # Project constraints and limitations
└── attribution/                 # Attribution and policies
    ├── policy.md                # AI usage attribution policy
    ├── credits.md               # AI-generated content tracking
    └── sync-config.json         # Automated sync configuration
```

## 🔄 AI Integration Policies

### Attribution Requirements
- All AI-generated code must be clearly marked with attribution comments
- AI-assisted reviews must be documented in commit messages
- Major AI contributions must be noted in CHANGELOG.md

### Quality Standards
- AI-generated code must pass all linting and testing requirements
- AI suggestions must be reviewed by human developers before merging
- Complex AI-generated features require additional testing

### Tool-Specific Adapters
Each AI tool integrated with the repository must have:
- Standardized prompt templates
- Quality validation checklists
- Attribution tagging mechanisms
- Output formatting requirements

## 🚀 Usage Guidelines

### For Developers
1. **Always check SSOT first**: Reference these standards before using AI tools
2. **Use standardized prompts**: Leverage `/prompts/` for consistent results
3. **Follow attribution rules**: Mark AI contributions appropriately
4. **Validate AI outputs**: Don't merge AI code without proper review

### For AI Tools
1. **Load context first**: Always include relevant `/context/` files
2. **Use appropriate prompts**: Select from `/prompts/` based on task type
3. **Follow formatting rules**: Adhere to project's code standards
4. **Generate attribution**: Include proper attribution in outputs

## 🔧 Integration Points

### Pre-commit Hooks
- AI-generated code validation
- Attribution compliance checks
- Prompt standardization enforcement

### CI/CD Pipeline
- AI contribution tracking
- Attribution verification
- Quality gate compliance

### Documentation Sync
- Automated SSOT updates across repos
- Attribution report generation
- Quality metrics collection

## 📊 Monitoring & Compliance

### Quality Metrics
- AI-generated code acceptance rate
- Attribution compliance percentage
- Review feedback on AI suggestions
- Error rates in AI-assisted development

### Audit Requirements
- Quarterly AI integration audits
- Annual attribution policy review
- Continuous monitoring of AI tool effectiveness

## 🔒 Security Considerations

### Data Protection
- No sensitive project data in AI prompts
- Sanitized context for external AI tools
- Encrypted storage of AI configurations

### Access Control
- Restricted editing of SSOT files
- Tool-specific authentication requirements
- Audit logging of AI tool usage

## 🛠️ Tool-Specific Configurations

### Claude
- **Location**: `tools/claude/`
- **Capabilities**: Code generation, review, documentation
- **Quality Gates**: Enhanced context loading, attribution tagging

### ChatGPT
- **Location**: `tools/chatgpt/`
- **Capabilities**: Refactoring, debugging, analysis
- **Quality Gates**: Prompt standardization, output validation

### GitHub Copilot
- **Location**: `tools/copilot/`
- **Capabilities**: Real-time code suggestions
- **Quality Gates**: Custom rules enforcement, attribution requirements

## 📝 Maintenance

### Regular Updates
- Monthly review of prompt effectiveness
- Quarterly update of context documentation
- Annual policy review and updates

### Change Management
- All SSOT changes require engineering review
- New AI tools require security and compliance approval
- Prompt updates must maintain backward compatibility

---

- **Capabilities**: Real-time code suggestions
