# Centralized DevTools Configuration

**Single source of truth for all IDE/CLI tools in this monorepo.**

## 📁 Structure

```
.devtools/
├── rules/                    # Shared development rules
│   ├── monorepo-structure.md
│   ├── code-quality.md
│   ├── security-first.md
│   ├── platform-architecture.md
│   └── automation-integration.md
│
├── mcps/                     # MCP configurations
│   ├── registry.json         # Available MCPs
│   └── settings.json         # MCP settings
│
├── integrations/             # Tool integrations
│   ├── shared.json           # Shared config
│   ├── turbo.json
│   ├── prisma.json
│   └── linting.json
│
├── setup.sh                  # Setup script
└── README.md                 # This file
```

## 🚀 Usage

### Initial Setup

```bash
# Run once to setup all tool integrations
bash .devtools/setup.sh
```

### Adding New Rules

```bash
# Create new rule file
echo "# My Rule" > .devtools/rules/my-rule.md

# All tools automatically inherit it
```

### Adding New MCPs

```bash
# Edit registry
nano .devtools/mcps/registry.json

# Add your MCP configuration
```

### Supported Tools

- **Amazon Q** - AI assistant in IDE
- **Cursor** - AI-powered code editor
- **Continue** - VS Code AI extension
- **Windsurf** - AI coding assistant
- **Cline** - CLI AI tool

## 🔗 How It Works

Each tool directory (`.amazonq/`, `.cursor/`, etc.) contains symlinks to `.devtools/`:

```bash
.amazonq/rules -> ../.devtools/rules
.cursor/rules -> ../.devtools/rules
```

**Benefits:**

- Update once, all tools inherit
- Zero duplication
- Version controlled
- Tool agnostic

## 📝 Maintenance

**Update a rule:**

```bash
nano .devtools/rules/code-quality.md
# Changes apply to all tools immediately
```

**Add new tool:**

```bash
# Edit setup.sh, add to TOOLS array
# Re-run setup
bash .devtools/setup.sh
```

## 🎯 Integration with Existing Automation

This configuration integrates with:

- `.automation/scripts/` - Automation scripts
- `.automation/hooks/` - Git hooks
- `turbo.json` - Task runner
- Existing security infrastructure

## 📞 Support

See main README.md or contact: meshal@berkeley.edu
