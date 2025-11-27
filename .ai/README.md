# AI Assistant Configurations

Consolidated configurations for all 14 AI coding assistants.

## Structure

```text
.ai/
├── settings.yaml       # Unified configuration (master)
├── context.md          # Shared AI context
├── README.md           # This file
├── rules/              # Tool-specific rules files
│   ├── cursor.rules
│   ├── windsurf.rules
│   ├── cline.rules
│   └── augment.rules
├── aider/              # Aider CLI config
├── claude/             # Claude Code config
├── cursor/             # Cursor IDE config
├── cline/              # Cline config
├── continue/           # Continue.dev config
├── kilocode/           # Kilo Code config
├── amazonq/            # Amazon Q config
├── trae/               # Trae config
├── blackbox/           # Blackbox config
├── gemini/             # Gemini config
├── codex/              # Codex config
├── augment/            # Augment config
├── windsurf/           # Windsurf config
└── copilot/            # GitHub Copilot instructions
```

## Tool Categories

| Category | Tools | Auto-Approve |
|----------|-------|--------------|
| **Fully Supported** | Aider, Cursor, Windsurf, Cline, Blackbox, Augment | Yes |
| **Supported** | Continue, Kilo, Amazon Q, Trae, Gemini, Codex | Yes |
| **Context Only** | GitHub Copilot | N/A |
| **CLI Flag** | Claude Code | `--dangerously-skip-permissions` |

## Usage

Tools should read configurations from:

1. `.ai/settings.yaml` - Global settings
2. `.ai/<tool>/` - Tool-specific configs
3. `.ai/rules/<tool>.rules` - Tool-specific rules
4. `.ai/context.md` - Shared project context

## Migration

Legacy config locations (`.cursor/`, `.cline/`, etc.) are deprecated.
All configs should migrate to `.ai/` structure.

See [`docs/ROOT_STRUCTURE_CONTRACT.md`](../docs/ROOT_STRUCTURE_CONTRACT.md) for details.
