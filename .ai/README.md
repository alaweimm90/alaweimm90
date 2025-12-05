# AI Orchestration Hub

Unified AI configurations and self-learning superprompt infrastructure.

## Structure

```text
.ai/
├── context.yaml              # Central AI configuration and routing
├── settings.yaml             # Unified tool configuration (master)
├── context.md                # Shared AI context
├── task-history.json         # Historical task data
│
├── agents/                   # Agent definitions (from automation/)
│   ├── config/               # Agent registry and configurations
│   └── templates/            # Agent templates
│
├── workflows/                # Workflow definitions (from automation/)
│   ├── config/               # Workflow registry
│   └── templates/            # Workflow templates
│
├── orchestration/            # Orchestration patterns (from automation/)
│   ├── config/               # Orchestration configuration
│   └── patterns/             # Orchestration patterns
│
├── scripts/                  # AI utility scripts (from tools/ai/)
│
├── superprompts/             # Meta-cognitive guidance systems
│   ├── codebase-sentinel.yaml    # Code quality & 7 Laws
│   ├── security-auditor.yaml     # OWASP & security scanning
│   ├── architect.yaml            # System design patterns
│   ├── refactoring-expert.yaml   # Safe code transformation
│   └── debugger.yaml             # Scientific bug hunting
│
├── prompt-engine/            # Intelligent selection system
│   ├── engine.py                 # Python implementation
│   ├── selector.yaml             # Classification rules
│   └── orchestration-integration.yaml
│
├── learning/                 # Effectiveness tracking
│   └── effectiveness-tracker.yaml
│
├── rules/                    # Tool-specific rules files
│   ├── cursor.rules
│   ├── windsurf.rules
│   ├── cline.rules
│   └── augment.rules
│
└── [tool-configs]/           # Per-tool configurations
    ├── aider/
    ├── claude/
    ├── cursor/
    └── ... (14 tools total)
```

## Superprompt System

Superprompts are YAML-defined meta-cognitive guidance systems that enhance AI capabilities for specific task types.

### Available Superprompts

| ID                   | Specialty                          | Triggers                            |
| -------------------- | ---------------------------------- | ----------------------------------- |
| `codebase-sentinel`  | Code quality, 7 Laws of Integrity  | audit, quality, lint, governance    |
| `security-auditor`   | OWASP Top 10, secret detection     | security, vulnerability, auth       |
| `architect`          | System design, patterns, decisions | design, architecture, scale         |
| `refactoring-expert` | Safe refactoring, code smells      | refactor, extract, rename, simplify |
| `debugger`           | Root cause analysis, bug hunting   | bug, error, crash, debug            |

### The 7 Laws of Codebase Integrity

1. **No Silent Exception Swallowing** - Catch specific exceptions
2. **Documentation Matches Reality** - Keep docs accurate
3. **Policies as Code** - Machine-enforceable rules
4. **Single Source of Truth** - No configuration drift
5. **Secure by Default** - Never commit secrets
6. **Tests Reflect Intent** - Tests as documentation
7. **Dependencies are Explicit** - No hidden coupling

### Prompt Engine Usage

```python
from prompt_engine.engine import PromptEngine

engine = PromptEngine()

# Select prompts for a task
result = engine.select("audit this codebase for security issues")
print(result.primary)      # security-auditor
print(result.secondary)    # [codebase-sentinel]
print(result.confidence)   # 0.92
```

CLI:

```bash
python .ai/prompt-engine/engine.py list
python .ai/prompt-engine/engine.py select "refactor this function"
```

### Composition Strategies

| Strategy     | When Used                                   |
| ------------ | ------------------------------------------- |
| Sequential   | Clear phases (analyze → design → implement) |
| Parallel     | Independent concerns                        |
| Hierarchical | Primary with advisors                       |
| Single       | Focused, specific task                      |

## Self-Learning System

The system tracks effectiveness and adapts prompt selection over time.

### Feedback Loop

```text
Task → Select Prompts → Execute → Record Outcome → Update Weights
                  ↑                                      ↓
                  └──────── Improved Selection ←─────────┘
```

### Adaptation Triggers

- Success rate < 60% → demote prompt
- Success rate > 90% → promote prompt
- New task type → explore alternatives

## Tool Categories

| Category            | Tools                                             | Notes                            |
| ------------------- | ------------------------------------------------- | -------------------------------- |
| **Fully Supported** | Aider, Cursor, Windsurf, Cline, Blackbox, Augment | Auto-approve                     |
| **Supported**       | Continue, Kilo, Amazon Q, Trae, Gemini, Codex     | Auto-approve                     |
| **Context Only**    | GitHub Copilot                                    | N/A                              |
| **CLI Flag**        | Claude Code                                       | `--dangerously-skip-permissions` |

## Configuration Priority

1. `.ai/settings.yaml` - Global settings
2. `.ai/<tool>/` - Tool-specific configs
3. `.ai/rules/<tool>.rules` - Tool-specific rules
4. `.ai/context.md` - Shared project context
5. `.ai/superprompts/` - Task-specific guidance

## Adding New Superprompts

1. Create YAML in `superprompts/`:

```yaml
metadata:
  id: your-prompt-id
  triggers: ['keyword1', 'keyword2']

identity:
  role: 'Role Name'
  north_star: 'Primary objective'
```

1. Register in `prompt-engine/selector.yaml`
1. Add intent patterns with keywords and weights

## Version History

- **v2.0** (2025-12-03) - Superprompt library, self-learning engine
- **v1.0** (2025-12-02) - Initial AI configuration hub

See [`docs/ROOT_STRUCTURE_CONTRACT.md`](../docs/ROOT_STRUCTURE_CONTRACT.md) for root structure details.
