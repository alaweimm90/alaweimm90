# AI Tool Profiles - Best-For & Typical Use

Comprehensive profiles for all 13 configured AI coding assistants.

---

## Aider

**Config:** `.aider.conf.yml`

**Best-For:**

- Git-centric workflows with auto-commits
- Batch file modifications across codebase
- Headless/CLI autonomous operations
- Pair programming with version control

**Typical Use:**

```bash
# Refactor across multiple files with auto-commit
aider --yes-always --auto-commits "Rename all instances of getUserData to fetchUserProfile"

# Add feature with context
aider --read docs/adr/*.md "Implement the caching strategy from ADR-003"
```

**Strengths:** True `--yes-always` autonomy, git integration, context file reading

**Limitations:** CLI-only, no GUI preview

---

## Cursor

**Config:** `.cursor/settings.json`, `.cursorrules`

**Best-For:**

- Inline code completion during active development
- Quick edits with visual diff preview
- Tab-completion workflow acceleration
- Real-time pair programming feel

**Typical Use:**

- Writing new functions with Tab completion
- Composer for multi-file changes
- Cmd+K for inline transformations
- Chat for explanations while coding

**Strengths:** Fastest completion, native IDE feel, Composer mode

**Limitations:** Requires Cursor IDE (VS Code fork)

---

## Windsurf

**Config:** `.windsurfrules`

**Best-For:**

- Flow-state coding with Cascade agent
- Large-scale refactoring with previews
- Teams wanting shared AI context
- Projects requiring deep codebase understanding

**Typical Use:**

- Cascade for autonomous multi-step tasks
- Supercomplete for intelligent completions
- Flows for repeatable AI workflows

**Strengths:** Cascade autonomous agent, flow state optimization

**Limitations:** Separate IDE installation required

---

## Claude Code

**Config:** `.claude/`

**Best-For:**

- Complex reasoning tasks requiring deep analysis
- Large context window operations (200K tokens)
- Security-sensitive code review
- Architecture decisions and planning

**Typical Use:**

```bash
# Full autonomous mode
claude --dangerously-skip-permissions

# Targeted task
claude -p "Analyze security vulnerabilities in auth module"
```

**Strengths:** Best reasoning, largest context, Anthropic safety

**Limitations:** CLI flag required for auto-approve

---

## Cline

**Config:** `.cline/settings.json`, `.clinerules`

**Best-For:**

- Full-stack debugging with browser control
- End-to-end testing automation
- Tasks requiring browser + terminal + editor
- Visual regression testing

**Typical Use:**

- Debug React app while inspecting DevTools
- Run tests and fix failures iteratively
- Browser automation for E2E scenarios

**Strengths:** Browser integration, MCP support, full autonomy

**Limitations:** Resource intensive with browser

---

## GitHub Copilot

**Config:** `.github/copilot-instructions.md`

**Best-For:**

- Inline suggestions during typing
- Boilerplate code generation
- Test case generation
- Documentation comments

**Typical Use:**

- Ghost text completions while typing
- `/tests` to generate unit tests
- `/doc` to add documentation
- Chat for explanations

**Strengths:** Ubiquitous, well-integrated, trained on GitHub

**Limitations:** No auto-approve (Microsoft policy)

---

## Continue

**Config:** `.continue/config.json`

**Best-For:**

- Open-source AI coding in VS Code
- Custom model configuration
- Privacy-focused development
- Local LLM integration

**Typical Use:**

- Chat with codebase context
- Custom slash commands
- Model switching mid-conversation

**Strengths:** Open source, model agnostic, extensible

**Limitations:** Experimental auto-approve

---

## Kilo Code

**Config:** `.kilocode/config.yaml`

**Best-For:**

- Careful, methodical code changes
- Code review assistance
- Learning/educational contexts
- Compliance-heavy environments

**Typical Use:**

- Step-by-step refactoring with explanations
- Code review with suggestions
- Learning new codebases

**Strengths:** Thoughtful approach, good explanations

**Limitations:** Slower than YOLO tools

---

## Amazon Q

**Config:** `.amazonq/settings.json`

**Best-For:**

- AWS infrastructure and services
- CDK/CloudFormation generation
- AWS SDK usage patterns
- Cloud architecture design

**Typical Use:**

- Generate Lambda functions
- Create CDK constructs
- Debug AWS SDK errors
- Design serverless architectures

**Strengths:** AWS expertise, security scanning

**Limitations:** AWS-focused, limited general coding

---

## Trae

**Config:** `.trae/config.json`

**Best-For:**

- Visual development workflows
- Design-to-code transitions
- UI/UX focused development
- Component-based architectures

**Typical Use:**

- Convert designs to code
- Generate UI components
- Style and layout assistance

**Strengths:** Visual understanding, UI focus

**Limitations:** Newer tool, evolving features

---

## Blackbox

**Config:** `.blackbox/config.json`

**Best-For:**

- Maximum speed YOLO development
- Rapid prototyping
- Hackathon-style coding
- Quick script generation

**Typical Use:**

```bash
# Full YOLO mode
blackbox --yolo "Create a REST API for user management"
```

**Strengths:** `yoloMode: true`, fastest prototyping

**Limitations:** May need review for production code

---

## Gemini

**Config:** `.gemini/settings.json`

**Best-For:**

- Multimodal tasks (images + code)
- Long context analysis (1M tokens)
- Research and documentation
- Cross-language translation

**Typical Use:**

- Analyze screenshots to generate code
- Process large codebases
- Generate documentation from diagrams

**Strengths:** Multimodal, massive context window

**Limitations:** Google ecosystem integration

---

## Codex

**Config:** `.codex/config.json`

**Best-For:**

- Pure code generation tasks
- Algorithm implementation
- Code translation between languages
- Completing partial implementations

**Typical Use:**

- Generate algorithms from descriptions
- Translate Python to JavaScript
- Complete function implementations

**Strengths:** Code-optimized, fast generation

**Limitations:** Less reasoning than chat models

---

## Workflow Combinations

### Speed-Optimized Pipeline

```text
Blackbox (prototype) → Cursor (refine) → Copilot (document)
```

**Use when:** Deadline pressure, POC development

### Quality-Optimized Pipeline

```text
Claude (plan) → Aider (implement) → Cline (test) → Kilo (review)
```

**Use when:** Production code, compliance requirements

### Full-Stack Pipeline

```text
Gemini (design) → Cursor (frontend) → Claude (backend) → Amazon Q (deploy)
```

**Use when:** End-to-end feature development

### Debug Pipeline

```text
Cline (reproduce) → Claude (analyze) → Aider (fix) → Cline (verify)
```

**Use when:** Complex bug investigation

---

## Selection Decision Tree

```text
START
  │
  ├─ Need browser control? → Cline
  │
  ├─ AWS infrastructure? → Amazon Q
  │
  ├─ Maximum speed? → Blackbox
  │
  ├─ Complex reasoning? → Claude Code
  │
  ├─ Git auto-commits? → Aider
  │
  ├─ IDE completion? → Cursor or Copilot
  │
  ├─ Multimodal input? → Gemini
  │
  └─ Default → Cursor + Claude combination
```
