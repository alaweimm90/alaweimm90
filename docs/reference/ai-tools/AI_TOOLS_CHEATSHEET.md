# AI TOOLS AUTO-APPROVE CHEATSHEET

## 🚀 COMMANDS TO USE

| Tool         | Command         | Flag Used      | Status            |
| ------------ | --------------- | -------------- | ----------------- |
| **Cline**    | `cline-auto`    | `--yolo`       | ✅ Verified       |
| **Kilo**     | `kilo-auto`     | `--auto`       | ✅ Verified       |
| **Aider**    | `aider-auto`    | `--yes-always` | ✅ Verified       |
| **Blackbox** | `blackbox-auto` | `--yolo`       | ⚠️ User-specified |

## 📝 EXAMPLES

```bash
# Run Cline in YOLO mode
cline-auto "Create a snake game in python"

# Run Kilo autonomously
kilo-auto "Refactor this directory"

# Run Aider without prompts
aider-auto
```

## 🔧 TROUBLESHOOTING

If a command isn't found, run:

```bash
source ~/.bashrc
```

## 📚 STRATEGY & WORKFLOWS

For a deep dive into how to combine these tools for maximum efficiency (e.g., Architect-Builder pipelines), see:
`cat ~/AI_WORKFLOW_STRATEGY.md`

## 🤖 AUTOMATED WORKFLOWS WITH ORCHESTRATION

### Multi-LLM Orchestration Workflow

This workflow demonstrates how to orchestrate multiple LLMs for different contexts and purposes, ensuring optimal performance and accuracy.

#### Workflow: Context-Aware Multi-LLM Pipeline

1. **Objective**: Use specialized LLMs for distinct tasks in a development pipeline.
2. **Tools Involved**:
   - **Cline**: For executing system-level tasks and managing dependencies.
   - **Kilo**: For architectural planning and high-level design.
   - **Aider**: For code refactoring and bug fixes.
   - **Blackbox**: For generating boilerplate code and quick snippets.

#### Steps:

1. **Architectural Planning**:

   ```bash
   kilo-auto --mode architect "Design a scalable microservices architecture for an e-commerce platform" > architecture_plan.md
   ```

2. **Boilerplate Generation**:

   ```bash
   blackbox-auto "Generate a Dockerfile and Kubernetes manifests for the architecture" > deployment_files/
   ```

3. **Implementation**:

   ```bash
   cline-auto "Set up the project structure and install dependencies based on architecture_plan.md"
   ```

4. **Code Refinement**:

   ```bash
   aider-auto "Refactor the generated code to follow best practices and add unit tests"
   ```

5. **Validation and Testing**:

   ```bash
   cline-auto "Run the test suite and report any issues"
   ```

6. **Iterative Fixes**:
   ```bash
   aider-auto "Fix the issues reported in the test suite"
   ```

#### Best Practices:

- **Context Switching**: Use the right tool for the right task to minimize errors and maximize efficiency.
- **Validation**: Always validate outputs at each step to ensure correctness.
- **Automation**: Integrate these steps into a CI/CD pipeline for continuous delivery.

By leveraging the strengths of each tool, this workflow ensures a seamless and efficient development process tailored to specific contexts and purposes.
By leveraging the strengths of each tool, this workflow ensures a seamless and efficient development process tailored to specific contexts and purposes.

## 🧠 ORCHESTRATION ENGINE

### Multi-LLM Routing Architecture

The orchestration engine intelligently routes tasks to the most suitable LLM based on context, purpose, and performance requirements.

## 🧠 ORCHESTRATION ENGINE

### Multi-LLM Routing Architecture

The orchestration engine intelligently routes tasks to the most suitable LLM based on context, purpose, and performance requirements.

#### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION ENGINE                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Kilo       │  │   Cline      │  │   Aider      │           │
│  │ (Architect)  │  │   (Builder)  │  │   (Fixer)    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         △                 △                 △                    │
│         │                 │                 │                    │
│  ┌──────┴─────────────────┴─────────────────┴──────┐            │
│  │          CONTEXT-AWARE ROUTER                    │            │
│  │                                                   │            │
│  │  • Analyze task requirements                     │            │
│  │  • Match to optimal tool                         │            │
│  │  • Queue execution                               │            │
│  └──────┬─────────────────┬─────────────────┬──────┘            │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Validation  │  │  Execution   │  │  Feedback    │           │
│  │    Layer     │  │    Queue     │  │    Loop      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Decision Tree: Tool Selection Logic

````
                              START
                               |
                        ┌──────┴──────┐
                        │ Task Type?  │
                        └──────┬──────┘
                         /  │  │  \
                        /   │  │   \
                       /    │  │    \
          ARCHITECT   /     │  │     \ SNIPPET
               |     /      │  │      \  |
               |    /       │  │       \ |
               ▼   /        │  │        ▼▼
            KILO  /         │  │      BLACKBOX
                 /          │  │
                /   BUILD   │  │
               /            │  │
              /             │  │ FIX/REFACTOR
             /              │  │    |
            /               │  │    |
           /                │  │    ▼
          /                 ▼  │  AIDER
         /                CLINE│
        /                      │
```ting_rules:
  - context: "architectural_planning"
    tool: "kilo"
    mode: "architect"
    priority: "high"
  - context: "system_setup"
    tool: "cline"
    mode: "react"
    priority: "high"
  - context: "code_refinement"
    tool: "aider"
    mode: "git"
    priority: "medium"
  - context: "rapid_generation"
    tool: "blackbox"
    mode: "rag"
    priority: "low"
````

#### Orchestration Command Template

````bash
orchestrate() {
  local context=$1
#### Orchestration Command Template

```bash
orchestrate() {
  local context=$1
  local prompt=$2

  case $context in
    "plan") kilo-auto --mode architect "$prompt" ;;
    "build") cline-auto "$prompt" ;;
    "fix") aider-auto "$prompt" ;;
    "snippet") blackbox-auto "$prompt" ;;
    *) echo "Unknown context"; exit 1 ;;
  esac
}

# Usage
orchestrate "plan" "Design a RESTful API for user management"
````

---

## 📖 TOOL SPECIFICATIONS & EXECUTION MODELS

### Aider (`aider-auto`) - Detailed Specification

#### Rules & Constraints

```yaml
AIDER_SPECIFICATION:
  execution_model: "Chat-to-Commit Loop"
  state_management: "Git-Based"

  input_handling:
    primary: "Natural language prompts"
    context_source: "Git history + working tree"
    max_context_window: "8k tokens"

  processing_rules:
    - Never modify untracked files without confirmation
    - Changes must be committable as atomic units
    - Always reference file paths relative to root
    - Preserve existing code style and conventions
    - Generate descriptive commit messages (Conventional Commits)

  output_requirements:
    - Code changes must be syntactically valid
    - All modifications must be git-tracked
    - Commit message format: "feat|fix|refactor|test: description"
    - Must include test coverage for significant changes

  language_understanding:
    prompt_style: "Conversational with code snippets"
    emphasis: "Git context and file relationships"
    examples_needed: "Yes, with file paths"
    tone: "Collaborative, incremental"
```

#### Workflow Execution

```
┌──────────────────────────────────────────────────────┐
│         AIDER EXECUTION WORKFLOW                     │
└────────────┬─────────────────────────────────────────┘
             │
    ┌────────▼──────────────┐
    │ 1. Parse Prompt       │
    │    • Extract intent   │
    │    • Identify files   │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 2. Build Git Context  │
    │    • Read history     │
    │    • Map dependencies │
    │    • Cache relevant   │
    │      files            │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 3. Generate Changes   │
    │    • Create patches   │
    │    • Syntax validate  │
    │    • Keep atomic      │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 4. Apply & Verify     │
    │    • Apply patches    │
    │    • Run basic linter │
    │    • Check conflicts  │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 5. Commit Changes     │
    │    • Prepare message  │
    │    • Git add files    │
    │    • Git commit       │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 6. Report Back        │
    │    • Show commit      │
    │    • List changes     │
    └────────┬──────────────┘
```

#### Language Processing

```yaml
AIDER_LANGUAGE_PROCESSING:
  primary_language: "English (conversational)"
  code_references: "File paths, function names, line numbers"

  input_patterns_understood:
    - "Refactor [file] to use [pattern]"
    - "Fix [error] in [file]"
    - "Add [feature] to [module]"
    - "Update [file] with [requirements]"
    - "Test [functionality] in [file]"

  critical_keywords:
    - "refactor": Structural code changes
    - "fix": Bug resolution
    - "add": Feature implementation
    - "remove": Code deletion
    - "test": Test creation/modification
    - "update": Content changes

  context_awareness:
    git_history: "Deep (uses git blame, log)"
    file_relationships: "Tracks imports and dependencies"
    api_consistency: "Ensures consistency with existing patterns"
```

---

### Cline (`cline-auto`) - Detailed Specification

#### Rules & Constraints

```yaml
CLINE_SPECIFICATION:
  execution_model: "ReAct Loop (Reason + Action)"
  terminal_access: "Full shell access"
  file_operations: "Read/Write entire trees"

  input_handling:
    primary: "High-level task descriptions"
    context_source: "Entire project filesystem"
    max_context_window: "12k+ tokens"
    web_access: "Yes (can browse URLs)"

  processing_rules:
    - Break tasks into sequential steps
    - Execute one action at a time
    - Capture output from each step
    - Adapt based on results
    - Install dependencies as needed
    - Can create new files and directories
    - Respects .gitignore

  output_requirements:
    - Executable in any shell (bash/zsh/pwsh)
    - Terminal commands must be idempotent where possible
    - Report success/failure clearly
    - Log all significant operations
    - Suggest next steps if incomplete

  language_understanding:
    prompt_style: "Task-oriented, imperative"
    emphasis: "What needs to be done (not how)"
    examples_needed: "Optional, task clarity more important"
    tone: "Action-focused, operational"
```

#### Workflow Execution

```
┌──────────────────────────────────────────────────────┐
│         CLINE EXECUTION WORKFLOW (ReAct Loop)        │
└────────────┬─────────────────────────────────────────┘
             │
    ┌────────▼──────────────┐
    │ 1. REASON              │
    │    • Understand task   │
    │    • Plan steps        │
    │    • Identify tools    │
    │      needed            │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 2. OBSERVE            │
    │    • Check filesystem │
    │    • Read config      │
    │    • Assess state     │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 3. ACT (Step 1)       │
    │    • Execute command  │
    │    • Create file      │
    │    • Run build        │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 4. OBSERVE OUTPUT     │
    │    • Capture stdout   │
    │    • Check exit code  │
    │    • Verify state     │
    └────────┬──────────────┘
             │
         ┌───┴──────┐
         │ Success? │
         └───┬─────┬┘
            YES  NO
             │    │
       ┌─────┘    └─────┐
       │                │
       ▼                ▼
    CONTINUE          TROUBLESHOOT
    (Loop)            & ADAPT
```

#### Language Processing

```yaml
CLINE_LANGUAGE_PROCESSING:
  primary_language: "English (imperative)"
  task_focus: "Action verbs first"

  input_patterns_understood:
    - "Create a [project type] project"
    - "Install [dependencies] and set up"
    - "Run [command] and report results"
    - "Build [component] with [tech stack]"
    - "Test [module] and fix failures"
    - "Deploy [app] to [platform]"

  critical_keywords:
    - "create": New file/project creation
    - "install": Dependency management
    - "run": Command execution
    - "build": Compilation/bundling
    - "test": Test execution
    - "deploy": Release/publish
    - "configure": Setup tasks

  context_awareness:
    filesystem: "Complete tree awareness"
    package_mgmt: "Understands npm, pip, cargo, etc."
    environment: "Detects runtime, OS, versions"
    errors: "Interprets terminal error messages"
```

---

### Kilo (`kilo-auto`) - Detailed Specification

#### Rules & Constraints

```yaml
KILO_SPECIFICATION:
  execution_model: "Role-Based Orchestration (Architect/Coder)"
  context_window: "Largest of all tools"
  analysis_depth: "Repository-wide"

  input_handling:
    primary: "Strategic questions and requirements"
    context_source: "Full codebase analysis"
    max_context_window: "16k+ tokens"

  processing_rules:
    architect_mode:
      - Think at system level
      - Propose designs before implementation
      - Identify architectural patterns
      - Plan scalability and maintainability
      - Consider trade-offs explicitly

    coder_mode:
      - Implement architectural decisions
      - Handle syntax and language specifics
      - Focus on code quality
      - Apply design patterns
      - Optimize for performance

  output_requirements:
    - Comprehensive design documents
    - Well-reasoned architectural decisions
    - Code that follows best practices
    - Documentation of trade-offs
    - Clear migration paths if refactoring

  language_understanding:
    prompt_style: "Strategic and analytical"
    emphasis: "Why and how at design level"
    examples_needed: "Yes, architectural patterns help"
    tone: "Thoughtful, comprehensive"
```

#### Workflow Execution

```
┌──────────────────────────────────────────────────────┐
│       KILO EXECUTION WORKFLOW (Dual Mode)            │
└────────────┬─────────────────────────────────────────┘
             │
    ┌────────▼──────────────┐
    │ 1. SELECT MODE        │
    │    (--mode architect)  │
    └────────┬──────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
   ARCHITECT      CODER
     MODE         MODE
      │             │
      ▼             ▼
 ┌─────────┐   ┌─────────┐
 │ 1. Deep │   │ 1. Read │
 │ Analysis│   │ Design  │
 │ Codebase│   │ Plan    │
 └────┬────┘   └────┬────┘
      │             │
      ▼             ▼
 ┌─────────┐   ┌─────────┐
 │ 2. Map  │   │ 2. Impl │
 │ Depndcy │   │ Logic   │
 │ Tree    │   │ Syntax  │
 └────┬────┘   └────┬────┘
      │             │
      ▼             ▼
 ┌─────────┐   ┌─────────┐
 │ 3. Propose
 │ Design │   │ 3. Optim
 │ Pattern│   │ Perf    │
 └────┬────┘   └────┬────┘
      │             │
      └──────┬──────┘
             │
        ┌────▼────┐
        │ Output  │
        │ Result  │
        └─────────┘
```

#### Language Processing

```yaml
KILO_LANGUAGE_PROCESSING:
  primary_language: "English (strategic)"
  reasoning_style: "Comprehensive analysis"

  input_patterns_understood:
    - "Analyze [codebase] architecture"
    - "How should we structure [feature]?"
    - "What's the best way to [implement]?"
    - "Design a system for [requirements]"
    - "Evaluate [approach] vs [alternative]"

  critical_keywords:
    - "analyze": Deep inspection
    - "design": Architectural planning
    - "structure": Organization and layout
    - "evaluate": Comparison and assessment
    - "propose": Design suggestions
    - "scalability": Growth planning
    - "maintainability": Long-term health

  context_awareness:
    full_codebase: "Yes, analyzes entire structure"
    design_patterns: "Recognizes and applies patterns"
    dependencies: "Maps full dependency graph"
    performance: "Considers computational complexity"
```

---

### Blackbox (`blackbox-auto`) - Detailed Specification

#### Rules & Constraints

```yaml
BLACKBOX_SPECIFICATION:
  execution_model: "RAG (Retrieval Augmented Generation)"
  knowledge_source: "Training data + web search"
  speed_focus: "Ultra-fast generation"

  input_handling:
    primary: "Specific code snippets/patterns"
    context_source: "Query only (no project context)"
    max_context_window: "4k tokens"

  processing_rules:
    - Retrieve similar patterns from knowledge base
    - Generate boilerplate quickly
    - Don't modify existing files
    - Output standalone snippets
    - Include basic explanations
    - Cite patterns when possible

  output_requirements:
    - Copy-paste ready code
    - Follows common conventions
    - Language-specific formatting
    - Minimal dependencies
    - Well-commented for clarity

  language_understanding:
    prompt_style: "Direct queries, pattern names"
    emphasis: "What pattern/snippet is needed"
    examples_needed: "Optional, pattern name sufficient"
    tone: "Efficient, practical"
```

#### Workflow Execution

```
┌──────────────────────────────────────────────────────┐
│      BLACKBOX EXECUTION WORKFLOW (RAG)               │
└────────────┬─────────────────────────────────────────┘
             │
    ┌────────▼──────────────┐
    │ 1. PARSE QUERY        │
    │    • Extract pattern  │
    │    • Identify language│
    │    • Detect framework │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 2. SEARCH KNOWLEDGE   │
    │    • Search training  │
    │      data             │
    │    • Rank relevance   │
    │    • Get top-N        │
    │      candidates       │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 3. AUGMENT WITH       │
    │    CONTEXT            │
    │    • Combine results  │
    │    • Add explanations │
    │    • Format cleanly   │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 4. GENERATE OUTPUT    │
    │    • Create snippet   │
    │    • Add comments     │
    │    • Include example  │
    │      usage            │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ 5. RETURN QUICKLY     │
    │    • No waiting       │
    │    • Copy-paste ready │
    └────────┬──────────────┘
```

#### Language Processing

```yaml
BLACKBOX_LANGUAGE_PROCESSING:
  primary_language: "English (query-based)"
  input_style: "Pattern names, tech stack"

  input_patterns_understood:
    - "Generate [pattern] in [language]"
    - "[Framework] [component] template"
    - "Boilerplate for [tech stack]"
    - "[Design pattern] implementation"
    - "[Tool] configuration example"

  critical_keywords:
    - "generate": Create from template
    - "template": Boilerplate code
    - "example": Sample implementation
    - "config": Configuration snippet
    - "setup": Initial setup code

  context_awareness:
    project_context: "No, query-only"
    language_detection: "From query keywords"
    framework_specific: "Yes, detects from pattern"
    best_practices: "Generic, general rules"
```

---

## 🔗 TOOL INTERACTION & CHAINING RULES

### Inter-Tool Communication Specifications

```yaml
TOOL_COMMUNICATION_MATRIX:
  Kilo_to_Cline:
    output_format: "Architecture specification (markdown)"
    expected_input: "Cline understands design docs"
    handoff_method: "File write → Cline reads"
    risk: "Cline may misinterpret abstract designs"
    mitigation: "Include concrete examples"

  Cline_to_Aider:
    output_format: "Working code files"
    expected_input: "Aider analyzes via git diff"
    handoff_method: "Commit → Aider reads git history"
    risk: "Aider may not know implementation intent"
    mitigation: "Clear commit messages"

  Aider_to_Blackbox:
    output_format: "Refactored code snippets"
    expected_input: "Blackbox generates alternatives"
    handoff_method: "Copy code → Blackbox queries"
    risk: "Context loss in isolated snippets"
    mitigation: "Include surrounding code context"

  Blackbox_to_Cline:
    output_format: "Boilerplate code"
    expected_input: "Cline integrates snippets"
    handoff_method: "Paste files → Cline modifies"
    risk: "Snippet dependencies unknown"
    mitigation: "Cline validates and fixes"
```

---

```bash
# Use Aider for debugging
aider-auto "Debug the authentication module. Run tests first, then fix failures."
```

**Characteristics**:

- Focus on error messages and stack traces
- Verify with test cases
- Ensure backward compatibility

### Optimization Context

```bash
# Use Kilo for performance analysis
kilo-auto --mode architect "Analyze performance bottlenecks in the database layer and propose optimizations."
```

**Characteristics**:

- Measure current performance
- Identify bottlenecks
- Propose trade-offs

### Security Context

```bash
# Use Aider with security constraints
aider-auto "Implement OAuth2 authentication. Ensure no hardcoded secrets. Follow OWASP Top 10."
```

**Characteristics**:

- Enforce security guidelines
- Validate against compliance standards
- Use environment variables for secrets

### Feature Development Context

```bash
# Use Cline for feature implementation
cline-auto "Implement a payment processing system with Stripe integration, including error handling and tests."
```

**Characteristics**:

- Multi-step execution
- System-level operations
- Comprehensive validation

---

## 🔄 WORKFLOW PATTERNS

### Pattern 1: Sequential Pipeline

Execute tasks one after another, with each step building on the previous.

```bash
# Step 1: Plan
kilo-auto --mode architect "Design database schema" > schema.md
## 🔄 WORKFLOW PATTERNS

### Visual Overview of All Patterns

```

┌─ SEQUENTIAL ──────────────────────────────────────────┐
│ KILO → BLACKBOX → CLINE → AIDER → CLINE (Testing) │
│ ↓ ↓ ↓ ↓ ↓ │
│ PLAN → GEN → BUILD → REF → VALIDATE │
└───────────────────────────────────────────────────────┘

┌─ PARALLEL ────────────────────────────────────────────┐
│ ┌─→ KILO (Design API) │
│ │ │
│ ┌──┼─→ BLACKBOX (Generate Schema) │
│ │ │ │
│ │ └─→ BLACKBOX (Generate UI) │
│ │ │
│ └────────→ CLINE (Integrate All) │
└───────────────────────────────────────────────────────┘

┌─ CONDITIONAL ─────────────────────────────────────────┐
│ CLINE (Run Tests) │
│ ↓ │
│ ┌───┴────────────┐ │
│ │ Tests Passed? │ │
│ └───┬────────┬───┘ │
│ YES NO │
│ ↓ ↓ │
│ DEPLOY AIDER (Fix) │
│ ↓ │
│ RE-TEST │
└───────────────────────────────────────────────────────┘

┌─ FEEDBACK LOOP ───────────────────────────────────────┐
│ ┌─────────────────────────────────┐ │
│ │ CLINE (Run Tests & Report) │ │
│ └──────────────┬──────────────────┘ │
│ ↓ │
│ ┌─────────────────────────────────┐ │
│ │ AIDER (Review & Fix Issues) │ │
│ └──────────────┬──────────────────┘ │
│ ↓ │
│ ┌────────┐ │
│ │ Iterate│ │
│ └────┬───┘ │
│ ↑ │
│ (3x max) │
└───────────────────────────────────────────────────────┘

```

### Pattern 1: Sequential Pipeline

Execute tasks one after another, with each step building on the previous.
cline-auto "Set up database migrations using schema.prisma"

# Step 4: Refine
aider-auto "Add validation and error handling to database layer"
```

### Pattern 2: Parallel Execution

Run independent tasks simultaneously to save time.

```bash
# Run in parallel
kilo-auto --mode architect "Design API endpoints" > api_design.md &
blackbox-auto "Generate API documentation template" > api_docs.md &
wait

# Combine results
cline-auto "Implement API based on design and docs"
```

### Pattern 3: Conditional Branching

Route based on conditions or results.

```bash
# Check tests
cline-auto "Run test suite and save results to test_results.txt"

# Conditional fix
if grep -q "FAILED" test_results.txt; then
  aider-auto "Fix failing tests"
else
  echo "All tests passed!"
fi
```

### Pattern 4: Feedback Loop

Iteratively refine outputs.

```bash
iteration=0
max_iterations=3

while [ $iteration -lt $max_iterations ]; do
  cline-auto "Run tests and generate report"
  aider-auto "Review report and fix issues"
  ((iteration++))
done
```

---

## ⚠️ ERROR HANDLING & RECOVERY

### Failure Detection

```bash
run_with_fallback() {
  local primary_tool=$1
  local fallback_tool=$2
  local prompt=$3
## ⚠️ ERROR HANDLING & RECOVERY

### Error Recovery Flow Diagram

```

┌──────────────────────────────────────────────────────┐
│ TASK EXECUTION START │
└────────────────┬─────────────────────────────────────┘
│
▼
┌──────────────┐
│ Try Primary │
│ Tool │
└──────┬───────┘
│
┌───────┴────────┐
│ Success? │
└───┬────────┬───┘
YES NO
│ │
▼ ▼
SUCCESS ┌──────────────┐
│ Retry Count? │
└──────┬───────┘
│
┌─────┴──────┐
│ < Max? │
└───┬────┬───┘
YES NO
│ │
▼ ▼
WAIT TRY FALLBACK
│ │
└─┬─────┘
▼
┌──────────────┐
│ Fallback OK? │
└──────┬───────┘
│
┌──────┴──────┐
│ Success? │
└───┬────┬────┘
YES NO
│ │
▼ ▼
SUCCESS ALERT
(Manual Review)

````

### Failure Detection

```bash
run_with_fallback() {
  local primary_tool=$1
  local fallback_tool=$2
  local prompt=$3

  if $primary_tool "$prompt"; then
    echo "Primary tool succeeded"
  else
    echo "Primary tool failed. Using fallback..."
    $fallback_tool "$prompt"
  fi
}

# Usage
run_with_fallback "cline-auto" "aider-auto" "Implement user authentication"
```  while [ $retry_count -lt $max_retries ]; do
    if $tool "$prompt"; then
      return 0
    fi
    ((retry_count++))
    sleep $((2 ** retry_count))
  done

  return 1
}
````

### Self-Healing Workflows

```bash
# Validate output and retry if needed
validate_and_fix() {
  local output=$1

  # Validate output
  if ! validate_syntax "$output"; then
    echo "Validation failed. Requesting fix..."
    aider-auto "Fix the syntax errors in the generated code"
  fi
}
```

---

## 📊 PERFORMANCE METRICS

### Key Metrics

| Metric                 | Tool           | Measurement                    |
| ---------------------- | -------------- | ------------------------------ |
| **Execution Time**     | All            | Time from prompt to completion |
| **Accuracy**           | Aider, Cline   | Percentage of correct outputs  |
| **Hallucination Rate** | Kilo, Blackbox | % of false/unsupported claims  |

## 📊 PERFORMANCE METRICS

### Workflow Performance Pipeline

```
┌────────────────────────────────────────────────────────┐
│            WORKFLOW EXECUTION PIPELINE                 │
└────────────┬─────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐      ┌──────────────┐
│  Stage  │      │ Metrics      │
│ Counter │      │ Collector    │
└────┬────┘      └──────┬───────┘
     │                  │
     └────────┬─────────┘
              ▼
        ┌──────────────┐
        │ Aggregator   │
        └──────┬───────┘
               ▼
    ┌──────────────────────┐
    │ Performance Report   │
    ├──────────────────────┤
    │ • Exec Time: 45.2s   │
    │ • Accuracy: 94%      │
    │ • Hallucin: 2%       │
    │ • Recovery: 100%     │
    │ • Context: 85%       │
    └──────────────────────┘
```

### Key Metrics

| Metric                  | Tool           | Measurement                    |
| ----------------------- | -------------- | ------------------------------ |
| **Execution Time**      | All            | Time from prompt to completion |
| **Accuracy**            | Aider, Cline   | Percentage of correct outputs  |
| **Hallucination Rate**  | Kilo, Blackbox | % of false/unsupported claims  |
| **Recovery Rate**       | Aider          | % of self-corrected errors     |
| **Context Utilization** | Kilo           | % of available context used    |

time $tool "$prompt"

echo "Success rate: $(calculate_success_rate $tool)"
echo "Average execution time: $(calculate_avg_time $tool)"
}

````

---

## 🚀 ADVANCED SCENARIOS

### Scenario 1: Full-Stack Feature Implementation

```bash
# 1. Architecture Phase
kilo-auto --mode architect "Design a user profile feature with authentication, database, and API" > feature_spec.md

# 2. Backend Implementation
## 🚀 ADVANCED SCENARIOS

### Full Development Lifecycle Flow

````

┌─────────────────────────────────────────────────────────────┐
│ FULL-STACK FEATURE IMPLEMENTATION FLOW │
└────────────────┬──────────────────────────────────────────┘
│
┌───────▼────────┐
│ KILO (Architect)
│ Design Features
└───────┬────────┘
│ ↓ feature_spec.md
┌───────▼────────┐
│ BLACKBOX │
│ Generate UI │
└───────┬────────┘
│ ↓ ui_components/
┌───────▼────────────┐
│ CLINE (Builder) │
│ Backend Setup │
└───────┬────────────┘
│ ↓ src/, migrations/
┌───────▼────────────┐
│ CLINE │
│ API Implementation │
└───────┬────────────┘
│ ↓ routes/, services/
┌───────▼────────────┐
│ CLINE │
│ Integration │
└───────┬────────────┘
│ ↓ integration.ts
┌───────▼────────────┐
│ CLINE │
│ Run Test Suite │
└───────┬────────────┘
│
┌───────┴──────────┐
│ Tests Passed? │
└───┬────────┬─────┘
YES NO
│ │
▼ ▼
READY ┌──────────────┐
│ AIDER │
│ Fix Issues │
└──────┬───────┘
│
RE-RUN TESTS

````

### Scenario 1: Full-Stack Feature Implementation

```bash
# 1. Architecture Phase
kilo-auto --mode architect "Design a user profile feature with authentication, database, and API" > feature_spec.md

# 2. Backend Implementation
cline-auto "Implement backend based on feature_spec.md with database schema and API endpoints"

# 3. Frontend Generation
blackbox-auto "Generate React components for user profile UI"

# 4. Integration & Testing
cline-auto "Set up API routes, integrate frontend, run full test suite"

# 5. Refinement
aider-auto "Optimize performance and fix any failing tests"
```er-auto "Create refactoring plan with file-by-file changes"

### Scenario 2: Large Codebase Refactoring

````

┌──────────────────────────────────────────────────────┐
│ LARGE CODEBASE REFACTORING WORKFLOW │
└──────────────┬───────────────────────────────────────┘
│
┌─────▼──────┐
│ KILO │
│ Analyze │
│ & Plan │
└─────┬──────┘
│ ↓ opportunities.md
┌─────▼──────┐
│ AIDER │
│ Create │
│ Plan │
└─────┬──────┘
│ ↓ refactor_plan.md
┌─────▼──────────────┐
│ CLINE │
│ Execute Changes │
│ (Step by Step) │
└─────┬──────────────┘
│ ↓ refactored_src/
┌─────▼──────┐
│ CLINE │
│ Run Tests │
│ & Bench │
└─────┬──────┘
│
┌─────┴────────────┐
│ Perf Improved? │
└────┬────────┬────┘
YES NO
│ │
▼ ▼
READY ┌──────────┐
│ AIDER │
│ Optimize │
└────┬─────┘
│
RE-RUN TESTS

````

```bash
# 1. Analysis
kilo-auto --mode architect "Analyze current architecture and identify refactoring opportunities"

# 2. Planning
aider-auto "Create refactoring plan with file-by-file changes"

# 3. Execution
cline-auto "Execute refactoring plan step by step"

# 4. Validation
cline-auto "Run full test suite and performance benchmarks"

# 5. Optimization
aider-auto "Fix any issues and optimize critical paths"
```ckbox-auto "Generate security best practices checklist"

### Scenario 3: Security Audit & Hardening

````

┌────────────────────────────────────────────────┐
│ SECURITY AUDIT & HARDENING WORKFLOW │
└────────────────┬─────────────────────────────┘
│
┌───────▼─────────┐
│ KILO │
│ Security Audit │
└───────┬─────────┘
│ ↓ vulnerabilities.md
┌───────▼──────────┐
│ BLACKBOX │
│ Best Practices │
└───────┬──────────┘
│ ↓ checklist.md
┌───────▼──────────┐
│ AIDER │
│ Implement Fixes │
│ (OWASP Top 10) │
└───────┬──────────┘
│ ↓ hardened_src/
┌───────▼──────────┐
│ CLINE │
│ Security Tests │
└───────┬──────────┘
│
┌───────┴──────────┐
│ Compliant? │
└────┬────────┬────┘
YES NO
│ │
▼ ▼
APPROVED ┌──────────┐
│ AIDER │
│ Refine │
└────┬─────┘
│
RE-RUN TESTS

````

```bash
# 1. Vulnerability Scan
kilo-auto "Perform security audit on the codebase"

# 2. Risk Assessment
blackbox-auto "Generate security best practices checklist"

# 3. Implementation
aider-auto "Implement security fixes following OWASP guidelines"

# 4. Verification
cline-auto "Run security tests and generate compliance report"
```kflow:
  name: "Sequential Feature Development"
  stages:
    - stage: plan
      tool: kilo
      mode: architect
      prompt: "Design the feature architecture"
    - stage: implement
      tool: cline
      mode: react
      prompt: "Implement the feature"
    - stage: refine
      tool: aider
      mode: git
      prompt: "Refactor and optimize"
    - stage: test
      tool: cline
      mode: react
      prompt: "Run comprehensive tests"
````

### Parallel Workflow Template

```yaml
workflow:
  name: "Parallel Development"
  parallel_stages:
    - task: api_design
      tool: kilo
      prompt: "Design API endpoints"
    - task: db_schema
      tool: blackbox
      prompt: "Generate database schema"
    - task: ui_components
      tool: blackbox
      prompt: "Generate UI components"
  merge_stage:
    tool: cline
    prompt: "Integrate all components"
```

### Conditional Workflow Template

```yaml
workflow:
  name: "Test-Driven Development"
  steps:
    - run_tests:
        tool: cline
        prompt: "Run test suite"
        output: test_results.txt
    - conditional:
        if: "tests_failed"
        then:
          tool: aider
          prompt: "Fix failing tests"
        else:
          tool: cline
          prompt: "Deploy to production"
```

---

## 🎯 RULES & CONSTRAINTS

### No-Hallucination Rule

**Definition**: All outputs must be grounded in provided context or verifiable knowledge.

**Implementation**:

```bash
## 🎯 RULES & CONSTRAINTS

### Autonomous Workflow Validation Loop

```

┌──────────────────────────────────────────────────────┐
│ RULE ENFORCEMENT SYSTEM │
└──────────────┬───────────────────────────────────────┘
│
┌────────┴────────┐
▼ ▼
┌──────────────┐ ┌──────────────┐
│ Context │ │ Output │
│ Validation │ │ Generation │
└──────┬───────┘ └──────┬───────┘
│ │
└────────┬────────┘
▼
┌────────────────────┐
│ Rule Checker │
└────┬───────┬───┬───┘
│ │ │
┌──────▼──┐ ┌┴──┐ │
│ No Halluc│ │Self│ │
└─────┬────┘ │Refut│ │
│ └──┬──┘ │
│ │ │
┌─────▼──────────▼─┐ │
│ Skeptic Check │ │
└────────┬─────────┘ │
│ │
┌────▼────────────▼─┐
│ Claim → Prove │
└────┬──────────────┘
│
┌──────┴───────┐
│ All Pass? │
└───┬────┬─────┘
YES NO
│ │
▼ ▼
APPROVED REVISE

```

### No-Hallucination Rule

**Definition**: All outputs must be grounded in provided context or verifiable knowledge.

**Implementation**:
aider-auto "$NO_HALLUCINATION_PROMPT: $USER_PROMPT"
```

**Verification**:

- Cross-reference outputs against codebase
- Validate function/module existence
- Verify API compatibility

### Self-Refutation Rule

**Definition**: Agents must identify and correct their own errors.

**Implementation**:

```bash
# Iterative Validation
validate_and_correct() {
  local output=$1
  local max_iterations=3
  local iteration=0

  while [ $iteration -lt $max_iterations ]; do
    if validate_output "$output"; then
      return 0
    fi
    output=$(aider-auto "Review your previous output and correct any errors: $output")
    ((iteration++))
  done
}
```

### Skepticism Rule

**Definition**: Question assumptions and verify claims before implementation.

**Implementation**:

```bash
# Challenge Assumptions
skeptical_prompt="
Before implementing, verify:
1. Is this approach optimal?
2. Are there edge cases?
3. What are the trade-offs?
4. Have we tested assumptions?
"

kilo-auto --mode architect "$skeptical_prompt: $TASK"
```

### Claim → Prove Rule

**Definition**: Every claim must be backed by evidence or passing tests.

**Implementation**:

```bash
claim_prove_workflow() {
  local claim=$1

  # Make claim
  echo "Claim: $claim"

  # Propose proof
  aider-auto "Create tests to prove: $claim"

  # Execute proof
  if cline-auto "Run the proof tests"; then
    echo "✓ Claim verified"
  else
    echo "✗ Claim refuted. Revising..."
    aider-auto "Revise the implementation based on test failures"
  fi
}
```

---

## 🔍 TOOL SELECTION MATRIX

| Task Type                | Best Tool | Secondary | Why                                            |
| ------------------------ | --------- | --------- | ---------------------------------------------- |
| Architectural Design     | Kilo      | Blackbox  | Kilo has superior context window and reasoning |
| Greenfield Development   | Cline     | Blackbox  | Cline excels at system setup and file creation |
| Code Refactoring         | Aider     | Cline     | Aider understands git history and dependencies |
| Bug Fixing               | Aider     | Cline     | Aider's git-centric approach aids debugging    |
| Boilerplate Generation   | Blackbox  | Cline     | Blackbox retrieves patterns quickly            |
| Performance Optimization | Kilo      | Aider     | Kilo analyzes at scale; Aider optimizes        |

## 🔍 TOOL SELECTION MATRIX

### Decision Tree for Tool Selection

```
                   ┌─ TASK RECEIVED ─┐
                   └────────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    ┌────────┐          ┌────────┐         ┌────────┐
    │ Planning│         │Building│         │ Fixing │
    │ Phase?  │         │ Phase?  │         │ Phase? │
    └────┬───┘          └────┬───┘         └────┬───┘
        │YES               │YES              │YES
        │                  │                 │
        ▼                  ▼                 ▼
    ┌─────────┐        ┌──────────┐      ┌────────┐
    │  KILO   │        │  CLINE   │      │ AIDER  │
    │Architect│        │ Task Exec│      │  Git   │
    └─────────┘        └──────────┘      │ Fixer  │
                                         └────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Output Check │
                    └──────┬───────┘
                           │
                    ┌──────┴──────┐
                    │ Quality OK? │
                    └───┬────┬────┘
                       YES  NO
                        │    │
                        ▼    ▼
                     CONTINUE RETRY
```

| Task Type                | Best Tool | Secondary | Why                                            |
| ------------------------ | --------- | --------- | ---------------------------------------------- |
| Architectural Design     | Kilo      | Blackbox  | Kilo has superior context window and reasoning |
| Greenfield Development   | Cline     | Blackbox  | Cline excels at system setup and file creation |
| Code Refactoring         | Aider     | Cline     | Aider understands git history and dependencies |
| Bug Fixing               | Aider     | Cline     | Aider's git-centric approach aids debugging    |
| Boilerplate Generation   | Blackbox  | Cline     | Blackbox retrieves patterns quickly            |
| Performance Optimization | Kilo      | Aider     | Kilo analyzes at scale; Aider optimizes        |
| Security Hardening       | Aider     | Kilo      | Aider commits security changes; Kilo verifies  |
| Testing Strategy         | Kilo      | Cline     | Kilo designs; Cline implements                 |

### Quick Reference: Use Case → Tool

```
┌─────────────────────────────────────────────────────┐
│          QUICK TOOL SELECTOR FLOWCHART              │
└─────────────────────────────────────────────────────┘

    I want to...
        │
    ┌───┼─────────────────────────────────────────┐
    │   │                                         │
    ▼   ▼                                         ▼
DESIGN? BUILD?                                FIX?
  │      │                                     │
  │      ├─ Green Field? ─┐                    │
  │      │                │ YES                │
  │      │                └─→ CLINE            │
  │      │                                     │
  │      └─ Setup Dependencies?                │
  │         └─ YES ─→ CLINE                    │
  │                                            │
  │ YES                                    YES
  └─→ KILO                            (Git History?)
      (Architect)                           │
                                       YES  │
                                      └─→ AIDER
                                      (Git Fixer)
```

```bash
# I want to...                  → Use this tool
# Design architecture          → kilo-auto --mode architect
# Set up new project           → cline-auto
# Fix bugs                      → aider-auto
# Generate templates           → blackbox-auto
# Optimize performance         → kilo-auto + aider-auto
# Implement feature            → cline-auto + aider-auto
# Audit code quality           → kilo-auto
# Write tests                  → cline-auto + aider-auto
```

# Implement feature → cline-auto + aider-auto

# Audit code quality → kilo-auto

# Write tests → cline-auto + aider-auto

````

---

## 🔗 MCP INTEGRATION

### Invoking MCPs in Workflows

#### Aider MCP Server Integration

```bash
# Start MCP Server
uv run aider-mcp-server --editor-model "gpt-4o" --current-working-dir "$PROJECT_PATH" &
MCP_PID=$!
## 🔗 MCP INTEGRATION

### Complete MCP Orchestration Architecture

````

┌─────────────────────────────────────────────────────────────────┐
│ COMPLETE MCP ORCHESTRATION SYSTEM │
└────────────┬──────────────────────────────────────────────────┘
│
┌────────┴────────┐
▼ ▼
┌──────────────┐ ┌──────────────────────────────┐
│ Local Tools │ │ MCP Servers (Distributed) │
│ │ │ │
│ • Cline │ │ ┌──────────────────────────┐ │
│ • Kilo │ │ │ Aider MCP Server │ │
│ • Aider │ │ │ • Code Refactoring │ │
│ • Blackbox │ │ │ • Test Writing │ │
│ │ │ └──────────────────────────┘ │
└──────┬───────┘ │ │
│ │ ┌──────────────────────────┐ │
│ │ │ AgentAPI MCP │ │
│ │ │ • Multi-agent Control │ │
│ │ │ • REST Orchestration │ │
│ │ └──────────────────────────┘ │
│ │ │
│ │ ┌──────────────────────────┐ │
│ │ │ Agor MCP │ │
│ │ │ • Parallel Execution │ │
│ │ │ • Workspace Management │ │
│ │ └──────────────────────────┘ │
│ └──────────────────────────────┘
│ │
└───────────┬───────┘
▼
┌──────────────────────┐
│ Unified Orchestrator │
│ │
│ • Route Tasks │
│ • Monitor Progress │
│ • Handle Failures │
└──────────┬───────────┘
│
▼
┌──────────────┐
│ Execution & │
│ Results │
└──────────────┘

```

### MCP Invocation Flow

```

┌─────────────────────────────────────────────────────┐
│ MCP TASK EXECUTION PIPELINE │
└────────────┬─────────────────────────────────────────┘
│
┌──────▼──────┐
│ Initialize │
│ MCP Server │
└──────┬──────┘
│ ┌─────────────────────┐
└→│ Port Assignment │
│ Config Loading │
│ Auth Setup │
└─────────┬───────────┘
│
┌──────▼──────────┐
│ Register Tools │
│ with Server │
└──────┬──────────┘
│
┌──────────┴──────────┐
▼ ▼
┌──────────────┐ ┌──────────────┐
│ Route Task │ │ Get Config │
│ to MCP │ │ Parameters │
└──────┬───────┘ └──────┬───────┘
│ │
└──────────┬────────┘
▼
┌──────────────────────┐
│ Execute on Remote │
│ MCP Server │
└──────┬───────────────┘
│
┌──────┴───────────┐
│ Success? │
└───┬────────┬─────┘
YES NO
│ │
▼ ▼
RETURN ERROR
RESULT HANDLE
&RETRY

````

### Invoking MCPs in Workflows

#### Aider MCP Server Integration

```bash
# Start MCP Server
uv run aider-mcp-server --editor-model "gpt-4o" --current-working-dir "$PROJECT_PATH" &
MCP_PID=$!

# Use Aider via MCP for orchestrated task
aider-mcp-client "Refactor auth module"

# Stop MCP Server
kill $MCP_PID
````

#### AgentAPI MCP for Multi-Agent Orchestration

```bash
# Start AgentAPI server
agentapi server --port 3284 --log-level debug &
AGENT_PID=$!

# Route task to appropriate agent via REST API
curl -X POST http://localhost:3284/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "context": "architecture",
    "tool": "kilo",
    "prompt": "Design microservices architecture",
    "mode": "architect"
  }'

kill $AGENT_PID
```

#### Agor MCP for Parallel Workflows

```bash
# Start Agor daemon
agor daemon start &

# Execute parallel tasks across git worktrees
agor task run --parallel \
  --task1 "kilo-auto --mode architect 'Design API'" \
  --task2 "blackbox-auto 'Generate schema'" \
  --task3 "cline-auto 'Set up project'"
```

### MCP-Based Orchestration Workflow

```
┌───────────────────────────────────────────────────────────┐
│    COMPLETE MCP-BASED ORCHESTRATION WORKFLOW              │
└────────────┬──────────────────────────────────────────────┘
             │
    ┌────────▼───────┐
    │ START WORKFLOW │
    └────────┬───────┘
             │
    ┌────────▼──────────────────────┐
    │ Initialize MCPs               │
    │ ├─ Aider MCP Server           │
    │ ├─ AgentAPI MCP Server        │
    │ └─ Agor MCP Daemon            │
    └────────┬──────────────────────┘
             │
    ┌────────▼──────────────────────┐
    │ Route Task to Optimal Agent   │
    │ (via AgentAPI Orchestrator)   │
    └────────┬──────────────────────┘
             │
      ┌──────┴───────────────────────┐
      │                              │
    DESIGN?                        BUILD?
      │                              │
      ▼                              ▼
 ┌────────┐                   ┌────────────┐
 │ Call   │                   │ Call Agor  │
 │ Kilo   │                   │ for Parallel│
 │ MCP    │                   │ Execution  │
 └───┬────┘                   └────┬───────┘
     │                             │
     └──────────┬──────────────────┘
                │
        ┌───────▼────────┐
        │ Monitor Results│
        │ via Aider MCP  │
        └───────┬────────┘
                │
        ┌───────▼─────────────┐
        │ Aggregate Results   │
        │ & Status            │
        └───────┬─────────────┘
                │
        ┌───────▼──────┐
        │ Return Output│
        │ to User      │
        └──────────────┘
```

```yaml
mcp_workflow:
  name: "Orchestrated Development Pipeline"
  mcps:
    - name: "aider-mcp"
      endpoint: "aider-mcp-server"
      port: 5000
      tasks:
        - refactor_code
        - fix_bugs
        - write_tests
    - name: "agentapi-mcp"
      endpoint: "localhost:3284"
      tasks:
        - route_to_optimal_tool
        - multi_agent_coordination
        - task_scheduling
    - name: "agor-mcp"
      endpoint: "agor-daemon"
      tasks:
        - parallel_execution
        - workspace_management
        - git_worktree_control
  execution_strategy: "intelligent_routing"
  fallback_strategy: "cascade"
  monitoring: "realtime"
```

---

## 📈 COMPLETE WORKFLOW LIFECYCLE

```
┌─────────────────────────────────────────────────────────┐
│       FROM TASK TO DEPLOYMENT LIFECYCLE                │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────▼────────┐
    │ 1. INPUT TASK   │
    └────────┬────────┘
             │
    ┌────────▼─────────────────┐
    │ 2. ANALYZE CONTEXT       │
    │    • Read Requirements    │
    │    • Check Codebase       │
    │    • Assess Complexity    │
    └────────┬─────────────────┘
             │
    ┌────────▼──────────────────┐
    │ 3. ROUTE TO TOOL          │
    │    (Decision Tree)        │
    │    • Architecture → Kilo  │
    │    • Build → Cline        │
    │    • Fix → Aider          │
    │    • Snippet → Blackbox   │
    └────────┬──────────────────┘
             │
    ┌────────▼──────────────────┐
    │ 4. EXECUTE TASK           │
    │    • Invoke Tool/MCP      │
    │    • Monitor Progress     │
    │    • Capture Output       │
    └────────┬──────────────────┘
             │
    ┌────────▼────────────────────┐
    │ 5. VALIDATE OUTPUT          │
    │    • No Hallucination?      │
    │    • Self-Refuted Errors?   │
    │    • Claim → Proof?         │
    │    • Quality Pass?          │
    └────────┬────────────────────┘
             │
      ┌──────┴───────────┐
      │ Quality OK?      │
      └───┬────────┬─────┘
         YES      NO
          │        │
          ▼        ▼
       ┌─────┐  ┌──────────────────┐
       │6.   │  │ RETRY/REFINE     │
       │TEST │  │ • Feed back to   │
       │     │  │   Aider/Kilo     │
       │     │  │ • Max 3 Retries  │
       └──┬──┘  └────────┬─────────┘
          │               │
          │       ┌───────┘
          │       │
    ┌─────▼───────▼────┐
    │ 7. FINAL REVIEW  │
    │    • All Tests OK│
    │    • Metrics OK  │
    │    • Ready?      │
    └────────┬─────────┘
             │
    ┌────────▼──────────┐
    │ 8. COMMIT & PUSH  │
    │    • Git Commit   │
    │    • CI/CD Trigger│
    └────────┬──────────┘
             │
    ┌────────▼──────────┐
    │ 9. DEPLOY/OUTPUT  │
    │    • Production   │
    │    • Report       │
    └────────┬──────────┘
             │
    ┌────────▼──────┐
    │ ✓ COMPLETE    │
    └───────────────┘
```

---
