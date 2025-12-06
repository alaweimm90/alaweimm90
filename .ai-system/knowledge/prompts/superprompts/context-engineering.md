# Context Engineering Prompt

## Purpose

Manage limited attention budget effectively for optimal AI performance.

## Source

Anthropic's context engineering guidelines

---

## System Prompt

### Context Management Guidelines

1. **PRIORITIZE**: Focus on high-signal information
2. **STRUCTURE**: Use XML tags to organize context
3. **MINIMIZE**: Remove redundant information
4. **REFRESH**: For long-horizon tasks, periodically compact context

---

## Context Budget Tracking

```
Current context budget: [TOKEN_BUDGET]
Used: [TOKENS_USED]
Remaining: [TOKENS_REMAINING]
Priority allocation:
  - System instructions: 10%
  - Task context: 40%
  - Examples: 20%
  - Working memory: 30%
```

---

## Structured Context Format

### XML Tag Organization

```xml
<system>
[Core instructions and persona]
</system>

<task>
[Current task description]
</task>

<constraints>
[Limitations and requirements]
</constraints>

<context>
[Relevant background information]
</context>

<examples>
[Few-shot examples if needed]
</examples>

<history>
[Relevant conversation history - summarized]
</history>
```

---

## Context Optimization Strategies

### 1. Information Density

- Remove filler words and redundancy
- Use bullet points over paragraphs
- Summarize long documents
- Extract only relevant sections

### 2. Priority Ordering

Place most important information:

- At the beginning (primacy effect)
- At the end (recency effect)
- Avoid burying critical info in the middle

### 3. Progressive Disclosure

```
Level 1: Essential context (always include)
Level 2: Supporting details (include if space)
Level 3: Nice-to-have (include if abundant space)
```

### 4. Context Refresh (Long Sessions)

For extended interactions:

```
Every N turns:
1. Summarize key decisions made
2. Compact conversation history
3. Refresh task objectives
4. Clear resolved items
```

---

## Memory Management Patterns

### Working Memory

```yaml
current_task: [active task]
key_decisions: [list of decisions]
pending_items: [todo list]
constraints: [active constraints]
```

### Long-Term Memory (External)

```yaml
user_preferences: [stored externally]
project_context: [loaded on demand]
historical_patterns: [retrieved as needed]
```

---

## Benefits

- **Maximized context utilization** within token limits
- **Reduced confusion** from irrelevant information
- **Better performance** on long-horizon tasks
- **Cost efficiency** through optimized token usage
