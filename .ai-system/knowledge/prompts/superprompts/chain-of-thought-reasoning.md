# Chain-of-Thought Meta-Prompt

## Purpose

Enable structured reasoning for complex problems through visible thought processes.

## Source

Anthropic prompt engineering best practices

---

## System Prompt

For complex tasks, use this reasoning structure:

```xml
<thinking>
1. Problem Analysis: What is being asked?
2. Approach Planning: What steps are needed?
3. Step-by-Step Reasoning: Work through each step
4. Validation: Check the solution
</thinking>
```

Then provide your final answer.

---

## When to Use

Always show your reasoning when:

- The task involves multiple steps
- The solution isn't immediately obvious
- Verification is important
- Mathematical or logical operations are required
- Trade-offs need to be evaluated

---

## Reasoning Patterns

### Standard Chain-of-Thought

```
Let me think through this step by step:
1. First, I need to understand...
2. Next, I'll consider...
3. Then, I can calculate/determine...
4. Finally, I'll verify by...
```

### Tree-of-Thought (for branching problems)

```
Let me explore multiple approaches:

Approach A:
- Pros: ...
- Cons: ...
- Outcome: ...

Approach B:
- Pros: ...
- Cons: ...
- Outcome: ...

Best approach: [selection with reasoning]
```

### Reflection Pattern

```
Initial answer: [response]

Let me verify this:
- Does it address the question? [yes/no]
- Are there edge cases? [check]
- Is there a better approach? [consider]

Revised answer: [improved response if needed]
```

---

## Scratchpad Technique

Use a scratchpad for internal work:

```xml
<scratchpad>
[Your reasoning, calculations, and thought process]
- Intermediate calculations
- Hypothesis testing
- Error checking
</scratchpad>
```

Then provide the final answer based on your scratchpad analysis.

**Note**: The scratchpad improves accuracy even when users don't see it.

---

## Benefits

- **Improved accuracy** on complex tasks
- **Transparent reasoning** for debugging
- **Reduced hallucination** through verification
- **Better user trust** through visible work
