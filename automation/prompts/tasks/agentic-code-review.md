# Agentic Code Review

## Purpose

AI-powered pull request reviews with comprehensive context analysis.

## Source

AutoGen, CrewAI patterns

---

## Code Review Checklist

### 1. CORRECTNESS

- [ ] Logic errors identified
- [ ] Edge cases handled
- [ ] Null/undefined handling verified
- [ ] Type safety ensured
- [ ] Algorithm correctness validated
- [ ] Business logic matches requirements

### 2. SECURITY

- [ ] Input validation present
- [ ] Authentication/authorization correct
- [ ] SQL injection risks mitigated
- [ ] XSS vulnerabilities addressed
- [ ] CSRF protection implemented
- [ ] Sensitive data not exposed
- [ ] Secrets not hardcoded
- [ ] Dependencies have no known CVEs

### 3. PERFORMANCE

- [ ] Algorithm complexity acceptable
- [ ] Database queries optimized (no N+1)
- [ ] Memory leaks prevented
- [ ] Caching opportunities identified
- [ ] Unnecessary computations removed
- [ ] Async operations used appropriately

### 4. MAINTAINABILITY

- [ ] Code is readable and clear
- [ ] Functions are single-purpose
- [ ] Documentation adequate
- [ ] Test coverage sufficient
- [ ] Naming conventions followed
- [ ] Magic numbers eliminated
- [ ] Dead code removed

### 5. BEST PRACTICES

- [ ] Language idioms followed
- [ ] Framework patterns used correctly
- [ ] DRY principle applied
- [ ] SOLID principles followed
- [ ] Error handling comprehensive
- [ ] Logging appropriate
- [ ] Configuration externalized

---

## Review Output Format

````markdown
## Code Review Summary

### Overview

- **PR Title**: [title]
- **Files Changed**: [count]
- **Lines Added/Removed**: +[added] / -[removed]
- **Risk Level**: [Low/Medium/High/Critical]

### Critical Issues 🔴

[List of blocking issues that must be fixed]

### Warnings ⚠️

[List of issues that should be addressed]

### Suggestions 💡

[List of optional improvements]

### Positive Feedback ✅

[What was done well]

---

### Detailed Findings

#### File: `path/to/file.ts`

**Line 42-45**: 🔴 SQL Injection Risk

```typescript
// Current code
const query = `SELECT * FROM users WHERE id = ${userId}`;

// Suggested fix
const query = 'SELECT * FROM users WHERE id = $1';
const result = await db.query(query, [userId]);
```
````

**Line 78**: ⚠️ Missing Error Handling

```typescript
// Current code
const data = await fetchData();

// Suggested fix
try {
  const data = await fetchData();
} catch (error) {
  logger.error('Failed to fetch data', { error });
  throw new DataFetchError('Unable to retrieve data');
}
```

**Line 120**: 💡 Performance Improvement

```typescript
// Current: O(n²) complexity
for (const item of items) {
  if (otherItems.includes(item)) { ... }
}

// Suggested: O(n) with Set
const otherSet = new Set(otherItems);
for (const item of items) {
  if (otherSet.has(item)) { ... }
}
```

````

---

## Review Workflow

### Step 1: Context Gathering

```yaml
gather:
  - pr_description
  - linked_issues
  - file_changes
  - commit_messages
  - test_results
  - ci_status
````

### Step 2: Static Analysis

```yaml
analyze:
  - linting_results
  - type_checking
  - security_scanning
  - complexity_metrics
  - dependency_audit
```

### Step 3: Semantic Review

```yaml
review:
  - business_logic_correctness
  - architectural_alignment
  - api_contract_changes
  - breaking_changes
  - documentation_updates
```

### Step 4: Generate Feedback

```yaml
output:
  - prioritized_issues
  - specific_line_comments
  - suggested_fixes
  - approval_recommendation
```

---

## Severity Levels

| Level       | Description                            | Action Required |
| ----------- | -------------------------------------- | --------------- |
| 🔴 Critical | Security vulnerability, data loss risk | Block merge     |
| 🟠 High     | Bug, incorrect behavior                | Request changes |
| 🟡 Medium   | Code smell, maintainability issue      | Suggest fix     |
| 🟢 Low      | Style, minor improvement               | Optional        |
| 💡 Info     | Suggestion, learning opportunity       | FYI             |

---

## Integration Commands

```bash
# Run automated review
automation review pr --number 123

# Review specific files
automation review files src/api/*.ts

# Generate review report
automation review report --format markdown --output review.md
```
