# Architecture Decision Records (ADRs)

**Purpose:** Document significant architectural decisions in the project

## What is an ADR?

An **Architecture Decision Record (ADR)** is a document that captures an important architectural decision made along with its context and consequences.

## When to Write an ADR

Write an ADR when making decisions about:

- Technology choices (languages, frameworks, databases)
- Architectural patterns (monorepo, microservices, event-driven)
- Development practices (testing strategy, CI/CD approach)
- Infrastructure decisions (cloud provider, deployment strategy)
- Security architecture (authentication, authorization, encryption)
- Data architecture (storage, caching, message queues)

**Rule of thumb:** If the decision will be hard to reverse or significantly impacts the project, document it.

## ADR Workflow

### 1. Create a New ADR

```bash
# Create ADR file with next sequential number
NEXT_NUM=$(ls docs/architecture/decisions/*.md 2>/dev/null | wc -l | xargs printf "%04d")
ADR_FILE="docs/architecture/decisions/$NEXT_NUM-[short-title].md"

# Copy template
cp docs/architecture/decisions/0000-template.md "$ADR_FILE"

# Edit the ADR
code "$ADR_FILE"  # or your preferred editor
```

### 2. Fill Out the Template

- **Title:** Short noun phrase describing the decision
- **Status:** Proposed, Accepted, Deprecated, Superseded
- **Context:** What forces are at play? What are the constraints?
- **Decision:** What did we decide? Be specific and concrete
- **Consequences:** What are the positive and negative outcomes?

### 3. Review Process

```bash
# Create branch
git checkout -b adr/[short-title]

# Commit ADR
git add docs/architecture/decisions/*.md
git commit -m "docs(adr): add ADR-XXXX [short-title]"

# Push and create PR
git push -u origin HEAD
gh pr create --title "ADR-XXXX: [short-title]" \
             --body "Architectural decision for [description]"
```

**Review Requirements:**

- Minimum 1 approval from technical lead
- Allow 2-3 days for discussion
- Address all comments before merging

### 4. Update Status

When an ADR is superseded or deprecated:

```markdown
## Status

~~Accepted~~ → **Superseded by [ADR-0005](./0005-new-approach.md)**

or

~~Accepted~~ → **Deprecated** (as of 2025-12-04)
Reason: [Brief explanation]
```

## ADR Numbering

- Use 4-digit sequential numbering: `0001`, `0002`, `0003`, etc.
- `0000-template.md` is reserved for the template
- Number is assigned when the ADR is created, not when accepted
- Gaps in numbering are okay if ADRs are rejected

## ADR Naming Convention

Format: `NNNN-short-kebab-case-title.md`

**Examples:**

- `0001-use-typescript-for-cli-tools.md`
- `0002-adopt-conventional-commits.md`
- `0003-implement-technical-debt-gate.md`
- `0004-use-github-actions-for-ci.md`

## Existing ADRs

| ADR                                                  | Title                              | Status   | Date       |
| ---------------------------------------------------- | ---------------------------------- | -------- | ---------- |
| [0000](./0000-template.md)                           | Template                           | Template | -          |
| [0001](./0001-use-adr-for-architecture-decisions.md) | Use ADR for Architecture Decisions | Accepted | 2025-12-04 |

---

## Tips for Writing Good ADRs

### Be Concise

- Keep ADRs short (1-2 pages)
- Focus on the decision, not implementation details
- Link to external docs for deeper technical details

### Focus on "Why"

- Explain the problem being solved
- Document alternatives considered
- Justify why this approach was chosen

### Be Honest About Trade-offs

- No decision is perfect
- Document negative consequences
- Explain why the trade-off is acceptable

### Use Clear Language

- Avoid jargon when possible
- Define technical terms
- Write for future team members who lack context

## Example Decision Process

```
Problem: We need to decide how to manage database migrations

Context:
- Using PostgreSQL for production
- Multiple environments (dev, staging, prod)
- Need version control for schema changes
- Team of 3-5 developers

Alternatives Considered:
1. Manual SQL scripts → Risky, error-prone
2. Flyway → Java dependency, heavyweight
3. Liquibase → XML configuration, complex
4. golang-migrate → Simple, CLI-based, version controlled

Decision: Use golang-migrate

Consequences:
+ Lightweight and fast
+ Works well with git workflow
+ Simple up/down migration model
- Less feature-rich than Liquibase
- Requires manual rollback planning
```

This becomes **ADR-0012: Use golang-migrate for database migrations**

---

## Tools

### List All ADRs

```bash
# List all ADRs with status
grep -r "^## Status" docs/architecture/decisions/*.md | \
  sed 's/.*\/\([0-9]*\)-\(.*\)\.md:## Status/\1: \2/' | \
  column -t -s ':'
```

### Find ADRs by Status

```bash
# Find all accepted ADRs
grep -l "Status.*Accepted" docs/architecture/decisions/*.md

# Find deprecated ADRs
grep -l "Status.*Deprecated\|Superseded" docs/architecture/decisions/*.md
```

### Generate ADR Index

```bash
# Auto-generate table of contents (updates this README)
# TODO: Create script to auto-update ADR index
```

---

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Michael Nygard's ADR Template](https://github.com/joelparkerhenderson/architecture-decision-record)
- [When to Write an ADR](https://github.com/joelparkerhenderson/architecture-decision-record#when-to-write-an-adr)
- [ADR Tools](https://github.com/npryce/adr-tools)
