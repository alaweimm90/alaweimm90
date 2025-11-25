# AI Agent Conventions & Task Routing

**Purpose**: Enforce consistent style across different LLMs and optimize subscription usage.

---

## 🎯 Task-to-Model Routing Matrix

### High-Value Tasks → Claude Sonnet 4.5 (Limited Subscription)

**When to use**: Complex reasoning, architecture decisions, critical refactoring

```yaml
use_claude_sonnet_4_5_for:
  - Architecture design and system planning
  - Complex refactoring with multi-file changes
  - Critical bug fixing requiring deep analysis
  - Security-sensitive code reviews
  - Performance optimization strategies
  - Database schema design
  - API contract design
  - Complex algorithm implementation

avoid_claude_for:
  - Simple code formatting
  - Boilerplate generation
  - Routine documentation updates
  - Simple bug fixes
  - Code completion
```

**Prompt prefix for Claude**:
```
CONTEXT: Multi-organization solo dev project
STYLE: Follow .metaHub/conventions/CODING_STYLE.md
CONSTRAINTS: TypeScript strict mode, functional patterns preferred
OUTPUT: Provide reasoning, then code with inline comments
```

---

### Quick Edits → Cursor AI (Moderate Subscription)

**When to use**: Fast iterations, boilerplate, quick fixes

```yaml
use_cursor_for:
  - Boilerplate code generation (CRUD, routes, models)
  - Quick bug fixes (<10 lines changed)
  - Adding tests for existing functions
  - Simple refactoring (rename, extract function)
  - Configuration file updates
  - Environment setup scripts

avoid_cursor_for:
  - Architecture decisions
  - Complex logic requiring deep thought
  - Security-critical code
```

**Prompt prefix for Cursor**:
```
TASK: [specific, narrow task]
STYLE: Match existing code style in this file
OUTPUT: Code only, minimal explanation
```

---

### Research & Documentation → Windsurf (Moderate Subscription)

**When to use**: Exploration, documentation, learning

```yaml
use_windsurf_for:
  - Codebase exploration and understanding
  - Writing comprehensive documentation
  - Research on libraries/frameworks
  - Generating README files
  - Creating architecture diagrams (Mermaid)
  - Analyzing dependencies
  - Technical writing

avoid_windsurf_for:
  - Writing production code
  - Quick fixes
  - Code completion
```

**Prompt prefix for Windsurf**:
```
GOAL: [research objective]
AUDIENCE: Solo developer, multi-org context
OUTPUT: Markdown with diagrams, actionable insights
```

---

### Inline Suggestions → GitHub Copilot (Unlimited)

**When to use**: Real-time coding assistance

```yaml
use_copilot_for:
  - Auto-completion while typing
  - Generating test cases
  - Writing repetitive code patterns
  - Function implementations from docstrings
  - Type definitions
  - Simple transformations

this_is_your_workhorse:
  - Use liberally for all routine coding
  - No subscription limits
  - Best for "known patterns"
```

---

## 📝 Universal Coding Style Rules

All AI agents MUST follow these rules:

### Code Style

```typescript
// ✅ DO: Functional, immutable patterns
const processUsers = (users: User[]): ProcessedUser[] =>
  users
    .filter(u => u.isActive)
    .map(u => ({ ...u, processed: true }));

// ❌ DON'T: Imperative, mutating patterns
function processUsers(users) {
  for (let i = 0; i < users.length; i++) {
    if (users[i].isActive) {
      users[i].processed = true; // mutation!
    }
  }
  return users;
}

// ✅ DO: Explicit types
function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0);
}

// ❌ DON'T: Implicit any
function calculateTotal(items) { // implicit any
  return items.reduce((sum, item) => sum + item.price, 0);
}

// ✅ DO: Early returns, guard clauses
function validateUser(user: User | null): ValidationResult {
  if (!user) return { valid: false, reason: 'User not found' };
  if (!user.email) return { valid: false, reason: 'Email required' };
  return { valid: true };
}

// ❌ DON'T: Nested conditionals
function validateUser(user) {
  if (user) {
    if (user.email) {
      return { valid: true };
    } else {
      return { valid: false, reason: 'Email required' };
    }
  } else {
    return { valid: false, reason: 'User not found' };
  }
}
```

### Naming Conventions

```typescript
// Files
user-service.ts        // kebab-case for files
UserRepository.ts      // PascalCase for classes

// Variables & Functions
const userId = '123';           // camelCase
const MAX_RETRIES = 3;          // SCREAMING_SNAKE for constants
function getUserById() {}       // camelCase, verb prefix

// Types & Interfaces
interface User {}               // PascalCase
type UserId = string;           // PascalCase
enum UserRole {}                // PascalCase

// Components (React)
const UserProfile = () => {};   // PascalCase

// Private/Internal
const _internalHelper = () => {}; // underscore prefix
```

### File Structure

```typescript
// ✅ DO: Organize imports
// 1. External dependencies
import { useState } from 'react';
import express from 'express';

// 2. Internal absolute imports
import { UserService } from '@/services/user-service';

// 3. Relative imports
import { formatDate } from '../utils/date';
import type { User } from '../types';

// 4. Styles (if applicable)
import styles from './Component.module.css';

// ❌ DON'T: Mix import styles randomly
import styles from './Component.module.css';
import { formatDate } from '../utils/date';
import express from 'express';
```

### Error Handling

```typescript
// ✅ DO: Typed errors, explicit handling
class UserNotFoundError extends Error {
  constructor(userId: string) {
    super(`User ${userId} not found`);
    this.name = 'UserNotFoundError';
  }
}

async function getUser(id: string): Promise<User> {
  const user = await db.users.findById(id);
  if (!user) throw new UserNotFoundError(id);
  return user;
}

// Usage
try {
  const user = await getUser('123');
} catch (error) {
  if (error instanceof UserNotFoundError) {
    // Handle specifically
  } else {
    // Handle unexpected
  }
}

// ❌ DON'T: Generic error handling
async function getUser(id) {
  try {
    return await db.users.findById(id);
  } catch (e) {
    console.log('Error:', e); // loses context
    return null; // swallows error
  }
}
```

---

## 📄 Writing Style for Documentation

### Commit Messages

```bash
# Format: <type>(<scope>): <description>

# ✅ DO:
feat(user-service): add email verification flow
fix(api): handle null response in getUserById
docs(readme): add Docker setup instructions
refactor(auth): extract token validation to middleware
test(user-repo): add integration tests for CRUD operations

# ❌ DON'T:
fixed bug
updated files
changes
wip
```

### README Structure

```markdown
# Project Name

Brief one-liner description.

## Quick Start

```bash
# Absolute minimum to get running
docker-compose up
```

## Prerequisites

- Node.js 20+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)

## Installation

[Detailed steps]

## Usage

[Examples]

## Architecture

[High-level overview with diagram]

## Development

[How to contribute, run tests, etc.]

## License

[License info]
```

### Code Comments

```typescript
// ✅ DO: Explain WHY, not WHAT
// Using exponential backoff because API rate limits reset after 60s
const retryDelay = Math.min(1000 * Math.pow(2, attempt), 60000);

// Cache for 5 minutes to reduce database load during high traffic
const CACHE_TTL = 5 * 60 * 1000;

// ❌ DON'T: State the obvious
// Set retry delay
const retryDelay = Math.min(1000 * Math.pow(2, attempt), 60000);

// Cache TTL is 5 minutes
const CACHE_TTL = 5 * 60 * 1000;
```

---

## 🤖 Agent-Specific Instructions

### For Claude Code

```
You are working on a multi-organization solo developer project.

CRITICAL RULES:
1. Always check .metaHub/conventions/ for style guides
2. Use TypeScript strict mode
3. Prefer functional patterns over imperative
4. Include Mermaid diagrams for architecture changes
5. Provide reasoning before code
6. Consider cross-organization impact

OUTPUT FORMAT:
1. Analysis (what/why)
2. Mermaid diagram (if applicable)
3. Code with inline comments
4. Testing strategy
5. Deployment considerations
```

### For Cursor AI

```
You are assisting with quick edits in a TypeScript project.

CRITICAL RULES:
1. Match existing code style in the file
2. Keep changes minimal and focused
3. Use TypeScript types (no implicit any)
4. Follow .metaHub/conventions/CODING_STYLE.md
5. No architectural changes without consultation

OUTPUT FORMAT:
Code only with minimal inline comments where needed.
```

### For Windsurf

```
You are documenting and researching for a solo developer.

CRITICAL RULES:
1. Write for future-you (6 months from now)
2. Include diagrams for complex concepts
3. Provide examples and use cases
4. Link to relevant documentation
5. Consider multiple organizations using this

OUTPUT FORMAT:
Markdown with:
- Clear hierarchy (H1, H2, H3)
- Code examples
- Mermaid diagrams
- Links to references
```

### For GitHub Copilot

```
[Copilot works via inline suggestions, no explicit prompting]

SETTINGS TO CONFIGURE:
- Enable suggestions: Yes
- Suggest whole functions: Yes
- Auto-imports: Yes
- Match existing style: Yes
```

---

## 🎨 Project-Type Specific Styles

### Backend Services (APIs)

```typescript
// Structure: src/
// ├── routes/
// ├── services/
// ├── repositories/
// ├── models/
// ├── middleware/
// └── utils/

// Example API route
export const userRoutes = (app: Express) => {
  app.get('/api/users/:id', async (req, res) => {
    try {
      const user = await userService.getUserById(req.params.id);
      res.json({ data: user });
    } catch (error) {
      if (error instanceof UserNotFoundError) {
        res.status(404).json({ error: error.message });
      } else {
        res.status(500).json({ error: 'Internal server error' });
      }
    }
  });
};
```

### Frontend Applications

```typescript
// Structure: src/
// ├── components/
// ├── pages/
// ├── hooks/
// ├── contexts/
// ├── services/
// └── utils/

// Example component
interface UserProfileProps {
  userId: string;
}

export const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const { user, loading, error } = useUser(userId);

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!user) return null;

  return (
    <div className={styles.profile}>
      <h1>{user.name}</h1>
      {/* ... */}
    </div>
  );
};
```

### Libraries & Tools

```typescript
// Structure:
// ├── src/
// │   └── index.ts (main export)
// ├── tests/
// ├── docs/
// └── examples/

// Example library export
export interface Config {
  apiKey: string;
  endpoint: string;
}

export class MyLibrary {
  constructor(private config: Config) {}

  async doSomething(): Promise<Result> {
    // Implementation
  }
}

// Default export for convenience
export default MyLibrary;
```

---

## ⚠️ Anti-Patterns to ALWAYS Avoid

```typescript
// ❌ NEVER: Silent failures
try {
  await criticalOperation();
} catch (e) {
  // empty catch - disaster waiting to happen
}

// ❌ NEVER: Implicit any
function process(data) { // What is data?!
  return data.map(x => x.value); // What if data isn't an array?
}

// ❌ NEVER: God objects/functions
function handleEverything(req, res, db, cache, logger, mailer, ...) {
  // 500 lines of spaghetti
}

// ❌ NEVER: Magic numbers
if (user.age > 18) { } // Why 18? What does it mean?

// ✅ DO:
const MINIMUM_AGE = 18; // Legal adult age
if (user.age > MINIMUM_AGE) { }

// ❌ NEVER: Callback hell
getData((data) => {
  processData(data, (processed) => {
    saveData(processed, (saved) => {
      notify(saved, (result) => {
        // ...
      });
    });
  });
});

// ✅ DO: async/await
const data = await getData();
const processed = await processData(data);
const saved = await saveData(processed);
await notify(saved);
```

---

## 📊 Enforcement via CI/CD

All projects MUST have these checks in CI:

```yaml
# .github/workflows/style-enforcement.yml
name: Style Enforcement

on: [push, pull_request]

jobs:
  style-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: TypeScript Check
        run: pnpm tsc --noEmit

      - name: Linting
        run: pnpm lint

      - name: Format Check
        run: pnpm format:check

      - name: Commit Message Check
        run: pnpm commitlint --from=HEAD~1
```

---

## 🔄 Continuous Improvement

This document is living and should be updated when:
- New patterns emerge
- New anti-patterns discovered
- Team preferences evolve
- New tools/LLMs adopted

**Last Updated**: 2025-11-24
**Next Review**: 2025-12-24
