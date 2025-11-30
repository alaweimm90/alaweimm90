# 🎯 Universal Prompt Optimizer

**Transform any prompt from mediocre to exceptional in 2 minutes.**

This system analyzes and optimizes prompts across 8 critical dimensions, giving you a clear score and actionable improvements.

---

## 🚀 Quick Start (30 seconds)

1. **Copy your prompt** into the Analysis Template below
2. **Score it** using the 8-Dimension Framework
3. **Apply fixes** from the Optimization Checklist
4. **Validate** with the Before/After Examples

**Result:** 3-5x better output from Claude Code (or any LLM).

---

## 📊 The 8-Dimension Scoring Framework

Score your prompt on each dimension (0-10). **Target: 70+/80 total.**

### 1. **Clarity** (0-10)
**What it measures:** Is the request unambiguous and easy to understand?

**Scoring:**
- 0-3: Vague, confusing, multiple interpretations possible
- 4-6: Generally clear but some ambiguity
- 7-8: Clear with minor room for interpretation
- 9-10: Crystal clear, zero ambiguity

**Red flags:**
- ❌ "Make it better"
- ❌ "Fix the issues"
- ❌ "Improve performance"

**Good examples:**
- ✅ "Refactor the UserAuth class to use async/await instead of callbacks"
- ✅ "Add input validation for email field (RFC 5322 compliant)"

---

### 2. **Specificity** (0-10)
**What it measures:** Does the prompt specify WHAT, WHERE, and HOW?

**Scoring:**
- 0-3: Generic, no details
- 4-6: Some details but missing key info
- 7-8: Specific with minor gaps
- 9-10: Exact files, functions, requirements specified

**Red flags:**
- ❌ "Update the API"
- ❌ "Fix the bug in the code"
- ❌ "Add a new feature"

**Good examples:**
- ✅ "In src/api/users.ts, add rate limiting (100 req/min per IP) to the POST /api/users/login endpoint using express-rate-limit"
- ✅ "Create a new React component <PricingCard> in src/components/pricing/ that displays: title, price, features list, CTA button. Use Tailwind for styling."

---

### 3. **Context** (0-10)
**What it measures:** Does the prompt provide necessary background information?

**Scoring:**
- 0-3: No context, Claude must guess intent
- 4-6: Basic context but missing key details
- 7-8: Good context with minor gaps
- 9-10: Complete context including constraints, goals, existing patterns

**Red flags:**
- ❌ No mention of existing architecture
- ❌ No explanation of WHY the change is needed
- ❌ No constraints mentioned (performance, compatibility, etc.)

**Good examples:**
- ✅ "We're migrating from REST to GraphQL. Update the user fetching logic in src/services/UserService.ts to use Apollo Client instead of fetch(). Keep the same return types to avoid breaking existing components."
- ✅ "Our tests are failing because we upgraded Jest from v27 to v29. Fix the failing test in __tests__/UserAuth.test.ts - specifically the mock implementation of localStorage."

---

### 4. **Constraints** (0-10)
**What it measures:** Are limitations and requirements explicitly stated?

**Scoring:**
- 0-3: No constraints mentioned
- 4-6: Some constraints but incomplete
- 7-8: Most constraints specified
- 9-10: All constraints clear (performance, compatibility, style, security, etc.)

**Key constraint types:**
- **Performance:** "Must complete in <100ms"
- **Compatibility:** "Must work on Node 16+"
- **Style:** "Follow existing patterns in auth module"
- **Security:** "Must sanitize all inputs"
- **Dependencies:** "Don't add new npm packages"
- **Breaking changes:** "Must not break existing API"

**Good examples:**
- ✅ "Add email validation WITHOUT using external libraries (use regex). Must handle RFC 5322 edge cases."
- ✅ "Optimize the image processing to handle 10MB files in under 2 seconds. Must maintain backward compatibility with existing uploads."

---

### 5. **Expected Output** (0-10)
**What it measures:** Is it clear what the end result should be?

**Scoring:**
- 0-3: Unclear what success looks like
- 4-6: General idea but vague
- 7-8: Clear output with minor ambiguity
- 9-10: Exact format, structure, and acceptance criteria defined

**Red flags:**
- ❌ "Build a dashboard"
- ❌ "Create tests"
- ❌ "Document the API"

**Good examples:**
- ✅ "Create a JSON API endpoint that returns: { users: [...], total: number, page: number }. Include Swagger/OpenAPI documentation."
- ✅ "Write 5 unit tests covering: (1) successful login, (2) wrong password, (3) missing email, (4) rate limit exceeded, (5) database timeout. Use Jest + Supertest."

---

### 6. **Actionability** (0-10)
**What it measures:** Can Claude immediately start working, or must it ask questions?

**Scoring:**
- 0-3: Multiple critical unknowns, can't proceed
- 4-6: Some unknowns but can make educated guesses
- 7-8: Minor clarifications needed
- 9-10: Zero questions needed, can execute immediately

**Red flags:**
- ❌ Referencing undefined variables/files
- ❌ Missing file paths
- ❌ Unclear which approach to use
- ❌ Ambiguous requirements ("fast", "good UX", etc.)

**Good examples:**
- ✅ "In src/components/Dashboard.tsx lines 45-67, refactor the useEffect hook to use React Query's useQuery() instead. Keep the same loading/error states."
- ✅ "Add these 3 environment variables to .env.example: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_ID. Use the same format as existing vars."

---

### 7. **Scope Definition** (0-10)
**What it measures:** Is the scope appropriately bounded?

**Scoring:**
- 0-3: Scope too broad ("build an app") or too narrow ("fix line 5")
- 4-6: Scope somewhat clear but could be better bounded
- 7-8: Well-scoped with minor edge cases
- 9-10: Perfect scope - not too big, not too small, edges defined

**Red flags:**
- ❌ "Rewrite the entire codebase"
- ❌ "Build a user authentication system" (too broad without details)
- ❌ "Change this variable name" (too trivial for a prompt)

**Good examples:**
- ✅ "Refactor the 3 SQL queries in src/db/users.ts to use parameterized queries (prevent SQL injection). Do NOT refactor other files yet."
- ✅ "Add dark mode toggle to the Settings page ONLY. Don't implement the actual theming logic - just the UI control that saves preference to localStorage."

---

### 8. **Testability** (0-10)
**What it measures:** Can the result be verified objectively?

**Scoring:**
- 0-3: No way to verify success
- 4-6: Some verification possible but subjective
- 7-8: Clear success criteria with minor subjectivity
- 9-10: 100% objective verification criteria

**Red flags:**
- ❌ "Make it look better"
- ❌ "Improve code quality"
- ❌ "Optimize performance" (without metrics)

**Good examples:**
- ✅ "Reduce the bundle size of src/App.tsx by at least 20%. Measure using: npm run build && ls -lh dist/assets/App-*.js"
- ✅ "Add TypeScript types to user.service.ts such that npm run type-check passes with zero errors."

---

## 🎯 Analysis Template

Copy this template and fill it out for any prompt you want to optimize:

```markdown
### MY PROMPT
[Paste your prompt here]

### SCORING
1. Clarity:          __/10
2. Specificity:      __/10
3. Context:          __/10
4. Constraints:      __/10
5. Expected Output:  __/10
6. Actionability:    __/10
7. Scope Definition: __/10
8. Testability:      __/10

**TOTAL: __/80**

### IDENTIFIED WEAKNESSES
- [ ] Missing file paths
- [ ] Unclear requirements
- [ ] No context provided
- [ ] Constraints not specified
- [ ] Expected output vague
- [ ] Can't verify success
- [ ] Scope too broad/narrow
- [ ] Other: ___________

### OPTIMIZATION ACTIONS
1. [ ] Add specific file paths
2. [ ] Define success criteria
3. [ ] Provide background context
4. [ ] List all constraints
5. [ ] Specify exact output format
6. [ ] Bound the scope clearly
7. [ ] Make objectively testable
8. [ ] Other: ___________

### OPTIMIZED PROMPT
[Paste improved version here]

### EXPECTED IMPROVEMENT
Before: __/80 → After: __/80 (Gain: +__)
```

---

## 📝 Optimization Checklist

Use this checklist to systematically improve any prompt:

### ✅ Clarity Boosters
- [ ] Replace vague verbs ("improve" → "refactor to use async/await")
- [ ] Remove ambiguous words ("better", "faster", "cleaner")
- [ ] Use concrete nouns (specific files, functions, endpoints)
- [ ] Break multi-part requests into numbered steps

### ✅ Specificity Boosters
- [ ] Add file paths (src/components/UserProfile.tsx)
- [ ] Add line numbers when relevant (lines 45-67)
- [ ] Specify technologies/frameworks (React Query, Tailwind, Jest)
- [ ] Include exact variable/function names
- [ ] Define data formats (JSON structure, API schema)

### ✅ Context Boosters
- [ ] Explain WHY you need this change
- [ ] Describe current state vs desired state
- [ ] Mention relevant architectural patterns in use
- [ ] Reference related changes or dependencies
- [ ] Note any recent changes that matter

### ✅ Constraint Boosters
- [ ] Performance requirements ("under 100ms", "handle 10MB files")
- [ ] Compatibility requirements ("Node 16+", "iOS 15+")
- [ ] Style/pattern requirements ("follow existing auth pattern")
- [ ] Security requirements ("sanitize inputs", "use HTTPS only")
- [ ] Dependency constraints ("don't add new packages")
- [ ] Breaking change constraints ("must not break existing API")

### ✅ Output Boosters
- [ ] Specify file types (.tsx, .test.ts, .json)
- [ ] Define data structures (interfaces, schemas)
- [ ] List required test coverage
- [ ] Describe acceptance criteria
- [ ] Include documentation requirements

### ✅ Actionability Boosters
- [ ] Verify all referenced files exist
- [ ] Ensure all variables/functions are defined
- [ ] Remove conditional language ("maybe", "if possible")
- [ ] Provide example data if needed
- [ ] Include error messages to fix (if debugging)

### ✅ Scope Boosters
- [ ] Use "ONLY" to limit scope ("Settings page ONLY")
- [ ] Use "Do NOT" to exclude areas ("Do NOT refactor tests yet")
- [ ] Break large requests into phases
- [ ] Specify what to defer to later

### ✅ Testability Boosters
- [ ] Add measurable criteria ("reduce by 20%", "pass all tests")
- [ ] Specify commands to verify ("npm run test", "npm run build")
- [ ] Include expected output samples
- [ ] Define what "working" means objectively

---

## 🔥 Before/After Examples

### Example 1: Vague → Specific

❌ **BEFORE (Score: 32/80)**
```
Fix the login bug
```

**Issues:**
- Clarity: 3/10 - What bug?
- Specificity: 2/10 - Where? What's broken?
- Context: 2/10 - No background
- Constraints: 3/10 - None specified
- Expected Output: 4/10 - Unclear
- Actionability: 3/10 - Can't start without more info
- Scope: 5/10 - Too vague
- Testability: 5/10 - How to verify?

✅ **AFTER (Score: 78/80)**
```
In src/auth/LoginForm.tsx, fix the bug where clicking "Login" with valid credentials shows "Invalid password" error.

Context: This started after we upgraded bcrypt from v5.0.1 to v5.1.0 yesterday. The password comparison logic is in src/services/AuthService.ts:45-52.

Requirements:
- Must maintain backward compatibility with existing hashed passwords
- Don't change the database schema
- Add a test case to prevent regression

Expected: Login works with correct credentials, test suite passes (npm run test:auth)
```

**Improvements:**
- Clarity: 3 → 10 (specific error described)
- Specificity: 2 → 10 (files, lines, exact issue)
- Context: 2 → 9 (recent change, relevant code location)
- Constraints: 3 → 9 (compatibility, no schema changes)
- Expected Output: 4 → 9 (working login + tests)
- Actionability: 3 → 10 (can start immediately)
- Scope: 5 → 10 (bounded to login logic)
- Testability: 5 → 10 (objective: tests pass, login works)

---

### Example 2: Too Broad → Well-Scoped

❌ **BEFORE (Score: 28/80)**
```
Build a user authentication system
```

**Issues:**
- Clarity: 5/10 - Clear goal but no details
- Specificity: 1/10 - No technical details
- Context: 2/10 - No architecture info
- Constraints: 2/10 - No requirements
- Expected Output: 3/10 - Too broad
- Actionability: 1/10 - Too many unknowns
- Scope: 4/10 - Enormous scope
- Testability: 3/10 - Unclear success criteria

✅ **AFTER (Score: 76/80)**
```
Create email/password authentication for our Next.js app using SuperTokens.

Scope (Phase 1 ONLY):
1. Add SuperTokens SDK (supertokens-node, supertokens-auth-react)
2. Create /api/auth/* endpoints in pages/api/auth/
3. Add <SessionAuth> wrapper to protected pages
4. Add login/signup forms in /components/auth/

Context:
- We're using Next.js 13 (App Router)
- PostgreSQL for user data (existing schema in prisma/schema.prisma)
- Tailwind for styling (follow existing components/Button.tsx pattern)

Constraints:
- Must work with existing User model (don't change schema)
- Must support both client and server-side session checking
- Email verification can wait (Phase 2)
- OAuth providers can wait (Phase 2)

Expected Output:
- Users can sign up (email + password)
- Users can log in
- Sessions persist across page reloads
- Protected routes redirect to /login
- Logout works correctly
- All tests pass (npm run test)

Test Criteria:
1. Manual: Can create account, login, access protected page, logout
2. Automated: Add tests for each API endpoint
3. Type safety: npm run type-check passes
```

**Improvements:**
- Clarity: 5 → 9 (specific technology, clear requirements)
- Specificity: 1 → 9 (tech stack, file structure, patterns)
- Context: 2 → 9 (framework, existing code, patterns)
- Constraints: 2 → 9 (existing schema, what to defer)
- Expected Output: 3 → 10 (exact features listed)
- Actionability: 1 → 9 (can start immediately)
- Scope: 4 → 10 (Phase 1 bounded, Phase 2 deferred)
- Testability: 3 → 10 (manual + automated criteria)

---

### Example 3: Missing Context → Rich Context

❌ **BEFORE (Score: 35/80)**
```
The tests are failing, please fix them
```

**Issues:**
- Clarity: 4/10 - Clear request but vague
- Specificity: 2/10 - Which tests?
- Context: 1/10 - No background
- Constraints: 4/10 - None specified
- Expected Output: 5/10 - Tests pass (clear but incomplete)
- Actionability: 2/10 - Need to investigate first
- Scope: 5/10 - Unclear scope
- Testability: 7/10 - Clear (tests must pass)

✅ **AFTER (Score: 77/80)**
```
Fix the 3 failing tests in __tests__/api/users.test.ts after we migrated from Jest 27 to Jest 29.

Error messages:
```
FAIL __tests__/api/users.test.ts
  ● GET /api/users › returns user list
    ReferenceError: TextEncoder is not defined
  ● POST /api/users › creates new user
    TypeError: Cannot read property 'json' of undefined
  ● DELETE /api/users/:id › deletes user
    ReferenceError: TextEncoder is not defined
```

Context:
- Yesterday we upgraded Jest 27 → 29 (package.json updated)
- Tests were passing on Jest 27
- We're using node-fetch v3 for API calls in tests
- The code being tested (src/api/users.ts) hasn't changed

Known Jest 29 changes:
- jsdom environment changed (likely TextEncoder issue)
- Test environment setup may need updating

Constraints:
- Don't change the actual API code (src/api/users.ts)
- Keep using node-fetch (don't switch to axios)
- Maintain same test coverage

Expected Output:
- All 3 tests pass: npm run test __tests__/api/users.test.ts
- No warnings in test output
- Test coverage remains >80%

Likely Fix:
- Add TextEncoder polyfill to jest.setup.ts
- Update test environment config in jest.config.js
```

**Improvements:**
- Clarity: 4 → 10 (specific tests, exact errors)
- Specificity: 2 → 10 (file, error messages)
- Context: 1 → 10 (what changed, what broke, known issues)
- Constraints: 4 → 8 (what not to change)
- Expected Output: 5 → 9 (tests pass, coverage maintained)
- Actionability: 2 → 10 (can fix immediately with hints)
- Scope: 5 → 9 (3 tests only)
- Testability: 7 → 10 (objective: tests pass)

---

## 🧠 Advanced Techniques

### 1. **The Constraint Sandwich**
Structure: Context → Request → Constraints

```
[CONTEXT] We just added TypeScript to our React project.
[REQUEST] Convert src/components/UserCard.jsx to TypeScript.
[CONSTRAINTS] Keep existing props interface. Don't add new dependencies. Follow the pattern in src/components/Button.tsx.
```

**Why it works:** Context prevents confusion, constraints prevent wrong approaches.

---

### 2. **The Example Pattern**
When output format is complex, provide examples.

```
Create a function that transforms user data.

Input example:
{ firstName: "John", lastName: "Doe", email: "john@example.com" }

Expected output:
{ fullName: "John Doe", contact: { email: "john@example.com" }, initials: "JD" }
```

**Why it works:** Examples eliminate ambiguity about data structures.

---

### 3. **The Decision Tree**
For conditional logic, spell out all branches.

```
Add error handling to the login function:

IF password is wrong:
  - Return 401 status
  - Increment failed_attempts in database
  - If failed_attempts >= 5: lock account

IF account is locked:
  - Return 423 status
  - Include unlock_at timestamp in response

IF network error:
  - Return 503 status
  - Log error to Sentry
  - Don't increment failed_attempts
```

**Why it works:** Prevents Claude from guessing edge cases.

---

### 4. **The Anti-Pattern List**
Tell Claude what NOT to do.

```
Refactor this function to be async.

DO NOT:
- Change the function signature (keep same params)
- Add new dependencies
- Remove error handling
- Break backward compatibility

DO:
- Convert to async/await
- Add try/catch blocks
- Maintain same return types
- Add JSDoc comments
```

**Why it works:** Prevents common mistakes proactively.

---

### 5. **The Success Criteria Matrix**
For complex tasks, create a checklist.

```
Add Stripe payment integration.

Success criteria:
- [ ] User can add payment method (works in test mode)
- [ ] Subscription creation works (Stripe test key)
- [ ] Webhook handles payment_succeeded event
- [ ] Database updates user.subscriptionStatus correctly
- [ ] Error states handled (card declined, network failure)
- [ ] Test coverage >80% for payment logic
- [ ] TypeScript types for Stripe objects
- [ ] Environment variables documented in .env.example
```

**Why it works:** Objective checklist = objective verification.

---

## 📏 Prompt Scoring Guide

| Score | Rating | Description | Action |
|-------|--------|-------------|--------|
| 0-20 | 🔴 Critical | Unusable, too vague | Complete rewrite needed |
| 21-40 | 🟠 Poor | Major gaps, requires clarification | Significant improvements needed |
| 41-55 | 🟡 Mediocre | Some clarity but missing key info | Add context, constraints, specifics |
| 56-65 | 🟢 Acceptable | Workable but room for improvement | Polish specificity and testability |
| 66-75 | 🔵 Good | Clear and actionable | Minor tweaks for optimization |
| 76-80 | 🟣 Excellent | Professional grade, ready to execute | Ship it! |

**Target:** Aim for 70+ on all prompts sent to Claude Code.

---

## 🎯 Quick Optimization Formula

**Can't score? Use this 30-second checklist:**

1. **WHO** is affected? (Which files/functions?)
2. **WHAT** needs to change? (Specific action)
3. **WHY** is this needed? (Context/problem)
4. **HOW** should it work? (Constraints/requirements)
5. **WHEN** is it done? (Success criteria)

**Example transformation:**

❌ "Fix the API"

✅ **WHO:** src/api/users.ts endpoint GET /api/users
✅ **WHAT:** Add pagination (limit/offset params)
✅ **WHY:** Current endpoint returns all 10K users, causing timeout
✅ **HOW:** Default limit=20, max=100, use SQL LIMIT/OFFSET
✅ **WHEN:** Returns { users: [...], total: N, page: N } and loads in <200ms

---

## 🔬 Self-Optimization Exercise

**Test yourself:** Score this prompt, then optimize it.

```
Make the dashboard faster
```

<details>
<summary>Click to see analysis</summary>

**Original Score: 18/80** 🔴

- Clarity: 3/10 - "Faster" is subjective
- Specificity: 1/10 - Which dashboard? What part?
- Context: 0/10 - No background
- Constraints: 1/10 - How fast is fast enough?
- Expected Output: 2/10 - Unclear
- Actionability: 1/10 - Can't start
- Scope: 5/10 - Too broad
- Testability: 2/10 - No metrics

**Optimized Version (Score: 76/80)** 🟣

```
Optimize the load time of src/pages/Dashboard.tsx from 3.2s to under 1s.

Current bottlenecks (from Chrome DevTools):
1. Fetching 10K users on mount (2.1s)
2. Rendering UserTable with 10K rows (0.8s)
3. Large bundle size: Dashboard.js is 450KB (0.3s download)

Optimizations to implement:
1. Add pagination to user fetch (load 50 at a time)
2. Virtualize UserTable (react-window)
3. Code split Dashboard (React.lazy)
4. Add loading skeleton (perceived performance)

Constraints:
- Must maintain current UI/UX
- Don't change API response format
- Must work on Chrome/Firefox/Safari
- Keep accessibility (keyboard nav, screen readers)

Success Criteria:
- Lighthouse Performance score >90
- Time to Interactive <1s (measured with Chrome DevTools)
- Bundle size <150KB (measured: npm run build)
- All existing tests pass
```

**Improvements:**
- Added specific file (src/pages/Dashboard.tsx)
- Defined measurable target (3.2s → <1s)
- Provided context (bottleneck analysis)
- Listed exact optimizations
- Clear constraints (compatibility, accessibility)
- Objective success criteria (Lighthouse score, TTI, bundle size)

</details>

---

## 🚀 Pro Tips

### 1. **Front-load the Critical Info**
Put file paths and key requirements in the first sentence.

❌ "We need to update the authentication logic because we're getting security warnings. Can you help?"

✅ "In src/auth/login.ts, replace MD5 password hashing with bcrypt (min cost factor: 12) to fix security audit warnings."

---

### 2. **Use Absolute Metrics, Not Relative**
❌ "Make it faster"
✅ "Reduce response time from 500ms to under 100ms"

❌ "Improve code quality"
✅ "Increase test coverage from 60% to 85%"

❌ "Better error handling"
✅ "Add try/catch blocks for all async operations, log errors to Sentry"

---

### 3. **Reference Existing Patterns**
❌ "Add authentication"
✅ "Add authentication following the same pattern as src/auth/apiAuth.ts (JWT with refresh tokens)"

**Why:** Ensures consistency with existing codebase.

---

### 4. **Specify the "Done" State**
Every prompt should answer: "How will I know this is complete?"

✅ "Done when: npm run test passes, npm run build completes without warnings, manual testing shows login works"

---

### 5. **Include Failure Scenarios**
Don't just describe the happy path.

```
Add file upload feature.

Success: User uploads PDF, stored in S3, DB updated
Failures to handle:
- File too large (>10MB): Show error "Max 10MB"
- Wrong file type: Show error "PDF only"
- S3 failure: Show error "Upload failed, try again"
- Network timeout: Retry 3 times, then show error
```

---

## 📊 Real-World Prompt Analysis

### Prompt: "Help me debug this"

**Score: 12/80** 🔴 **Critical**

**Issues:**
- No code provided
- No error message
- No context
- No file location
- Can't even begin

**Optimization:**
```
Debug the "TypeError: Cannot read property 'length' of undefined" error in src/utils/validators.ts:23.

Code (lines 20-25):
```
function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return email.length > 0 && emailRegex.test(email)  // Line 23 fails
}
```

Error occurs when: User submits login form without entering email

Expected fix: Add null/undefined check before accessing .length

Context: This started appearing after we made email field optional in LoginForm.tsx
```

**New Score: 74/80** 🔵 **Good**

---

## 💡 Common Mistakes & Fixes

| Mistake | Fix |
|---------|-----|
| "Fix the bug" | "Fix the TypeError at line 45 in auth.ts where user.email is undefined" |
| "Make it responsive" | "Add mobile breakpoints (<768px) to Dashboard.tsx using Tailwind's md: prefix. Stack sidebar below header." |
| "Add tests" | "Add Jest tests for 5 scenarios in userService.test.ts: create, read, update, delete, error handling" |
| "Improve performance" | "Reduce API response time from 800ms to <200ms by adding Redis caching (5min TTL)" |
| "Better UI" | "Update Button component to match Tailwind UI design: rounded-lg, shadow-sm, transition-colors" |
| "Refactor this" | "Refactor UserProfile.tsx to extract 3 sub-components: Avatar, Bio, Stats. Keep same props interface." |

---

## 🎓 Graduation Test

**You've mastered prompt optimization when you can:**

1. ✅ Score any prompt in under 60 seconds
2. ✅ Identify the weakest dimension instantly
3. ✅ Optimize prompts to 70+ consistently
4. ✅ Write prompts that require zero follow-up questions
5. ✅ Use the 5W formula (Who/What/Why/How/When) instinctively

**Practice on your last 5 prompts to Claude Code. Average score should be 65+.**

---

## 📚 Additional Resources

### Templates for Common Tasks

**Debug Request Template:**
```
Debug [ERROR_MESSAGE] in [FILE:LINE]

Code context:
[Paste relevant code]

Error occurs when: [TRIGGER]

Expected behavior: [WHAT_SHOULD_HAPPEN]

Context: [WHY_THIS_IS_BREAKING]
```

**Feature Request Template:**
```
Add [FEATURE_NAME] to [LOCATION]

Requirements:
1. [REQUIREMENT_1]
2. [REQUIREMENT_2]
...

Context: [WHY_NEEDED]

Constraints:
- [CONSTRAINT_1]
- [CONSTRAINT_2]

Success criteria:
- [ ] [CRITERION_1]
- [ ] [CRITERION_2]
```

**Refactor Request Template:**
```
Refactor [FILE/FUNCTION] to [GOAL]

Current approach: [HOW_IT_WORKS_NOW]
Target approach: [HOW_IT_SHOULD_WORK]

Constraints:
- Don't change: [WHAT_TO_KEEP]
- Must maintain: [REQUIREMENTS]

Pattern to follow: [REFERENCE_FILE]

Success: [VERIFICATION_METHOD]
```

---

## 🏁 Summary

**The Universal Prompt Optimizer gives you:**

✅ **8-dimension scoring framework** - Objective quality assessment
✅ **Analysis template** - Systematic improvement process
✅ **Optimization checklist** - Actionable improvements
✅ **Before/after examples** - Learn from real transformations
✅ **Advanced techniques** - Pro-level prompt engineering
✅ **Common mistake fixes** - Avoid pitfalls
✅ **Ready-to-use templates** - Start with strong foundations

**Result:** 3-5x better responses from Claude Code with minimal effort.

---

## 🎯 Your Next Action

1. **Test it now:** Take your last prompt to Claude Code
2. **Score it:** Use the 8-dimension framework
3. **Optimize it:** Apply the checklist
4. **Resend it:** See the difference

**Target: 70+/80 on every prompt you send.**

---

**Built for builders who demand excellence from their tools.** 🚀

*Now go optimize every prompt you send!*
