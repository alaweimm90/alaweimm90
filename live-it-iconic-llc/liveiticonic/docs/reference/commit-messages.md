# Commit Message Standards

We use Conventional Commits:

Format: `type(scope): subject`

Common types:
- feat: a new feature
- fix: a bug fix
- docs: documentation only changes
- style: formatting, missing semi colons, etc; no code change
- refactor: code change that neither fixes a bug nor adds a feature
- perf: performance improvements
- test: adding missing tests or refactoring tests
- build: changes that affect the build system or external dependencies
- ci: changes to our CI configuration files and scripts
- chore: other changes that don't modify src or test files
- revert: reverts a previous commit

Examples:
- `feat(catalog): add size filter to product list`
- `fix(pwa): correct service worker registration`
- `docs: update contributing guidelines`

Subject should be concise and in imperative mood. Avoid capitalized subject and trailing period.

