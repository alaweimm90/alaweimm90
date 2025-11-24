# Contributing
## How to Contribute
We welcome contributions to this project! Here's how you can help:
### Getting Started
1. **Fork the repository** - Create your own fork of the project
2. **Clone your fork** - `git clone https://github.com/your-username/repo-name.git`
3. **Create a branch** - `git checkout -b feature/your-feature-name`
4. **Make your changes** - Write code, fix bugs, or improve documentation
5. **Test your changes** - Ensure all tests pass
6. **Commit your changes** - Use conventional commit messages (see below)
7. **Push to your fork** - `git push origin feature/your-feature-name`
8. **Create a Pull Request** - Submit a PR with a clear description
### Commit Message Convention
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
```
<type>(<scope>): <subject>
```
Types:
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code refactoring without changing functionality
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system or dependency changes
- **ci**: CI/CD configuration changes
- **chore**: Other changes that don't modify src or test files
Example: `feat(auth): add OAuth2 authentication`
### Code Style
- Follow the existing code style
- Run `npm run lint` before committing
- Run `npm run format` to auto-format code
- Ensure TypeScript types are properly defined
### Testing
- Write tests for new features
- Ensure all tests pass: `npm test`
- Maintain test coverage above 80%
- Run `npm run test:coverage` to check coverage
### Documentation
- Update README.md if needed
- Add JSDoc/TSDoc comments for public APIs
- Update relevant documentation in `/docs`
- Include examples for new features
### Pull Request Guidelines
1. **PR Title**: Use conventional commit format
2. **Description**: Clearly explain what and why
3. **Screenshots**: Include for UI changes
4. **Breaking Changes**: Clearly mark if applicable
5. **Issue Reference**: Link related issues
### Code Review Process
1. Automated checks must pass
2. At least one maintainer approval required
3. Address all review comments
4. Keep PRs focused and reasonably sized
### Security
- Never commit secrets or credentials
- Report security vulnerabilities privately
- Follow security best practices
- Run security scans before submitting
### Questions?
- Open an issue for bugs or feature requests
- Join discussions in existing issues
- Contact maintainers for guidance
## Development Setup
```bash
# Install dependencies
npm install
# Run development server
npm run dev
# Run tests
npm test
# Run linting
npm run lint
# Run type checking
npm run type-check
```
## License
By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).
Thank you for contributing! 🎉
