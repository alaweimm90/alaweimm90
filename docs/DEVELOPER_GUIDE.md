# Developer Guide
## 🚀 Getting Started
### Prerequisites
- Node.js 18+ & pnpm 8+
- Docker (for containerized services)
- Git 2.30+
### Installation
```bash
# Install dependencies
pnpm install
# Setup environment
cp .env.example .env
# Start development environment
pnpm dev
```
## 🏗️ Project Structure
```text
.github/               # GitHub workflows and templates
.governance/          # Repository governance
│   ├── audit/        # Audit configurations
│   ├── policies/     # Repository policies
│   └── validators/   # Custom validation scripts
packages/             # Shared packages and libraries
services/             # Microservices
web/                  # Frontend applications
tools/                # Development tools and scripts
infra/                # Infrastructure as Code
docs/                 # Documentation
tests/                # Test suites
```
## 🛠️ Development Workflow
### Branch Naming
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test related changes
- `chore/` - Build process or tooling changes
### Commit Message Format
```text
<type>(<scope>): <description>
[optional body]
[optional footer]
```
Example:
```text
feat(auth): add password reset functionality
- Add password reset endpoint
- Implement email service integration
- Add rate limiting for reset requests
Closes #123
```
## 🧪 Testing
```bash
# Run all tests
pnpm test
# Run tests with coverage
pnpm test:coverage
# Run specific test file
pnpm test -- <test-file>
# Run in watch mode
pnpm test:watch
```
## 🔧 Code Quality
### Linting
```bash
# Run linter
pnpm lint
# Fix linting issues
pnpm lint:fix
```
### Formatting
```bash
# Check formatting
pnpm format:check
# Format code
pnpm format
```
## 🚀 Deployment
### Staging
```bash
# Deploy to staging
pnpm deploy:staging
```
### Production
```bash
# Create production build
pnpm build
# Deploy to production
pnpm deploy:prod
```
## 🔒 Security
- Run security audit: `pnpm audit`
- Check for vulnerable dependencies: `pnpm audit --audit-level=high`
- Update dependencies: `pnpm update`
## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
