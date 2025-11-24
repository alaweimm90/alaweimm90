# Security Policy

## Supported Versions

Security support for metaHub:

| Version  | Supported         |
| -------- | ----------------- |
| Latest   | ✅ Yes            |
| Previous | ✅ Yes (3 months) |
| Older    | ❌ No             |

**Note**: metaHub is a coordination hub, not a production system. Security issues should be reported, but the priority may differ from production code.

## Reporting a Vulnerability

Found a security issue? Please report it responsibly.

### How to Report

1. **GitHub Private Vulnerability Reporting**: Use GitHub's private vulnerability reporting feature
2. **Direct Email**: security@example.com
3. **GitHub**: Contact repository maintainer directly via message

**Do NOT open public issues for security vulnerabilities.**

### What to Include

- Description of the vulnerability
- Location (file, line number if applicable)
- Steps to reproduce
- Potential impact
- Suggested remediation (if applicable)

### Response Timeline

- **Critical** (e.g., leaked secrets): Within 4 hours
- **High** (e.g., injection vulnerabilities): Within 24 hours
- **Medium** (e.g., documentation issues): Within 48 hours
- **Low** (e.g., typos): Best effort

## Security Considerations

### metaHub as a Coordination Hub

metaHub is a documentation and tooling repository, not a production system. However, security is important:

- **Documentation**: Should not contain secrets, API keys, or credentials
- **Scripts**: Should not expose sensitive data or enable privilege escalation
- **Configuration**: Should be safe for all contributors

### What NOT to Commit

Never commit to metaHub:

- API keys or credentials
- Private keys or certificates
- Passwords or tokens
- Sensitive configuration values
- Confidential information

Use `.gitignore` and `.env` patterns:

```bash
# Environment files
.env
.env.local
.env.*.local

# Secrets
*.key
*.pem
secrets/
credentials/
```

### Automated Secret Detection

We use git hooks to prevent accidental secret commits:

```bash
# Setup pre-commit hooks
./metaHub/scripts/setup/setup-pre-commit.sh
```

If you accidentally commit a secret:

1. Contact security@example.com immediately
2. The commit will be removed from history
3. Rotate any exposed credentials

## Security Best Practices

### For Contributors

1. **Never hardcode secrets** - Use environment variables
2. **Keep dependencies updated** - Run security audits regularly
3. **Review scripts carefully** - Understand before running
4. **Report issues privately** - Use the reporting mechanism above
5. **Follow standards** - See REPO_STANDARDS.md

### For Script Writers

When writing scripts, ensure:

- Input validation (prevent injection attacks)
- Error handling (don't expose sensitive data)
- Safe defaults (principle of least privilege)
- Documentation (explain security implications)
- Testing (verify security behavior)

Example:

```bash
#!/bin/bash
set -euo pipefail

# Validate input
if [[ ! "$1" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: Invalid input" >&2
    exit 1
fi

# Use absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Don't expose sensitive data in error messages
if ! command -v jq &> /dev/null; then
    echo "Error: Required tool not found" >&2
    exit 1
fi
```

### For Documentation

When writing documentation:

- Don't include example credentials or API keys
- Warn about security implications
- Link to security guidelines
- Update when security practices change

Example:

```markdown
### API Configuration

⚠️ **Security Warning**: Never commit API keys to the repository.

Use environment variables:

\`\`\`bash
export API_KEY="your-key-here" # Never commit this
\`\`\`

See [SECURITY.md](../../metaHub/SECURITY.md) for details.
```

## Compliance and Standards

metaHub follows these security standards:

- **REPO_STANDARDS.md** - All projects must follow standards
- **OWASP Top 10** - Principles for avoiding common vulnerabilities
- **NIST Cybersecurity Framework** - Security best practices
- **Zero Trust** - Assume nothing is secure by default

## Vulnerability Types

### Critical

- Arbitrary code execution
- Authentication bypass
- Credential exposure
- Data exfiltration

### High

- Injection vulnerabilities (SQL, command, etc.)
- Privilege escalation
- Path traversal attacks
- XXE vulnerabilities

### Medium

- Information disclosure
- Insecure defaults
- Missing security headers
- Weak cryptography

### Low

- Typos in security documentation
- Outdated security links
- Verbose error messages
- Missing security warnings

## Security Operations

### Regular Audits

- **Weekly**: Check for exposed secrets using git history
- **Monthly**: Review dependencies for vulnerabilities
- **Quarterly**: Security audit of scripts and documentation
- **Annually**: Full security assessment

### Incident Response

If a security incident occurs:

1. **Identify**: Determine scope and severity
2. **Contain**: Revoke compromised credentials
3. **Investigate**: Understand how it happened
4. **Remediate**: Fix the underlying issue
5. **Communicate**: Notify affected parties
6. **Document**: Update policies to prevent recurrence

### Dependency Management

Keep dependencies up to date:

```bash
# Check for vulnerable dependencies
npm audit              # Node.js
pip install safety    # Python
```

## Questions?

- **Email**: security@example.com
- **Issues**: GitHub issue tracker (for non-security issues)
- **Private Report**: GitHub's private vulnerability reporting

---

Thank you for helping keep metaHub secure!
