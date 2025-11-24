# Quick Fix Guide
## DO THIS FIRST (24-48 Hours)
### 1. Emergency Secret Rotation
```bash
# All these exposed secrets MUST be rotated IMMEDIATELY
ADMIN_PASSWORD=tkng/PUEUeeGkqeGSS63v1Vl9kLLXWI3Qi9+dMJWFlo=
API_KEY=7xd65tOkyTTUSGmKMh+08lni1t8wz4BYWl978dNz18U=
GRAFANA_PASSWORD=+MzujKN94pv2LxNUdmpvLSCxhrc7x7kEhiJTl/L6AF8=
JUPYTER_TOKEN=Kq4+43C+MMFZIVoMaXioBvX6g5RMEPtqtge1ts7ngDk=
JWT_SECRET=sU2fTjHbMDHamluEVbxN8dSWKP1CGT9qpPeldKVIxJc=
PGADMIN_PASSWORD=dJLoHSkzWBquvQ41QhrTH1RS7ojJ4jZ3WWbcsRGvTFo=
POSTGRES_PASSWORD=Bhha0r7arpGZHcUGmWK1+iS1s7I/7kqVi55JuF7Z1vM=
REDIS_PASSWORD=19cWhlDIJ4WBI+CAt882uFkU9ba7RcY94ov8Bm8945s=
SECRET_KEY=MslDJryaL1Q4V00jfYVKnkdAADX1FYbF6q1nNaNOWDA=
# Steps to take:
1. Regenerate all passwords using secure methods
2. Update all systems with new credentials
3. Audit logs for unauthorized access
4. Clear any exposed API keys from external services
```
### 2. Remove Secrets from Git History
```bash
# PERMANENTLY REMOVE .env.secure from git history
git filter-branch --tree-filter 'rm -f .env.secure' -- --all
# Force push to all branches (WARNING: rewrites history)
git push origin --force --all
# Verify removal
git log --all --full-history --oneline -- .env.secure
```
### 3. Replace Deprecated Crypto (Critical #1)
**Files to fix**:
- `/alaweimm90/automation/government/security/index.js` - Line 315
- `/alaweimm90/automation/finance/compliance/pci-dss-framework.js` - Lines 105, 147
- `/alaweimm90/automation/government/integrations/index.js`
- `/alaweimm90/automation/government/data/index.js`
- `/alaweimm90/automation/retail/compliance/customer-data-manager.js`
- `/alaweimm90/automation/finance/compliance/audit-logger.js`
**BEFORE (Vulnerable)**:
```javascript
const cipher = crypto.createCipher(algorithm, key);
let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
encrypted += cipher.final('hex');
const authTag = cipher.getAuthTag();
```
**AFTER (Secure)**:
```javascript
const algorithm = 'aes-256-gcm';
const key = crypto.scryptSync(process.env.MASTER_PASSWORD, salt, 32);
const iv = crypto.randomBytes(16);
const cipher = crypto.createCipheriv(algorithm, key, iv);
let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
encrypted += cipher.final('hex');
const authTag = cipher.getAuthTag();
// Return with IV and auth tag
return {
  encrypted,
  iv: iv.toString('hex'),
  authTag: authTag.toString('hex'),
  algorithm
};
```
### 4. Fix Command Injection (Critical #3)
**Files to fix** (8 files):
- `/.automation/modules/task-automation.js` (line 47)
- `/.automation/modules/workflow-tools.js` (line 7)
- `/.automation/modules/security-module.js` (line 7)
- `/.automation/modules/repository-scanner.js`
- `/.automation/workflows/comprehensive-automation.js`
- `/.automation/optimization/self-optimizer.js`
- `/.automation/monitoring/intelligent-monitor.js`
- `/.automation/quality/code-quality-enforcer.js`
**BEFORE (Vulnerable)**:
```javascript
const { stdout, stderr } = await execAsync(command, { cwd, timeout });
```
**AFTER (Secure)**:
```javascript
// Option 1: For simple commands
const { execFile } = require('child_process');
const { promisify } = require('util');
const execFileAsync = promisify(execFile);
const result = await execFileAsync('npm', ['test'], {
  cwd,
  timeout,
  shell: false
});
// Option 2: For complex commands with arguments
const parts = command.split(' ');
const cmd = parts[0];
const args = parts.slice(1);
const result = await execFileAsync(cmd, args, {
  cwd,
  timeout,
  shell: false,
  stdio: 'pipe'
});
```
### 5. Remove Hardcoded JWT Secret (Critical #4)
**File**: `/alaweimm90/automation/government/security/index.js` - Lines 259, 284
**BEFORE (Vulnerable)**:
```javascript
const token = jwt.sign(
  { id: user.id, username: user.username, clearance: user.clearance },
  this.config.encryptionKey || 'default-secret',  // BAD!
  { expiresIn: '8h' }
);
```
**AFTER (Secure)**:
```javascript
if (!this.config.encryptionKey) {
  throw new Error('JWT_SECRET environment variable is required');
}
const token = jwt.sign(
  { id: user.id, username: user.username, clearance: user.clearance },
  this.config.encryptionKey,  // No fallback
  { expiresIn: '8h', algorithm: 'HS256' }
);
```
### 6. Fix SQL Injection (Critical #5)
**File**: `/alaweimm90/automation/government/data/index.js` - Line 96
**BEFORE (Vulnerable)**:
```javascript
const { userClearance } = req.query;  // Direct from user!
const result = await db.query(`SELECT * FROM data WHERE user_clearance <= ${userClearance}`);
```
**AFTER (Secure)**:
```javascript
const Joi = require('joi');
// Validate input
const schema = Joi.object({
  userClearance: Joi.number().integer().min(0).max(5).required()
});
const { error, value } = schema.validate(req.query);
if (error) {
  return res.status(400).json({ error: 'Invalid input' });
}
// Use parameterized query
const result = await db.query(
  'SELECT * FROM classified_data WHERE clearance <= $1',
  [value.userClearance]
);
```
### 7. Replace MD5 with SHA-256 (Critical #6)
**Files to fix** (2 files):
- `/.metaHub/governance/registry/file-tracking-system.js`
- `/alaweimm90/scripts/files/util.js`
**BEFORE (Vulnerable)**:
```javascript
return crypto.createHash('md5').update(filePath).digest('hex');
```
**AFTER (Secure)**:
```javascript
return crypto.createHash('sha256').update(filePath).digest('hex');
```
### 8. Add Path Validation (Critical #7)
**File**: `/.automation/modules/task-automation.js` - Lines 94-95
**BEFORE (Vulnerable)**:
```javascript
for (const dir of directories) {
  const dirPath = path.join(process.cwd(), dir);
  // Can use ../../ for path traversal!
}
```
**AFTER (Secure)**:
```javascript
const ALLOWED_DIRS = ['dist', 'build', '.cache', 'tmp'];
for (const dir of directories) {
  // Whitelist validation
  if (!ALLOWED_DIRS.includes(dir)) {
    throw new Error(`Directory not allowed: ${dir}`);
  }
  const dirPath = path.join(process.cwd(), dir);
  // Verify path is within allowed directory
  const realPath = path.resolve(dirPath);
  const allowedPath = path.resolve(process.cwd());
  if (!realPath.startsWith(allowedPath)) {
    throw new Error('Path traversal attempt');
  }
  await fs.rm(dirPath, { recursive: true, force: true });
}
```
---
## WITHIN 1 WEEK
### 9. Fix CORS Configuration (Critical #8)
**File**: `/alaweimm90/automation/healthcare/config/default.js`
**BEFORE (Vulnerable)**:
```javascript
cors: {
  origins: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['http://localhost:3000'],
  credentials: true
}
```
**AFTER (Secure)**:
```javascript
const allowedOrigins = (process.env.CORS_ORIGINS || 'http://localhost:3000')
  .split(',')
  .map(o => o.trim());
// Validate no wildcard with credentials
if (allowedOrigins.includes('*') && credentials) {
  throw new Error('Cannot use wildcard origin with credentials');
}
app.use(cors({
  origin: allowedOrigins,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  maxAge: 3600
}));
```
### 10. Add Rate Limiting
**File**: `/alaweimm90/automation/government/security/index.js`
```javascript
const rateLimit = require('express-rate-limit');
// Strict limit for login
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 5,  // 5 attempts
  skipSuccessfulRequests: true,
  message: 'Too many login attempts'
});
// General API limit
const apiLimiter = rateLimit({
  windowMs: 60 * 1000,  // 1 minute
  max: 30,  // 30 requests
  message: 'Too many requests'
});
router.post('/login', loginLimiter, loginHandler);
router.use('/api/', apiLimiter);
```
### 11. Add Security Headers
```javascript
const helmet = require('helmet');
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'"]
    }
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  },
  xFrameOptions: { action: 'deny' },
  xContentTypeOptions: { noSniff: true }
}));
// Force HTTPS
app.use((req, res, next) => {
  if (process.env.NODE_ENV === 'production' && !req.secure) {
    return res.redirect('https://' + req.get('host') + req.url);
  }
  next();
});
```
### 12. Upgrade bcrypt rounds
**File**: `/alaweimm90/automation/government/security/index.js` - Line 186
**BEFORE**:
```javascript
const hashedPassword = await bcrypt.hash(password, 12);
```
**AFTER**:
```javascript
const hashedPassword = await bcrypt.hash(password, 14);  // At minimum
// Or better yet, use Argon2:
const argon2 = require('argon2');
const hashedPassword = await argon2.hash(password);
```
---
## TESTING CHECKLIST
After applying fixes, test:
```bash
# 1. Check for remaining secrets
npm install -g snyk
snyk secret test
# 2. Run dependency audit
npm audit
npm audit fix
# 3. Check for common vulnerabilities
npm install -g eslint-plugin-security
eslint . --ext .js --plugin security
# 4. Run SAST scanning
npx semgrep --config=p/security-audit
# 5. Check crypto usage
grep -r "createCipher\|createDecipher\|md5\|sha1" src/ --include="*.js"
# 6. Check command execution
grep -r "exec\|eval\|Function" src/ --include="*.js"
# 7. Check input validation
grep -r "req.query\|req.body\|req.params" src/ --include="*.js" | grep -v validate
```
---
## DEPLOYMENT CHECKLIST
Before any production deployment:
- [ ] All 8 CRITICAL vulnerabilities fixed
- [ ] All 12 HIGH vulnerabilities fixed
- [ ] All exposed secrets rotated
- [ ] `.env.secure` removed from git history
- [ ] Security headers implemented
- [ ] Rate limiting configured
- [ ] Input validation added
- [ ] Encryption at rest implemented
- [ ] Audit logging enabled
- [ ] Secrets vault configured
- [ ] Monitoring/alerting set up
- [ ] Penetration testing completed
- [ ] Security code review completed
- [ ] Compliance audit passed
- [ ] Team trained on security best practices
---
## COMPLIANCE REQUIREMENTS
### Before Production:
**PCI-DSS**:
- Strong cryptography (not MD5, not createCipher)
- Encrypted card data
- Audit logging (1 year retention)
- Access control
- Penetration testing
**HIPAA**:
- Encrypted patient data
- Audit trails
- Access controls
- Incident response plan
**GDPR**:
- No exposed personal data
- Data retention policies
- Right to deletion
- Breach notification
**SOC 2**:
- Logging and monitoring
- Encryption at rest and in transit
- Access controls
- Change management
---
Generated: 2025-11-23
