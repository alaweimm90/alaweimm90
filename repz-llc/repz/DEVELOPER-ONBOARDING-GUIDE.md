# 🚀 REPZ Developer Onboarding Guide

## 👋 **Welcome to the REPZ Platform Development Team**

This guide will get you up to speed with our comprehensive system audit and validation tools, ensuring you can maintain the platform's enterprise-grade reliability and performance standards.

---

## 🎯 **Quick Start Checklist**

### **1. Environment Setup**
```bash
# Clone the repository
git clone [repository-url]
cd repzcoach-pro-ai

# Install dependencies
npm install

# Run initial system validation
npm run validate:production
```

### **2. Verify Your Setup**
```bash
# Ensure TypeScript is working
npm run type-check
# ✅ Expected: No TypeScript errors

# Run quick system audit
npm run audit:quick
# ✅ Expected: 0 high-severity issues

# Start development server
npm run dev
# ✅ Expected: Server starts on http://localhost:8080
```

### **3. Access Development Tools**
- **System Audit Dashboard:** http://localhost:8080/audit-dashboard (admin login required)
- **Production Monitoring:** http://localhost:8080/monitoring-dashboard (admin login required)
- **Main Application:** http://localhost:8080

---

## 🔧 **Essential Development Commands**

### **Daily Development Workflow**
```bash
# Before starting work
npm run audit:quick                 # Quick system health check
npm run type-check                  # Ensure no TypeScript errors

# During development
npm run dev                         # Start development server
npm run test                        # Run tests in watch mode

# Before committing
npm run audit:full                  # Comprehensive validation
npm run lint                        # Code quality check
npm run build:production            # Verify production build
```

### **Audit & Validation Commands**
```bash
# System Health
npm run audit:quick                 # Fast critical issue detection
npm run audit:routes                # Dead page and route validation  
npm run audit:full                  # Complete system audit

# Specialized Testing
npm run test:auth                   # Authentication system tests
npm run test:supabase               # Database integration tests
npm run test:stripe                 # Payment processing tests
npm run test:e2e                    # End-to-end user journey tests

# Production Readiness
npm run validate:production         # Complete pre-deployment validation
npm run build:production            # Production build with optimization
npm run analyze:bundle              # Bundle size and optimization analysis
```

---

## 🏗️ **System Architecture Overview**

### **Core Components**
```
src/
├── components/
│   ├── testing/                    # 🔍 System audit and validation tools
│   │   ├── SystemAuditDashboard.tsx      # Comprehensive health monitoring
│   │   ├── DashboardTestSuite.tsx        # Role/tier access validation
│   │   └── SupabaseIntegrityChecker.tsx  # Database health monitoring
│   ├── admin/                      # 👑 Admin-only components
│   │   └── ProductionMonitoringDashboard.tsx
│   └── ui/                         # 🎨 Design system components
├── constants/
│   └── routes.ts                   # 🗺️ Centralized route management
├── hooks/                          # ⚡ Custom React hooks
├── integrations/supabase/          # 🗄️ Database client and types
└── pages/                          # 📄 Route-level page components

scripts/
├── quick-audit.cjs                 # 🚀 Fast system validation
└── audit-dead-pages.cjs            # 🔍 Route and component analysis
```

### **Tier-Based Architecture**
```typescript
// Four subscription tiers with feature gating
'core'        // $89/month  - Essential training and nutrition
'adaptive'    // $149/month - Interactive coaching with tracking
'performance' // $229/month - Advanced biohacking and AI
'longevity'   // $349/month - Premium concierge service

// Role-based access control
'client' | 'coach' | 'admin'
```

---

## 🛡️ **System Audit Tools Guide**

### **1. System Audit Dashboard** (`/audit-dashboard`)

**Access:** Admin role required
**Purpose:** Real-time system health monitoring with actionable insights

**Key Features:**
- **Overview Tab:** System health scores across 8 categories
- **Routing Tab:** Dead page detection and route validation
- **Dashboards Tab:** Role/tier access control testing
- **Database Tab:** Supabase data integrity monitoring
- **Components Tab:** Unused component detection

**When to Use:**
- Daily development health checks
- Before major feature releases
- Investigating system issues
- Production readiness validation

### **2. Dashboard Test Suite**

**Purpose:** Validates all role/tier dashboard combinations

**Test Scenarios:**
```typescript
// Client tier combinations
{ role: 'client', tier: 'core' }        // Basic features only
{ role: 'client', tier: 'adaptive' }    // + Biomarkers, weekly check-ins
{ role: 'client', tier: 'performance' } // + AI coaching, advanced analytics
{ role: 'client', tier: 'longevity' }   // + Concierge, in-person training

// Staff roles
{ role: 'coach' }                       // Client management, analytics
{ role: 'admin' }                       // System health, user management
```

**When to Use:**
- After modifying dashboard components
- Adding new tier-gated features
- Before production deployments
- Investigating access control issues

### **3. Supabase Integrity Checker**

**Purpose:** Detects zombie accounts and data inconsistencies

**Validation Categories:**
- **Zombie Accounts:** Users without proper profiles
- **Role Validation:** Invalid client/coach/admin roles
- **Tier Consistency:** Subscription/profile tier mismatches
- **Relationship Integrity:** Orphaned coach-client assignments

**When to Use:**
- Weekly data health checks
- After user data migrations
- Investigating authentication issues
- Before major database changes

### **4. Quick Audit Script**

**Command:** `npm run audit:quick`
**Purpose:** Fast critical issue detection (< 30 seconds)

**Checks:**
- Hardcoded navigation paths
- Commented out routes
- Missing dashboard security
- Import path errors

**When to Use:**
- Before every commit
- After route modifications
- Pre-commit hook validation
- Quick development health checks

### **5. Dead Page Detection**

**Command:** `npm run audit:routes`
**Purpose:** Comprehensive route and component analysis

**Detection:**
- Routes without corresponding pages
- Orphaned page components
- Unused components (sample analysis)
- Suspicious code patterns

**When to Use:**
- Monthly codebase cleanup
- Before major refactoring
- Bundle size optimization
- Code review processes

---

## 🎨 **Development Best Practices**

### **1. Route Management**
```typescript
// ✅ DO: Use centralized route constants
import { ROUTES, navigateToPricing } from '@/constants/routes';

const handleUpgrade = () => {
  navigateToPricing(); // Safe, validated navigation
};

// ❌ DON'T: Use hardcoded navigation
const handleUpgrade = () => {
  window.location.href = '/pricing'; // Audit will flag this
};
```

### **2. Component Creation**
```typescript
// ✅ DO: Include error boundaries and tier access
const MyDashboard: React.FC = () => {
  const { hasMinimumTier } = useTierAccess();
  
  if (!hasMinimumTier('adaptive')) {
    return <TierUpgradePrompt requiredTier="adaptive" />;
  }
  
  return (
    <ErrorBoundary componentName="MyDashboard">
      {/* Component content */}
    </ErrorBoundary>
  );
};

// ❌ DON'T: Create dashboards without access control
const MyDashboard: React.FC = () => {
  return <div>{/* Unprotected content */}</div>;
};
```

### **3. Database Operations**
```typescript
// ✅ DO: Validate data consistency
const updateUserTier = async (userId: string, newTier: TierType) => {
  // Update profile
  await supabase
    .from('client_profiles')
    .update({ tier: newTier })
    .eq('user_id', userId);
    
  // Validate consistency
  await validateUserTierConsistency(userId);
};

// ❌ DON'T: Update data without validation
const updateUserTier = async (userId: string, newTier: string) => {
  await supabase
    .from('client_profiles')
    .update({ tier: newTier }) // No validation, potential inconsistency
    .eq('user_id', userId);
};
```

---

## 🚨 **Common Issues & Solutions**

### **Issue: "Route not found" errors in audit**
```bash
# Check for typo in route definition
npm run audit:routes

# Verify component exists
ls src/pages/YourComponent.tsx

# Ensure proper import in App.tsx
```

### **Issue: Dashboard access control failing**
```bash
# Run dashboard test suite
# Navigate to /audit-dashboard → Dashboards tab
# Check specific role/tier combination

# Verify tier access hook usage
grep -r "useTierAccess" src/components/
```

### **Issue: Zombie accounts detected**
```bash
# Run database integrity check
# Navigate to /audit-dashboard → Database tab
# Review affected user IDs

# Fix via admin dashboard or direct database update
```

### **Issue: Build size too large**
```bash
# Analyze bundle composition
npm run analyze:bundle

# Run unused component detection
npm run audit:routes

# Consider lazy loading heavy components
```

---

## 📊 **Monitoring & Alerts**

### **Daily Monitoring**
- [ ] Check `/audit-dashboard` for system health
- [ ] Run `npm run audit:quick` before major changes
- [ ] Monitor bundle size with `npm run analyze:bundle`

### **Weekly Reviews**
- [ ] Full system audit via `/audit-dashboard`
- [ ] Database integrity check for zombie accounts
- [ ] Performance regression analysis
- [ ] Security vulnerability scan with `npm audit`

### **Pre-Deployment**
- [ ] Complete `PRODUCTION-DEPLOYMENT-CHECKLIST.md`
- [ ] Run `npm run validate:production`
- [ ] Manual testing of critical user journeys
- [ ] Verify all CI/CD checks pass

---

## 🔗 **Key Resources**

### **Documentation**
- [CLAUDE.md](CLAUDE.md) - Project overview and architecture
- [PRODUCTION-DEPLOYMENT-CHECKLIST.md](PRODUCTION-DEPLOYMENT-CHECKLIST.md) - Pre-deployment validation
- [TECHNICAL-AUDIT-RESOLUTION-REPORT.md](TECHNICAL-AUDIT-RESOLUTION-REPORT.md) - System audit resolution summary

### **Development Tools**
- **System Audit:** `/audit-dashboard` (admin access)
- **Performance Monitoring:** `/monitoring-dashboard` (admin access)
- **Component Testing:** `src/components/testing/`
- **Route Management:** `src/constants/routes.ts`

### **Scripts & Automation**
- `scripts/quick-audit.cjs` - Fast validation
- `scripts/audit-dead-pages.cjs` - Comprehensive analysis
- `.github/workflows/audit-validation.yml` - CI/CD automation

---

## 🎓 **Learning Path**

### **Week 1: System Understanding**
1. Set up development environment
2. Run through all audit tools
3. Understand tier-based architecture
4. Practice with dashboard test suite

### **Week 2: Feature Development** 
1. Create a simple component with proper error boundaries
2. Add tier-gated feature with access control
3. Practice centralized route management
4. Run full audit validation

### **Week 3: Advanced Topics**
1. Database integrity monitoring
2. Performance optimization techniques
3. CI/CD pipeline understanding
4. Production deployment process

### **Ongoing: Best Practices**
- Always run `npm run audit:quick` before commits
- Use audit dashboard for troubleshooting
- Follow tier-based access control patterns
- Maintain centralized route management

---

## 🆘 **Getting Help**

### **System Issues**
1. Check `/audit-dashboard` for specific error details
2. Run relevant audit script (`npm run audit:quick` or `npm run audit:routes`)
3. Review GitHub Actions for CI/CD failures
4. Consult technical documentation in repository

### **Development Questions**
1. Review existing components for patterns
2. Check `CLAUDE.md` for architecture guidance
3. Use audit tools to validate your approach
4. Test with dashboard test suite before committing

### **Emergency Issues**
1. Check production monitoring dashboard
2. Review deployment checklist for rollback procedures
3. Use audit tools to identify root cause
4. Follow incident response procedures

---

## ✅ **Onboarding Completion Checklist**

- [ ] ✅ Development environment set up and validated
- [ ] ✅ Successfully accessed all audit dashboards
- [ ] ✅ Ran complete audit suite without critical issues
- [ ] ✅ Created test component with proper error boundaries
- [ ] ✅ Understands tier-based access control system
- [ ] ✅ Practiced centralized route management
- [ ] ✅ Completed first full audit validation cycle
- [ ] ✅ Familiar with CI/CD audit pipeline
- [ ] ✅ Knows where to find help and documentation

**🎉 Welcome to the team! You're now ready to maintain and enhance the REPZ platform with confidence.**

---

**📞 Team Contacts:**
- **Tech Lead:** [Name] - [Contact]
- **DevOps:** [Name] - [Contact]  
- **Database Admin:** [Name] - [Contact]

**🔗 Quick Links:**
- [Live Audit Dashboard](/audit-dashboard)
- [Production Monitoring](/monitoring-dashboard)
- [GitHub Repository](.)
- [CI/CD Pipeline](.github/workflows/audit-validation.yml)