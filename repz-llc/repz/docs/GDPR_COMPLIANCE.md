# GDPR Compliance Guide for REPZ Platform

## Overview

This document outlines GDPR compliance measures implemented in the REPZ fitness coaching platform, covering data protection, user rights, and privacy best practices.

## Table of Contents

1. [Data Protection Principles](#data-protection-principles)
2. [User Rights](#user-rights)
3. [Data Collection](#data-collection)
4. [Data Storage](#data-storage)
5. [Third-Party Integrations](#third-party-integrations)
6. [User Consent](#user-consent)
7. [Data Breach Protocol](#data-breach-protocol)
8. [Implementation Checklist](#implementation-checklist)

---

## Data Protection Principles

### Lawfulness, Fairness, and Transparency

- Users are informed about data collection through clear privacy policy
- Consent is obtained before processing personal data
- Users can access all collected data through dashboard

### Purpose Limitation

- Data collected only for specified, explicit purposes
- Fitness data used solely for coaching and analytics
- Marketing communications require separate opt-in

### Data Minimization

- Only necessary data is collected
- Optional fields clearly marked
- Users can skip non-essential questions

### Accuracy

- Users can update profile information anytime
- Data sync frequency configurable per integration
- Incorrect data can be corrected or deleted

### Storage Limitation

- Data retention policies defined per data type
- Inactive accounts deleted after 2 years
- Users can request immediate deletion

### Integrity and Confidentiality

- All data encrypted at rest and in transit
- Row Level Security (RLS) enforced
- Access logs maintained for audit

### Accountability

- Data Protection Officer designated
- Regular compliance audits conducted
- Privacy impact assessments for new features

---

## User Rights

### Right to Access

**Implementation:**
```typescript
// Export user data
export async function exportUserData(userId: string) {
  const data = await Promise.all([
    getUserProfile(userId),
    getUserWorkouts(userId),
    getUserNutrition(userId),
    getUserIntegrations(userId),
    getUserReferrals(userId),
  ]);

  return {
    profile: data[0],
    workouts: data[1],
    nutrition: data[2],
    integrations: data[3],
    referrals: data[4],
    exportedAt: new Date().toISOString(),
  };
}
```

**User Interface:**
- Settings → Privacy → Download My Data
- Data exported in JSON format
- Includes all personal and usage data

### Right to Rectification

**Implementation:**
- All profile fields editable by user
- Update triggers stored in database
- Audit trail maintained for changes

**User Interface:**
- Settings → Profile → Edit Information
- Real-time validation
- Change confirmation

### Right to Erasure ("Right to be Forgotten")

**Implementation:**
```typescript
// Delete user account and all data
export async function deleteUserAccount(userId: string) {
  // 1. Disconnect all integrations
  await disconnectAllIntegrations(userId);

  // 2. Delete user data (cascading)
  await supabase.from('client_profiles').delete().eq('user_id', userId);
  await supabase.from('user_integrations').delete().eq('user_id', userId);
  await supabase.from('referral_codes').delete().eq('user_id', userId);

  // 3. Delete auth account
  await supabase.auth.admin.deleteUser(userId);

  // 4. Log deletion for compliance
  await logDataDeletion(userId);
}
```

**User Interface:**
- Settings → Privacy → Delete Account
- Two-step confirmation required
- 30-day grace period option

### Right to Restrict Processing

**Implementation:**
```typescript
// Suspend data processing
export async function restrictProcessing(userId: string) {
  await supabase
    .from('client_profiles')
    .update({
      processing_restricted: true,
      restricted_at: new Date().toISOString(),
    })
    .eq('user_id', userId);
}
```

**User Interface:**
- Settings → Privacy → Restrict Data Processing
- Disables: analytics, AI recommendations, third-party sync
- Maintains: essential account functions

### Right to Data Portability

**Implementation:**
```typescript
// Export data in machine-readable format
export async function exportPortableData(userId: string) {
  const data = await exportUserData(userId);

  return {
    format: 'JSON',
    schema_version: '1.0',
    data: data,
    compatible_with: ['REPZ', 'Standard Fitness Format'],
  };
}
```

**User Interface:**
- Settings → Privacy → Export Data
- Formats: JSON, CSV, Standard Fitness Format
- Direct download or email delivery

### Right to Object

**Implementation:**
```typescript
// Opt out of specific processing
export async function objectToProcessing(
  userId: string,
  processingType: string
) {
  const objections = await getUserObjections(userId);
  objections.push({
    type: processingType,
    objected_at: new Date().toISOString(),
  });

  await supabase
    .from('client_profiles')
    .update({ processing_objections: objections })
    .eq('user_id', userId);
}
```

**User Interface:**
- Settings → Privacy → Data Processing Preferences
- Options: Marketing, AI Analysis, Behavioral Analytics, Third-Party Sharing

---

## Data Collection

### Personal Data

**What We Collect:**
- Name, email, phone (required for account)
- Age, gender, location (optional)
- Health metrics (user-provided)
- Payment information (via Stripe, not stored)

**Legal Basis:**
- Contract performance (account management)
- Consent (health data, marketing)
- Legitimate interest (fraud prevention)

### Sensitive Data

**Health Data:**
- Workout logs and performance metrics
- Nutrition data and meal plans
- Biometric data from integrations
- Medical clearances and conditions

**Special Protections:**
- Explicit consent required
- Extra encryption layer
- Limited access (coaches only)
- Regular security audits

### Automated Data

**Collected Automatically:**
- IP address and device information
- Usage analytics and session data
- Error logs and performance metrics
- Geolocation (with permission)

**Purpose:**
- Service improvement
- Security and fraud detection
- Personalized recommendations
- System optimization

---

## Data Storage

### Database Security

**Supabase Implementation:**
- PostgreSQL with full encryption
- Row Level Security (RLS) policies
- Automatic backups
- Multi-region redundancy

**Access Control:**
```sql
-- Example RLS policy
CREATE POLICY "Users can only access own data"
ON client_profiles FOR SELECT
USING (auth.uid() = user_id);
```

### Third-Party Services

**Stripe (Payments):**
- PCI DSS Level 1 compliant
- Tokenized payment methods
- No raw card data stored

**Whoop, Strava, Apple Health:**
- OAuth tokens encrypted
- Data synced on-demand
- Can be disconnected anytime

### Data Retention

**Active Users:**
- Profile data: Indefinite (while account active)
- Workout logs: 5 years
- Analytics data: 2 years
- Session logs: 90 days

**Inactive Accounts:**
- Warning after 18 months
- Deletion after 24 months
- Option to export before deletion

---

## Third-Party Integrations

### OAuth Consent Flow

1. **Clear Disclosure**
   - What data will be accessed
   - How data will be used
   - Can be revoked anytime

2. **Minimal Scopes**
   - Only request necessary permissions
   - No access to unrelated data

3. **Revocation**
   - One-click disconnect
   - Tokens immediately revoked
   - Data sync stops instantly

### Data Processor Agreements

All third-party integrations have signed Data Processor Agreements ensuring:
- GDPR compliance
- Data security measures
- Sub-processor notification
- Audit rights
- Data breach notification

---

## User Consent

### Consent Management

**Implementation:**
```typescript
interface UserConsent {
  terms_accepted: boolean;
  privacy_policy_accepted: boolean;
  marketing_emails: boolean;
  analytics_tracking: boolean;
  ai_recommendations: boolean;
  third_party_integrations: boolean;
  accepted_at: string;
  ip_address: string;
}

// Track consent
export async function recordConsent(
  userId: string,
  consent: UserConsent
) {
  await supabase
    .from('user_consent')
    .insert({
      user_id: userId,
      ...consent,
    });
}
```

### Consent Requirements

**Required (cannot use service without):**
- Terms of Service
- Privacy Policy
- Essential cookies

**Optional (can withdraw anytime):**
- Marketing communications
- Analytics and tracking
- AI-powered recommendations
- Third-party data sharing

### Withdrawal Process

**User Interface:**
- Settings → Privacy → Manage Consent
- Toggle switches for each category
- Immediate effect upon change
- Confirmation email sent

---

## Data Breach Protocol

### Detection

- Automated security monitoring
- Anomaly detection systems
- Regular security audits
- User reporting mechanism

### Response Plan

**Within 72 Hours:**
1. Contain the breach
2. Assess severity and scope
3. Notify supervisory authority (if required)
4. Document the incident

**User Notification:**
- Email within 72 hours (if high risk)
- Clear explanation of breach
- Steps users should take
- Support contact information

### Prevention

- Regular penetration testing
- Security awareness training
- Incident response drills
- Third-party security audits

---

## Implementation Checklist

### Technical Measures

- [x] Database encryption at rest
- [x] SSL/TLS for data in transit
- [x] Row Level Security (RLS)
- [x] OAuth 2.0 for integrations
- [x] Password hashing (bcrypt)
- [x] API rate limiting
- [x] Input validation and sanitization
- [x] CSRF protection
- [x] XSS prevention

### Organizational Measures

- [ ] Data Protection Officer appointed
- [ ] Privacy policy published
- [ ] Cookie policy implemented
- [ ] Data processing agreements signed
- [ ] Employee training completed
- [ ] Incident response plan tested
- [ ] Regular compliance audits
- [ ] Privacy impact assessments

### User Rights

- [x] Data access functionality
- [x] Data export capability
- [x] Account deletion process
- [x] Consent management system
- [x] Privacy settings dashboard
- [ ] Data rectification interface
- [ ] Processing restriction option
- [ ] Objection mechanism

### Documentation

- [x] Privacy policy
- [x] Terms of service
- [x] Cookie policy
- [x] Data retention policy
- [x] Security measures documentation
- [x] User rights information
- [x] Integration documentation
- [x] GDPR compliance guide

---

## Compliance Monitoring

### Regular Audits

**Monthly:**
- Access log review
- Integration status check
- Consent record validation

**Quarterly:**
- Security assessment
- Privacy policy review
- Data retention cleanup

**Annually:**
- Full compliance audit
- Third-party security review
- Privacy impact assessment

### Metrics to Track

- Data access requests
- Deletion requests
- Consent withdrawals
- Security incidents
- Integration activity
- Data breach attempts

---

## Contact Information

### Data Protection Officer

- **Email**: dpo@repz-platform.com
- **Response Time**: 48 hours
- **Escalation**: privacy@repz-platform.com

### Supervisory Authority

- **Name**: [Local Data Protection Authority]
- **Website**: [Authority Website]
- **Complaint Process**: [Link to Process]

---

## Resources

### External Links

- [GDPR Full Text](https://gdpr-info.eu/)
- [ICO Guidance](https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/)
- [Article 29 Working Party Guidelines](https://ec.europa.eu/newsroom/article29/items/612053)

### Internal Documentation

- Privacy Policy: `/docs/privacy-policy.md`
- Terms of Service: `/docs/terms-of-service.md`
- Security Policy: `/docs/SECURITY.md`
- Integration Guide: `/docs/INTEGRATIONS.md`

---

*Last Updated: January 18, 2025*
*Version: 1.0.0*
*Review Date: July 18, 2025*
