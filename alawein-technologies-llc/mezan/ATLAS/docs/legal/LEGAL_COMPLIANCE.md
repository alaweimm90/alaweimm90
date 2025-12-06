# ⚖️ Legal & Compliance Toolkit

**Essential legal templates and compliance checklists for launching a SaaS product.**

**⚠️ CRITICAL DISCLAIMER:**
This is NOT legal advice. These templates are starting points. You MUST:
1. Consult with a qualified attorney before launching
2. Customize templates for your specific situation
3. Stay updated on changing regulations

**Cost to do it right:** $2,000-5,000 for lawyer review (worth every penny to avoid $50K+ fines)

---

## 📋 Table of Contents

1. [GDPR Compliance Checklist](#gdpr-compliance-checklist)
2. [Privacy Policy Template](#privacy-policy-template)
3. [Terms of Service Template](#terms-of-service-template)
4. [Cookie Policy](#cookie-policy)
5. [Data Processing Agreement (DPA)](#data-processing-agreement)
6. [User Data Rights](#user-data-rights)
7. [Security Incident Response Plan](#security-incident-response-plan)
8. [Compliance Timeline](#compliance-timeline)

---

## 🛡️ GDPR Compliance Checklist

**What:** General Data Protection Regulation (EU law)
**Applies to:** Any service used by EU residents (even if you're US-based)
**Penalties:** Up to €20M or 4% of global revenue (whichever is higher)

### ✅ Pre-Launch Checklist

#### Legal Basis for Processing
- [ ] **Identify lawful basis** for each type of data processing:
  - [ ] Consent (marketing emails, analytics)
  - [ ] Contract (account creation, service delivery)
  - [ ] Legitimate interest (fraud prevention, security)
  - [ ] Legal obligation (tax records, law compliance)

- [ ] **Document why you collect each data point**:
  ```
  Email address: Required for account creation (contract)
  Name: Required for personalization (contract)
  IP address: Fraud prevention (legitimate interest)
  Usage analytics: Product improvement (consent via cookie banner)
  Payment info: Process transactions (contract)
  ```

#### Consent Management
- [ ] **Cookie consent banner** implemented (before loading analytics)
- [ ] **Granular consent options** (can opt-in to analytics separately from essential cookies)
- [ ] **Easy to withdraw consent** (one click in settings)
- [ ] **Consent logged** with timestamp, IP, what they consented to
- [ ] **Re-request consent** if you add new data collection

#### Privacy Policy
- [ ] **Privacy Policy page** written in plain language
- [ ] **Linked from:** Footer, signup form, cookie banner
- [ ] **Includes:**
  - [ ] What data you collect
  - [ ] Why you collect it
  - [ ] How long you store it
  - [ ] Who you share it with (subprocessors)
  - [ ] User rights (access, deletion, portability)
  - [ ] Contact information (DPO if required)
  - [ ] How to file complaints

#### User Rights Implementation
- [ ] **Right to Access** - users can download their data
  - [ ] Technical implementation: `/api/user/export` endpoint
  - [ ] Format: JSON or CSV download
  - [ ] Timeline: Within 30 days of request

- [ ] **Right to Deletion** - users can delete their account + data
  - [ ] Technical implementation: `/api/user/delete` endpoint
  - [ ] Process: Hard delete from all databases (or anonymize if legally required to retain)
  - [ ] Timeline: Within 30 days of request
  - [ ] Exceptions documented (e.g., financial records for tax purposes)

- [ ] **Right to Portability** - users can export data in machine-readable format
  - [ ] Format: JSON with clear schema
  - [ ] Includes: All user-generated content, preferences, history

- [ ] **Right to Rectification** - users can correct inaccurate data
  - [ ] Technical implementation: Settings page for profile edits

- [ ] **Right to Restrict Processing** - users can limit how you use their data
  - [ ] Technical implementation: Granular privacy settings

- [ ] **Right to Object** - users can opt out of marketing, profiling
  - [ ] Technical implementation: Unsubscribe links, opt-out in settings

#### Data Security
- [ ] **Encryption at rest** (database encryption enabled)
- [ ] **Encryption in transit** (HTTPS everywhere, TLS 1.2+)
- [ ] **Password hashing** (bcrypt with cost factor 12+)
- [ ] **Access controls** (role-based access, principle of least privilege)
- [ ] **Audit logs** (who accessed what data when)
- [ ] **Regular backups** (encrypted, tested recovery process)
- [ ] **Secure development practices** (no secrets in code, environment variables)

#### Data Breach Response
- [ ] **Breach notification process** defined (see Security Incident Response Plan below)
- [ ] **72-hour notification** to supervisory authority (if high risk to users)
- [ ] **User notification** process (if high risk to individual rights)
- [ ] **Contact:** Data Protection Authority in your jurisdiction

#### Third-Party Processors
- [ ] **List all subprocessors** (AWS, Stripe, SendGrid, analytics, etc.)
- [ ] **Data Processing Agreements (DPAs)** signed with each
- [ ] **Privacy Policy discloses** all third parties with access to data
- [ ] **Verify** each subprocessor is GDPR-compliant

**Common subprocessors for SaaS:**
```
Infrastructure:
- [ ] AWS/GCP/Azure - DPA: Available on their websites
- [ ] Vercel/Netlify - DPA: Available

Payment:
- [ ] Stripe - DPA: Automatic for EU customers

Email:
- [ ] SendGrid/Postmark - DPA: Available

Analytics:
- [ ] Google Analytics - Switch to GA4 (more GDPR-friendly) or use Plausible
- [ ] Mixpanel - DPA: Available

Support:
- [ ] Intercom/Zendesk - DPA: Available
```

#### Data Retention
- [ ] **Define retention periods** for each data type:
  ```
  Account data: As long as account is active + 30 days after deletion request
  Payment records: 7 years (tax law requirement)
  Analytics data: 24 months
  Logs: 90 days
  Marketing consent: Until withdrawn
  ```

- [ ] **Automated deletion** implemented for expired data

#### International Transfers
- [ ] **If storing data outside EU:**
  - [ ] Use Standard Contractual Clauses (SCCs)
  - [ ] Or: Use EU-based hosting (easiest option)
  - [ ] Document legal basis for transfer

#### Documentation
- [ ] **Record of Processing Activities** (ROPA) maintained:
  ```
  Data Type | Purpose | Legal Basis | Retention | Recipients
  Email     | Account mgmt | Contract | Active+30d | SendGrid
  Name      | Personalization | Contract | Active+30d | None
  IP        | Fraud prev. | Legit interest | 90 days | AWS
  ```

- [ ] **Privacy Impact Assessment** (if processing high-risk data like health, biometric)

---

## 📄 Privacy Policy Template

```markdown
# Privacy Policy

**Last Updated:** [DATE]

## Introduction

[COMPANY NAME] ("we," "our," or "us") operates [WEBSITE/APP] (the "Service"). This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our Service.

**By using the Service, you agree to the collection and use of information in accordance with this policy.**

---

## 1. Information We Collect

### 1.1 Information You Provide

**Account Information:**
- Email address (required for account creation)
- Name (optional, for personalization)
- Password (hashed, never stored in plain text)

**Payment Information:**
- We use [Stripe/PayPal] to process payments. We do NOT store full credit card numbers.
- We store: Last 4 digits of card, expiration date, billing address

**User Content:**
- [Papers you upload, research hypotheses, notes, etc. - BE SPECIFIC]
- Feedback and support messages

### 1.2 Information Collected Automatically

**Usage Data:**
- Pages visited, features used, time spent
- Device information (browser type, OS, screen size)
- IP address (for fraud prevention and analytics)

**Cookies and Tracking:**
- Essential cookies (required for Service to function)
- Analytics cookies (with your consent, to improve the Service)
- See our [Cookie Policy](#cookie-policy) for details

---

## 2. How We Use Your Information

We use your information for the following purposes:

| Purpose | Legal Basis (GDPR) | Data Used |
|---------|-------------------|-----------|
| Provide the Service | Contract | Account info, content |
| Process payments | Contract | Payment info |
| Send transactional emails | Contract | Email address |
| Customer support | Contract | Email, support messages |
| Improve the Service | Consent | Usage analytics, feedback |
| Prevent fraud | Legitimate interest | IP address, usage patterns |
| Marketing emails | Consent | Email address |

**We will NEVER:**
- Sell your data to third parties
- Use your research content to train AI models shared with others (your data is private)
- Share your email with marketers

---

## 3. Data Sharing and Disclosure

### 3.1 Third-Party Service Providers

We share data with the following subprocessors:

| Provider | Purpose | Data Shared | Privacy Policy |
|----------|---------|-------------|----------------|
| AWS | Hosting | All data | [AWS Privacy](https://aws.amazon.com/privacy/) |
| Stripe | Payment processing | Payment info | [Stripe Privacy](https://stripe.com/privacy) |
| SendGrid | Transactional emails | Email, name | [SendGrid Privacy](https://www.twilio.com/legal/privacy) |
| [Analytics Tool] | Usage analytics | Usage data, IP | [Link] |

**All subprocessors have signed Data Processing Agreements (DPAs) and are GDPR-compliant.**

### 3.2 Legal Requirements

We may disclose your information if required by law or in response to:
- Court orders or subpoenas
- Government or law enforcement requests
- Protection of our legal rights

### 3.3 Business Transfers

If we are acquired or merge with another company, your data may be transferred. You will be notified via email and/or a prominent notice on our Service.

---

## 4. Data Retention

We retain your data for the following periods:

| Data Type | Retention Period | Reason |
|-----------|------------------|--------|
| Account data | Active + 30 days after deletion | Service provision |
| Payment records | 7 years | Tax/legal requirements |
| Usage analytics | 24 months | Product improvement |
| Support tickets | 2 years | Reference and quality |
| Marketing consent | Until withdrawn | Legal requirement |

After these periods, data is permanently deleted or anonymized.

---

## 5. Your Rights (GDPR & CCPA)

### 5.1 Access
You can download all your data at any time:
- Go to Settings → Privacy → Download My Data
- Receives JSON file with all your information

### 5.2 Deletion
You can delete your account and all associated data:
- Go to Settings → Account → Delete Account
- Confirms via email
- All data deleted within 30 days (except legally required records)

### 5.3 Rectification
You can update your information:
- Go to Settings → Profile to edit name, email, etc.

### 5.4 Portability
You can export your data in machine-readable format (JSON)

### 5.5 Withdrawal of Consent
You can withdraw consent for marketing and analytics:
- Marketing: Click "Unsubscribe" in any email
- Analytics: Go to Settings → Privacy → Cookie Preferences

### 5.6 Complaints
You have the right to lodge a complaint with a supervisory authority:
- EU: [Your country's Data Protection Authority]
- US (California): California Attorney General

---

## 6. Data Security

We implement industry-standard security measures:

**Technical Measures:**
- Encryption in transit (HTTPS/TLS 1.3)
- Encryption at rest (AES-256)
- Password hashing (bcrypt, cost factor 12)
- Regular security audits

**Organizational Measures:**
- Access controls (role-based, least privilege)
- Employee training on data protection
- Audit logs for all data access
- Incident response plan

**No system is 100% secure.** If we discover a breach, we will notify affected users within 72 hours.

---

## 7. International Data Transfers

Your data is stored in [REGION - e.g., "EU (Frankfurt)" or "US (Virginia)"].

If you are in the EU and your data is stored outside the EU, we use:
- [ ] Standard Contractual Clauses (SCCs) approved by the EU Commission
- [ ] Or: We only use EU-based hosting to avoid transfers

---

## 8. Children's Privacy

Our Service is NOT intended for users under 16 (or 13 in the US).

We do not knowingly collect data from children. If you believe we have collected data from a child, contact us immediately at [EMAIL] and we will delete it.

---

## 9. Changes to This Policy

We may update this Privacy Policy from time to time. We will:
- Notify you via email (to the address on your account)
- Post a notice on the Service
- Update the "Last Updated" date at the top

**Continued use of the Service after changes constitutes acceptance.**

---

## 10. Contact Us

**For privacy-related questions, requests, or complaints:**

Email: privacy@[YOUR-DOMAIN]
Address: [YOUR REGISTERED BUSINESS ADDRESS]

**Data Protection Officer (if required):**
[NAME]
Email: dpo@[YOUR-DOMAIN]

**Response time:** We will respond to privacy requests within 30 days.

---

**Effective Date:** [DATE]
```

---

## 📜 Terms of Service Template

```markdown
# Terms of Service

**Last Updated:** [DATE]

**PLEASE READ THESE TERMS CAREFULLY BEFORE USING THE SERVICE.**

---

## 1. Acceptance of Terms

By accessing or using [SERVICE NAME] (the "Service"), you agree to be bound by these Terms of Service ("Terms"). If you do not agree, do not use the Service.

---

## 2. Description of Service

[SERVICE NAME] provides [DESCRIPTION - e.g., "AI-powered research validation tools"].

**The Service includes:**
- [Feature 1]
- [Feature 2]
- [Feature 3]

**We reserve the right to:**
- Modify or discontinue features at any time
- Refuse service to anyone for any reason

---

## 3. Account Registration

### 3.1 Eligibility
You must be at least 16 years old (or 13 in the US with parental consent).

### 3.2 Account Security
- You are responsible for maintaining the confidentiality of your password
- You are responsible for all activity under your account
- Notify us immediately of unauthorized access: security@[DOMAIN]

### 3.3 Prohibited Uses
You may NOT use the Service to:
- Violate any laws or regulations
- Infringe intellectual property rights
- Upload malicious code (viruses, malware)
- Harass or harm others
- Scrape or data mine without permission
- Resell or redistribute the Service
- Use for any illegal purpose

**Violation may result in immediate termination of your account.**

---

## 4. User Content

### 4.1 Ownership
You retain all rights to content you upload ("User Content"). We do not claim ownership.

### 4.2 License Grant
By uploading User Content, you grant us a limited license to:
- Store and process your content to provide the Service
- Create backups for disaster recovery
- Display your content to you and authorized users

**We will NOT:**
- Use your content to train AI models shared with other users
- Sell or share your content with third parties (except subprocessors like AWS for storage)
- Publicly display your content without permission

### 4.3 Responsibility
You are solely responsible for your User Content. You represent that:
- You own the content or have the right to upload it
- The content does not violate any laws or third-party rights
- The content does not contain malware or illegal material

### 4.4 Prohibited Content
You may not upload:
- Copyrighted material you don't have rights to
- Personal information of others without consent
- Malicious code or exploits
- Illegal content (child abuse material, terrorist content, etc.)

**We reserve the right to remove any content that violates these Terms.**

---

## 5. Payments and Subscriptions

### 5.1 Pricing
Current pricing is available at [PRICING PAGE].

We reserve the right to change pricing with 30 days' notice.

### 5.2 Billing
- Subscriptions are billed monthly or annually (based on your selection)
- Billing occurs on the same day each month
- Payment is processed by [Stripe/PayPal]
- All fees are in USD and non-refundable (except as required by law)

### 5.3 Cancellation
- You may cancel at any time: Settings → Subscription → Cancel
- Cancellation takes effect at the end of the current billing period
- You retain access until the end of the paid period
- No refunds for partial months

### 5.4 Failed Payments
If payment fails:
- Day 1: Email notification, retry payment
- Day 7: Second retry
- Day 14: Account downgraded to free tier (if available) or suspended

---

## 6. Intellectual Property

### 6.1 Our IP
The Service, including all software, designs, and trademarks, is owned by [COMPANY NAME].

You may NOT:
- Copy, modify, or distribute our software
- Reverse engineer the Service
- Use our trademarks without permission

### 6.2 Feedback
If you provide feedback or suggestions, we may use them without compensation or attribution.

---

## 7. Disclaimers and Limitations

### 7.1 "AS IS" Service
THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND.

We do not guarantee:
- The Service will be error-free or uninterrupted
- Results will be accurate or reliable (especially AI-generated content)
- The Service will meet your specific requirements

**You use the Service at your own risk.**

### 7.2 Limitation of Liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW:

- OUR TOTAL LIABILITY IS LIMITED TO THE AMOUNT YOU PAID IN THE PAST 12 MONTHS
- WE ARE NOT LIABLE FOR INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
- WE ARE NOT LIABLE FOR DATA LOSS (maintain your own backups!)

**Some jurisdictions do not allow these limitations, so they may not apply to you.**

### 7.3 Indemnification
You agree to defend and indemnify us from any claims arising from:
- Your use of the Service
- Your User Content
- Your violation of these Terms

---

## 8. Termination

### 8.1 Termination by You
You may terminate at any time by:
- Canceling your subscription (Settings → Subscription → Cancel)
- Deleting your account (Settings → Account → Delete Account)

### 8.2 Termination by Us
We may terminate or suspend your account immediately if:
- You violate these Terms
- You engage in fraudulent activity
- Required by law
- We discontinue the Service

**We will provide 30 days' notice for non-violation terminations (when possible).**

### 8.3 Effect of Termination
Upon termination:
- Your access to the Service ends immediately
- Your data will be deleted within 30 days (see Privacy Policy)
- No refunds for remaining subscription time (except as required by law)

---

## 9. Dispute Resolution

### 9.1 Governing Law
These Terms are governed by the laws of [YOUR JURISDICTION - e.g., "California, USA" or "Ireland"].

### 9.2 Arbitration (US)
**For US users:** Disputes will be resolved by binding arbitration (not court), except:
- Small claims court (under $10,000)
- Intellectual property disputes

**Arbitration rules:** American Arbitration Association (AAA)

**NO CLASS ACTIONS:** You agree to resolve disputes individually, not as part of a class action.

### 9.3 EU Users
EU users have the right to resolve disputes through:
- Courts in their country of residence
- Online Dispute Resolution (ODR) platform: https://ec.europa.eu/consumers/odr

---

## 10. Miscellaneous

### 10.1 Changes to Terms
We may update these Terms from time to time:
- We will notify you via email and/or a notice on the Service
- Continued use after changes = acceptance
- If you don't agree, cancel your account

### 10.2 Severability
If any provision is found invalid, the rest of these Terms remain in effect.

### 10.3 No Waiver
Our failure to enforce any right does not waive that right.

### 10.4 Entire Agreement
These Terms (+ Privacy Policy) constitute the entire agreement between you and us.

---

## 11. Contact

**For questions about these Terms:**

Email: legal@[YOUR-DOMAIN]
Address: [YOUR REGISTERED BUSINESS ADDRESS]

---

**Effective Date:** [DATE]
```

---

## 🍪 Cookie Policy

```markdown
# Cookie Policy

**Last Updated:** [DATE]

## What Are Cookies?

Cookies are small text files stored on your device when you visit our website. They help the website remember your preferences and improve your experience.

---

## Types of Cookies We Use

### 1. Essential Cookies (Always Active)
**Purpose:** Required for the website to function

| Cookie Name | Purpose | Expiration |
|-------------|---------|------------|
| `session_id` | Keeps you logged in | 30 days |
| `csrf_token` | Security (prevents CSRF attacks) | Session |
| `cookie_consent` | Remembers your cookie preferences | 1 year |

**Legal Basis:** Necessary for contract performance (GDPR Art. 6(1)(b))

**You cannot disable these without breaking the website.**

---

### 2. Analytics Cookies (Optional - Requires Consent)
**Purpose:** Help us understand how you use the Service

| Cookie Name | Purpose | Provider | Expiration |
|-------------|---------|----------|------------|
| `_ga` | Google Analytics - user identification | Google | 2 years |
| `_gid` | Google Analytics - session tracking | Google | 24 hours |

**Legal Basis:** Consent (GDPR Art. 6(1)(a))

**Data collected:** Pages visited, time on site, browser type, anonymous user ID

**Privacy-friendly alternative:** We're considering switching to Plausible (no cookies, fully anonymous).

---

### 3. Advertising Cookies (We Don't Use These)
We do NOT use advertising cookies or third-party ad networks.

---

## Managing Cookies

### Option 1: Cookie Preferences (Recommended)
Click the "Cookie Preferences" link in the footer to:
- Accept all cookies
- Accept essential cookies only
- Customize which cookies you allow

### Option 2: Browser Settings
You can block cookies in your browser:
- Chrome: Settings → Privacy and Security → Cookies
- Firefox: Preferences → Privacy & Security → Cookies
- Safari: Preferences → Privacy → Cookies

**Warning:** Blocking essential cookies will break the website.

### Option 3: Opt Out of Google Analytics
Use Google's opt-out browser extension: https://tools.google.com/dlpage/gaoptout

---

## Third-Party Cookies

We do not control cookies set by:
- Stripe (payment processing)
- OAuth providers (if you log in with Google, etc.)

Refer to their privacy policies for details.

---

## Changes to This Policy

We will notify you of changes via:
- Update to "Last Updated" date
- Banner on the website (for material changes)

---

## Contact

Questions about cookies? Email: privacy@[YOUR-DOMAIN]
```

---

## 📝 Data Processing Agreement (DPA) Template

**Use this when customers ask for a DPA (common for B2B sales)**

```markdown
# Data Processing Agreement

This Data Processing Agreement ("DPA") is entered into between:

**Data Controller:** [CUSTOMER NAME]
**Data Processor:** [YOUR COMPANY NAME]

**Effective Date:** [DATE]

---

## 1. Definitions

- **Controller:** The entity that determines the purposes and means of processing Personal Data (the Customer)
- **Processor:** The entity that processes Personal Data on behalf of the Controller ([YOUR COMPANY])
- **Personal Data:** Any information relating to an identified or identifiable natural person
- **Processing:** Any operation performed on Personal Data (collection, storage, use, disclosure, deletion)

---

## 2. Scope of Processing

**Nature and Purpose:**
Processing is necessary to provide the Service as described in the Terms of Service.

**Duration:**
For the duration of the Service Agreement.

**Types of Personal Data:**
- Email addresses
- Names
- [Other data types specific to your service]

**Categories of Data Subjects:**
- The Controller's employees and authorized users

---

## 3. Processor's Obligations

The Processor shall:

1. **Process only on instructions** from the Controller (via use of the Service)
2. **Ensure confidentiality** of personnel processing data
3. **Implement appropriate security measures** (see Section 5)
4. **Engage sub-processors** only with Controller's consent (see Section 4)
5. **Assist with data subject rights** requests (access, deletion, etc.)
6. **Assist with security breaches** and notifications
7. **Delete or return data** upon termination (unless legally required to retain)
8. **Make available** all information necessary to demonstrate compliance

---

## 4. Sub-Processors

The Controller authorizes the use of the following sub-processors:

| Sub-Processor | Service | Location | DPA Link |
|---------------|---------|----------|----------|
| AWS | Hosting | [EU/US] | [Link] |
| Stripe | Payments | US | [Link] |
| SendGrid | Email | US | [Link] |

**Changes:** We will notify Controller 30 days before adding new sub-processors. Controller may object within 30 days.

---

## 5. Security Measures

The Processor implements the following measures:

**Technical:**
- Encryption in transit (TLS 1.3)
- Encryption at rest (AES-256)
- Password hashing (bcrypt)
- Access controls (role-based)
- Audit logging

**Organizational:**
- Employee confidentiality agreements
- Security training
- Background checks for personnel with data access
- Incident response plan

---

## 6. Data Breach Notification

In the event of a Personal Data breach, the Processor shall:
1. Notify the Controller **within 72 hours** of becoming aware
2. Provide details: nature of breach, affected data, likely consequences, mitigation measures
3. Cooperate with Controller's investigation

---

## 7. Data Subject Rights

The Processor shall assist the Controller in responding to data subject requests:
- Right to access
- Right to rectification
- Right to erasure
- Right to data portability
- Right to restriction of processing
- Right to object

**Assistance provided within 30 days of Controller's request.**

---

## 8. Audits

The Controller has the right to:
- Request information demonstrating compliance
- Conduct audits (with reasonable notice, max once per year)

**Processor will provide:** SOC 2 reports, security certifications, or equivalent.

---

## 9. Data Return and Deletion

Upon termination of the Service Agreement:
- Processor will delete all Personal Data within **30 days**
- Exception: Data legally required to be retained (e.g., financial records)
- Controller may request data export before deletion (in machine-readable format)

---

## 10. Liability and Indemnification

The Processor's liability is limited to the amounts specified in the Service Agreement.

The Controller shall indemnify the Processor for claims arising from Controller's instructions.

---

## 11. Governing Law

This DPA is governed by the same law as the Service Agreement.

---

## 12. Signatures

**Data Controller:**

Signature: _________________
Name: _________________
Title: _________________
Date: _________________

**Data Processor:**

Signature: _________________
Name: _________________
Title: _________________
Date: _________________
```

---

## 🚨 Security Incident Response Plan

### 1. Detection (Hour 0)

**How you might detect a breach:**
- Monitoring alerts (unusual data access, failed login attempts)
- User reports (suspicious emails, unauthorized access)
- Third-party notification (e.g., "AWS detected suspicious activity")
- Penetration test findings

**Immediate actions:**
1. **DO NOT PANIC** - follow this plan
2. **Assemble incident team:**
   - Tech lead (investigate technical details)
   - Legal counsel (assess legal obligations)
   - Communications lead (draft user notifications)
   - CEO/founder (final decisions)

3. **Secure evidence:**
   - Take database snapshots
   - Save logs (don't modify!)
   - Document timeline

---

### 2. Assessment (Hours 1-4)

**Answer these questions:**
1. **What happened?**
   - Unauthorized access? Data leak? Malware? DDoS?

2. **What data was affected?**
   - Personal data (names, emails)? Sensitive data (passwords, payment info)? How many users?

3. **How did it happen?**
   - Vulnerability exploited? Compromised credentials? Insider threat?

4. **Is the threat contained?**
   - No: Containment is priority #1 (e.g., revoke API keys, block IPs, take service offline if necessary)
   - Yes: Proceed to impact analysis

---

### 3. Containment (Hours 1-6)

**Immediate actions:**
- [ ] **Isolate affected systems** (prevent spread)
- [ ] **Revoke compromised credentials** (API keys, passwords, access tokens)
- [ ] **Block attack vectors** (firewall rules, disable vulnerable endpoints)
- [ ] **Preserve evidence** (logs, snapshots - don't delete!)

**Example actions for common incidents:**

| Incident Type | Containment Actions |
|---------------|-------------------|
| Database breach | Rotate DB credentials, restrict IP access, audit access logs |
| API key leaked | Revoke key, audit all requests made with it, rotate all keys |
| Malware detected | Isolate infected systems, run antivirus, restore from clean backup |
| DDoS attack | Enable rate limiting, use CDN/DDoS protection (Cloudflare) |
| Phishing attack | Notify affected users, force password reset, enable 2FA |

---

### 4. Legal Assessment (Hours 4-12)

**Consult with legal counsel to determine:**

1. **Is this a "Personal Data Breach" under GDPR?**
   - Yes if: Personal data was accessed, disclosed, or lost

2. **Is notification required?**

**GDPR Requirements:**
- **Notify supervisory authority** within **72 hours** if:
  - Breach is likely to result in a risk to individuals' rights and freedoms
  - Exceptions: Data was encrypted/pseudonymized, or risk is unlikely

- **Notify affected individuals** "without undue delay" if:
  - Breach is likely to result in a **high risk** to their rights and freedoms

**US State Laws (CCPA, etc.):**
- Requirements vary by state
- Generally: notify if unencrypted data was accessed

---

### 5. Notification (Hours 12-72)

#### A. Notify Supervisory Authority (GDPR - if required)

**Who:** Your country's Data Protection Authority
- Ireland (if you're EU-based): Data Protection Commission
- US companies often use Ireland DPC if they process EU data

**How:** Online form on their website

**What to include:**
```
1. Nature of the breach (what happened)
2. Categories and number of affected data subjects
3. Categories and number of affected personal data records
4. Contact point for more information (DPO email)
5. Likely consequences of the breach
6. Measures taken or proposed to address the breach
```

**Template:**
```
Subject: Personal Data Breach Notification - [YOUR COMPANY]

1. Nature of Breach:
On [DATE], we discovered unauthorized access to our database containing user email addresses and hashed passwords.

2. Affected Data Subjects:
Approximately [NUMBER] users in the EU

3. Affected Personal Data:
- Email addresses: [NUMBER] records
- Hashed passwords: [NUMBER] records (bcrypt, cost factor 12)
- No plain-text passwords, payment info, or other sensitive data was accessed

4. Contact Point:
[DPO NAME]
Email: dpo@[DOMAIN]
Phone: [NUMBER]

5. Likely Consequences:
Risk of phishing emails to affected users. Password risk is low (strong hashing).

6. Mitigation Measures:
- Forced password reset for all affected users
- Notified affected users via email
- Implemented additional access controls (IP whitelisting, 2FA required)
- Engaged third-party security firm to audit systems
```

---

#### B. Notify Affected Users (if required)

**When:** If high risk to individual rights (e.g., SSNs leaked, weak password hashing, financial data accessed)

**How:** Email to affected users

**Template:**
```
Subject: Important Security Notice - Action Required

Dear [Name],

We are writing to inform you of a security incident that may have affected your account.

**What Happened:**
On [DATE], we discovered that [BRIEF DESCRIPTION - e.g., "an unauthorized party gained access to our database"].

**What Information Was Involved:**
[SPECIFIC DATA - e.g., "Your email address and password (which was encrypted)".]

**What Was NOT Involved:**
[e.g., "Your payment information, research content, and other personal data were NOT accessed."]

**What We're Doing:**
- We have secured the vulnerability that allowed this access
- We have forced a password reset for all affected accounts
- We have engaged a third-party security firm to audit our systems
- We have notified the relevant authorities

**What You Should Do:**
1. Reset your password immediately: [RESET LINK]
2. Enable two-factor authentication: [2FA SETUP LINK]
3. Watch for suspicious emails (we will never ask for your password via email)
4. If you used the same password on other sites, change it there too

**More Information:**
[LINK TO FAQ PAGE WITH DETAILS]

**Questions?**
Contact our security team: security@[DOMAIN]

We sincerely apologize for this incident and are committed to protecting your data.

Sincerely,
[CEO/FOUNDER NAME]
[TITLE]
[COMPANY]
```

---

### 6. Remediation (Days 1-30)

**Fix the root cause:**
- [ ] Patch vulnerability
- [ ] Implement additional security controls
- [ ] Conduct security audit (consider third-party penetration test)
- [ ] Review and improve incident response plan

**Long-term improvements:**
- [ ] Implement bug bounty program
- [ ] Regular security training for team
- [ ] Automated security scanning (Dependabot, Snyk)
- [ ] Regular penetration testing

---

### 7. Post-Incident Review (Day 30)

**Questions to answer:**
1. What went wrong? (Root cause)
2. What went right? (What we did well)
3. What should we change? (Process improvements)
4. Who needs additional training?
5. What tools/processes should we implement?

**Document findings and update this plan.**

---

## 📅 Compliance Timeline

**Use this timeline to ensure you launch legally:**

### 4 Weeks Before Launch
- [ ] Draft Privacy Policy (use template above)
- [ ] Draft Terms of Service (use template above)
- [ ] Draft Cookie Policy (use template above)
- [ ] Identify all third-party subprocessors
- [ ] Sign DPAs with subprocessors (AWS, Stripe, etc.)
- [ ] Implement cookie consent banner
- [ ] Implement data export functionality (`/api/user/export`)
- [ ] Implement account deletion functionality (`/api/user/delete`)
- [ ] Set up GDPR-compliant analytics (Google Analytics 4 or Plausible)

### 2 Weeks Before Launch
- [ ] **Hire lawyer** to review Privacy Policy, Terms of Service (~$1,000-2,000)
- [ ] Implement security measures (HTTPS, encryption, password hashing)
- [ ] Set up security monitoring (AWS CloudWatch, Sentry for errors)
- [ ] Create incident response team contact list
- [ ] Test data export/deletion flows

### 1 Week Before Launch
- [ ] Final legal review
- [ ] Publish Privacy Policy, Terms, Cookie Policy to website
- [ ] Link from footer, signup page
- [ ] Test cookie consent banner
- [ ] Ensure all forms have Privacy Policy checkbox
- [ ] Document "Record of Processing Activities" (ROPA)

### Launch Day
- [ ] Double-check all legal pages are live and linked
- [ ] Verify cookie consent works
- [ ] Monitor for security issues

### Post-Launch (Ongoing)
- [ ] Respond to data access/deletion requests within 30 days
- [ ] Review and update Privacy Policy annually (or when changes occur)
- [ ] Monitor for security incidents
- [ ] Renew DPAs with subprocessors annually
- [ ] Conduct annual security audit

---

## 🎓 Compliance Checklist (Full)

**Use this master checklist to ensure full compliance:**

### Legal Documents
- [ ] Privacy Policy (published, linked from footer and signup)
- [ ] Terms of Service (published, linked from footer and signup)
- [ ] Cookie Policy (published, linked from footer)
- [ ] DPAs signed with all subprocessors

### User Rights Implementation
- [ ] Data export (downloadable JSON/CSV)
- [ ] Account deletion (hard delete within 30 days)
- [ ] Data rectification (editable profile settings)
- [ ] Consent withdrawal (cookie preferences, email unsubscribe)

### Consent Management
- [ ] Cookie consent banner (before loading analytics)
- [ ] Granular consent options (analytics separate from essential)
- [ ] Consent logged (timestamp, IP, what was consented to)
- [ ] Easy to withdraw consent (one-click in settings)

### Security
- [ ] HTTPS everywhere (TLS 1.2+)
- [ ] Database encryption at rest
- [ ] Password hashing (bcrypt, cost 12+)
- [ ] API authentication (JWT or sessions)
- [ ] Rate limiting (prevent brute force)
- [ ] Security headers (CSP, X-Frame-Options, etc.)
- [ ] Regular backups (encrypted, tested restore)
- [ ] Monitoring and alerting (intrusion detection)

### Data Management
- [ ] Data retention policy defined
- [ ] Automated deletion for expired data
- [ ] Audit logs (who accessed what when)
- [ ] Access controls (role-based, least privilege)

### Incident Response
- [ ] Security incident response plan (see above)
- [ ] Incident response team identified
- [ ] Security contact email (security@domain)
- [ ] DPA contact email (privacy@domain or dpo@domain)

### Third-Party Management
- [ ] List of all subprocessors documented
- [ ] DPAs signed with each
- [ ] Privacy Policy discloses all subprocessors
- [ ] Process to notify customers of new subprocessors

### Documentation
- [ ] Record of Processing Activities (ROPA)
- [ ] Legal basis for each type of processing documented
- [ ] Data retention schedule documented
- [ ] Privacy Impact Assessment (if processing high-risk data)

---

## 💰 Legal Budget

**Realistic costs for compliance:**

| Item | Cost | Notes |
|------|------|-------|
| Lawyer review (Privacy Policy, Terms) | $1,000-2,500 | One-time |
| DPA templates | $0-500 | Often free from subprocessors |
| Cookie consent tool | $0-50/mo | Free: Cookiebot free tier; Paid: Termly |
| Security audit | $2,000-10,000 | Annual or pre-launch |
| Penetration test | $3,000-15,000 | Annual recommended |
| Bug bounty program | $500+/mo | Ongoing (HackerOne, Bugcrowd) |
| Legal retainer (ongoing) | $500-2,000/mo | For questions, updates, contract review |

**Minimum to launch:** $1,500-3,000
**Recommended for year 1:** $5,000-10,000

**This is cheap insurance against:**
- GDPR fines: Up to €20M or 4% of revenue
- US state fines: $2,500-7,500 per violation (can add up fast)
- Customer trust loss (priceless)

---

## 🏁 Summary

**Legal compliance is not optional. It's:**
1. **Legally required** (fines up to millions)
2. **Customer expectation** (builds trust)
3. **Good business** (avoid lawsuits, data breaches)

**Minimum to launch:**
✅ Privacy Policy
✅ Terms of Service
✅ Cookie consent
✅ User data export/deletion
✅ Basic security (HTTPS, encryption, password hashing)

**Recommended:**
✅ All of the above
✅ Lawyer review ($1,500-2,500)
✅ DPAs with subprocessors
✅ Security audit
✅ Incident response plan

---

## 🎯 Your Next Action

1. **Copy the Privacy Policy template** above
2. **Customize** with your company name, data types, subprocessors
3. **Schedule consult with lawyer** ($200-300 for initial call)
4. **Implement cookie consent** (free tool: Cookiebot)
5. **Build data export/deletion** endpoints

**Don't launch without legal compliance. It's not worth the risk.**

---

**Built for builders who launch legally.** ⚖️

*Now go get compliant!*
