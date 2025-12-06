# Attributa.dev Privacy Policy

*Last updated: August 10, 2025*

## Overview

Attributa.dev is a privacy-first attribution intelligence tool designed to help developers build ethical AI. We are committed to protecting your privacy and maintaining transparency about our data practices.

## Data Collection & Processing

### Local-First Architecture
- **Primary Mode**: All analysis runs locally in your browser using WebAssembly models
- **No Server Processing**: Text content is not transmitted to external servers by default
- **Browser Storage**: Analysis results are stored locally in your browser's IndexedDB

### Optional External Services
When you explicitly enable external APIs and provide API keys:
- **Crossref API**: Used for citation validation (with rate limiting and proper attribution)
- **External Models**: Only when you opt-in and configure API keys

### Data We Don't Collect
- ❌ Personal identification information
- ❌ Content of your documents (unless you opt-in to external APIs)
- ❌ Browsing history or analytics tracking
- ❌ Location data
- ❌ Device fingerprinting

### Data We May Collect
- ✅ Error reports (anonymous, no content included)
- ✅ Performance metrics (anonymous)
- ✅ Feature usage statistics (anonymous)

## Third-Party Services

### Crossref API
- **Purpose**: Citation validation and metadata lookup
- **Data Sent**: DOI identifiers only (not full content)
- **User-Agent**: "Attributa/1.0 (mailto:support@attributa.dev)"
- **Privacy**: Respects Crossref's rate limits and terms of service

### External AI APIs (Optional)
- **OpenAI, Anthropic, etc.**: Only when you provide API keys
- **Your Responsibility**: Review third-party privacy policies
- **Control**: You can disable external APIs at any time

## Data Storage

### Local Storage
- Analysis results stored in browser IndexedDB
- API keys stored locally (encrypted in browser storage)
- Settings and preferences stored locally

### No Cloud Storage
- We do not store your content on our servers
- No backups or synchronization to external services
- Data remains on your device

## Data Sharing

We do not share, sell, or transfer your data to third parties, except:
- When required by law
- When you explicitly use external APIs with your own keys

## Security

### Technical Measures
- Local processing minimizes data exposure
- HTTPS encryption for all communications
- No server-side data storage
- Regular security audits of dependencies

### Your Security
- Keep your API keys secure
- Use strong, unique passwords for any accounts
- Report security issues to: security@attributa.dev

## Your Rights

### Control Your Data
- **Access**: View all data stored locally in your browser
- **Delete**: Clear workspace and browser storage at any time
- **Export**: Download your analysis results
- **Opt-out**: Disable external services and analytics

### Contact Us
- **Email**: privacy@attributa.dev
- **Support**: support@attributa.dev
- **Security**: security@attributa.dev

## Changes to This Policy

We will notify users of material changes through:
- Updates to this page with new "Last updated" date
- Notice in the application interface
- Email notification (if you've provided contact information)

## Children's Privacy

Attributa.dev is not intended for children under 13. We do not knowingly collect personal information from children.

## International Users

As a privacy-first tool with local processing, Attributa.dev respects international privacy laws including GDPR, CCPA, and similar regulations.

## Open Source

Attributa.dev is open source. You can review our code and privacy practices at: https://github.com/alawein/Attributa

---

*For questions about this privacy policy, contact us at privacy@attributa.dev*