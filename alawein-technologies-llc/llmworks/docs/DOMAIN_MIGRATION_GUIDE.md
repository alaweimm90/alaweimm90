# Domain Migration Guide: aegis-ai.com → llmworks.dev

## Overview
This guide covers the technical implementation for migrating from the old domain to llmworks.dev while preserving SEO value and user experience.

## Pre-Migration Checklist

### 1. Domain Setup
- [ ] Register llmworks.dev domain
- [ ] Configure DNS records (A, CNAME, MX if needed)
- [ ] Set up SSL certificate for llmworks.dev
- [ ] Configure CDN (Cloudflare, etc.) if used

### 2. Email & Authentication
- [ ] Update SPF/DKIM/DMARC records for new domain
- [ ] Configure email forwarding from old domain
- [ ] Update OAuth app configurations (Google, GitHub, etc.)
- [ ] Update Supabase project URL if needed

### 3. Third-Party Services
- [ ] Update Google Analytics property
- [ ] Update Google Search Console property
- [ ] Reconfigure social login providers
- [ ] Update API webhook URLs
- [ ] Update monitoring/logging services

## Implementation Steps

### Phase 1: Soft Launch (Week 1)
1. Deploy site to llmworks.dev subdomain/staging
2. Test all functionality on new domain
3. Update internal links and references
4. Configure redirects (see files created)

### Phase 2: Redirect Setup (Week 2)  
1. Implement redirects from old domain
2. Update sitemap.xml with new URLs
3. Submit new sitemap to Google Search Console
4. Monitor redirect functionality

### Phase 3: Communication (Week 3)
1. Announce migration to users via email
2. Update social media profiles/handles
3. Update documentation and README files
4. Notify partners and integrators

### Phase 4: Cleanup (Week 4)
1. Monitor traffic and fix any redirect issues
2. Update remaining external references
3. Archive old domain (keep redirects active)
4. Document lessons learned

## Redirect Strategy

### Server Configuration Files Created
- `public/_redirects` - For Netlify/Vercel deployments
- `public/.htaccess` - For Apache servers
- Both include security headers and SPA routing support

### Redirect Mapping
```
Old Pattern              New Pattern              Status
aegis-ai.com/*          llmworks.dev/*           301
app.aegis-ai.com/*      llmworks.dev/*           301  
/aegis/*                /arena/*                 301
/aegis-ai/*             /*                       301
/platform/*             /*                       301
/docs/aegis/*           /docs/*                  301
```

## SEO Migration Plan

### 1. Google Search Console
```bash
# Add new property
1. Add llmworks.dev as new property
2. Verify ownership via DNS TXT record
3. Submit new sitemap.xml
4. Set preferred domain to llmworks.dev
```

### 2. Sitemap Updates
- Update all URLs in sitemap.xml
- Include canonical URLs pointing to new domain
- Monitor crawl errors in Search Console

### 3. Backlink Preservation
- Keep 301 redirects permanent (minimum 1 year)
- Reach out to major referring sites for updates
- Monitor referring domain reports

## Analytics Migration

### Google Analytics 4
```javascript
// Update tracking code
gtag('config', 'GA_MEASUREMENT_ID', {
  'custom_map.page_location': 'llmworks.dev'
});

// Set up cross-domain tracking if needed
gtag('config', 'GA_MEASUREMENT_ID', {
  'linker': {
    'domains': ['llmworks.dev', 'aegis-ai.com']
  }
});
```

### Custom Events
```javascript
// Track migration-related events
gtag('event', 'domain_migration', {
  'event_category': 'migration',
  'event_label': 'old_to_new_domain',
  'value': 1
});
```

## Testing Checklist

### Functional Testing
- [ ] All pages load correctly on new domain
- [ ] Authentication/login works
- [ ] API endpoints respond correctly  
- [ ] File uploads/downloads function
- [ ] Email notifications use new domain

### Redirect Testing
```bash
# Test key redirects
curl -I https://aegis-ai.com
curl -I https://aegis-ai.com/arena
curl -I https://aegis-ai.com/docs

# Should return 301 responses to llmworks.dev
```

### Performance Testing
- [ ] Page load times comparable to old domain
- [ ] CDN configuration working
- [ ] SSL certificate valid
- [ ] Mobile responsiveness intact

## Rollback Plan

### Emergency Rollback (if critical issues)
1. Update DNS to point back to old domain
2. Disable redirects temporarily
3. Restore old analytics tracking
4. Communicate issue to users

### File Locations for Rollback
- DNS records (restore A/CNAME)
- `public/_redirects` (comment out redirects)
- `public/.htaccess` (comment out redirects)
- Google Analytics tracking ID

## Monitoring & Metrics

### Week 1-2 Metrics
- 301 redirect response codes
- Page load times
- Error rates
- User complaint volume

### Week 3-4 Metrics  
- Organic search traffic recovery
- Direct traffic to new domain
- Social media traffic
- Email click-through rates

### Long-term Monitoring
- SEO rankings for key terms
- Backlink profile changes
- Domain authority metrics
- User retention/churn

## Communication Templates

### User Email Notification
```
Subject: We're now LLM Works! New domain: llmworks.dev

Hi [Name],

We're excited to announce our rebrand to LLM Works with a new home at llmworks.dev!

What's changing:
✓ New website: llmworks.dev (your bookmarks will redirect automatically)
✓ Same powerful evaluation platform you know and trust
✓ Enhanced features and improved performance

What stays the same:  
✓ Your account and evaluation history
✓ All existing functionality
✓ Our commitment to open-source AI evaluation

Visit your dashboard: https://llmworks.dev/dashboard

Questions? Just reply to this email.

The LLM Works Team
```

### Social Media Announcement
```
🚀 Big news! We're now LLM Works!

New domain: llmworks.dev
Same powerful LLM evaluation platform
Enhanced features & performance

All your bookmarks will redirect automatically. 

#Rebrand #LLMWorks #AIEvaluation
```

---

*Created: January 12, 2025*  
*Status: Ready for implementation*  
*Estimated Timeline: 4 weeks*