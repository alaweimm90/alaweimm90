# 📊 REPZ Post-Launch Monitoring Guide

**Use this guide for the first 48 hours after launch**

---

## 🎯 Hour 1: Immediate Monitoring

### Check Every 15 Minutes

**1. Sentry Dashboard**
- [ ] Go to: https://sentry.io
- [ ] Check for new errors
- [ ] **Red flags:**
  - Any error with >5 occurrences
  - Payment processing errors
  - Database connection errors
  - Auth errors

**2. Vercel Analytics**
- [ ] Go to: Vercel Dashboard → Analytics
- [ ] Check for:
  - Page load times (<3 seconds is good)
  - Error rates (<1% is good)
  - Traffic spikes

**3. Stripe Dashboard**
- [ ] Go to: https://dashboard.stripe.com/payments
- [ ] Check for:
  - Successful payments
  - Failed payments (investigate why)
  - Webhook delivery status

**4. SendGrid Activity**
- [ ] Go to: https://app.sendgrid.com/email_activity
- [ ] Check for:
  - Emails delivered
  - Bounces (should be 0)
  - Spam reports (should be 0)

---

## 📈 Hour 2-24: Active Monitoring

### Check Every Hour

**Health Check**
```bash
# Run this command
curl -I https://yourdomain.com

# Should return: HTTP/2 200
# If not, investigate immediately
```

**Key Metrics to Track:**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Uptime | 100% | 99%+ | <99% |
| Response Time | <2s | 2-5s | >5s |
| Error Rate | 0% | <1% | >1% |
| Email Deliverability | 100% | 95%+ | <95% |
| Payment Success Rate | 100% | 98%+ | <98% |

---

## 🔍 What to Monitor

### 1. Application Health

**Endpoint to check:**
- https://yourdomain.com/ (should load)
- https://yourdomain.com/intake (intake form)
- https://yourdomain.com/login (auth working)
- https://yourdomain.com/admin (admin access)

### 2. Database Health

**Check in Supabase:**
- [ ] Go to Database → Table Editor
- [ ] Verify `non_portal_clients` table accessible
- [ ] Check recent entries are appearing
- [ ] Monitor connection count (should be <50)

### 3. Payment Flow

**Test every 6 hours (first day):**
```
1. Go to /intake
2. Fill form
3. Select payment tier
4. Use TEST card: 4242 4242 4242 4242
5. Verify:
   - Payment succeeds
   - Confirmation email received
   - Client appears in admin dashboard
   - Stripe webhook triggers
```

### 4. Error Patterns

**Check Sentry for:**
- TypeScript errors
- Network errors (API failures)
- Payment errors
- Auth errors
- Database errors

**Common issues to watch:**
- CORS errors (means env vars not set)
- 500 errors (server-side issues)
- 401 errors (auth problems)
- Database timeout errors

---

## 🚨 Alert Triggers

**IMMEDIATE action needed if:**

1. **Uptime drops below 99%**
   - Check Vercel deployment status
   - Check Supabase status page

2. **Payment success rate < 95%**
   - Check Stripe webhook configuration
   - Verify API keys are correct
   - Check Stripe logs

3. **Email bounce rate > 5%**
   - Check SendGrid domain verification
   - Check email templates
   - Verify FROM address

4. **Error rate > 5%**
   - Check Sentry for error details
   - Roll back if necessary
   - Fix and redeploy

---

## 📱 Monitoring Tools

### Set Up Mobile Alerts

**Sentry:**
- Install Sentry mobile app
- Enable push notifications
- Set alert threshold: >5 errors in 5 minutes

**Vercel:**
- Install Vercel app
- Enable deployment notifications
- Enable error notifications

**Stripe:**
- Enable email notifications for:
  - Failed payments
  - Successful payments (first week)
  - Disputes

---

## 📊 Daily Checks (Days 2-7)

### Morning Routine (10 minutes)

```bash
# 1. Check overnight metrics
- Sentry: Any new errors?
- Stripe: Any payments?
- Vercel: Uptime status?
- SendGrid: Email delivery rate?

# 2. Review yesterday's stats
- Total visitors
- Conversions (form submissions)
- Payment success rate
- Average response time

# 3. Check social mentions
- Twitter/X
- LinkedIn
- Instagram
- Reddit (if posted)
```

### Evening Routine (5 minutes)

```bash
# 1. Quick health check
curl -I https://yourdomain.com

# 2. Check Sentry
- Any unresolved errors?

# 3. Review day's performance
- How many visitors?
- How many signups?
- Any payments?
```

---

## 🔧 Common Issues & Fixes

### Issue: High error rate on /intake

**Diagnosis:**
```bash
# Check Sentry for error details
# Common causes:
- Missing env vars (VITE_STRIPE_PUBLIC_KEY)
- CORS issues
- Database connection errors
```

**Fix:**
1. Check Vercel env vars
2. Redeploy if vars were missing
3. Test intake form again

### Issue: Emails not sending

**Diagnosis:**
```bash
# Check SendGrid activity feed
# Common causes:
- Domain not verified
- API key incorrect
- FROM address not verified
```

**Fix:**
1. Re-verify domain in SendGrid
2. Check API key in Vercel env vars
3. Test with SendGrid email testing tool

### Issue: Payments failing

**Diagnosis:**
```bash
# Check Stripe dashboard → Logs
# Common causes:
- Webhook URL incorrect
- Webhook signing secret wrong
- API keys in test mode vs live mode
```

**Fix:**
1. Verify webhook URL in Stripe
2. Check webhook signing secret
3. Ensure using live API keys (pk_live_, sk_live_)

---

## 📈 Success Metrics

### Week 1 Goals

- [ ] **Uptime:** 99.9%+
- [ ] **First paying customer** 🎉
- [ ] **Zero critical errors**
- [ ] **Email deliverability:** 98%+
- [ ] **Payment success rate:** 100%
- [ ] **5+ beta signups**

### What "Good" Looks Like

**After 7 days:**
- 100+ unique visitors
- 20+ form starts
- 5+ form completions
- 1-3 paying customers
- 0 critical bugs
- 95%+ uptime

---

## 🎯 Next Steps

**Week 2:**
- Reduce monitoring to 2x daily
- Focus on customer acquisition
- Gather user feedback
- Iterate based on data

**Month 1:**
- Daily monitoring sufficient
- Set up automated alerts
- Focus on growth
- Plan feature updates

---

## 🆘 Emergency Contacts

**Critical issues (site down, payments broken):**
1. Check Vercel status: https://vercel.com/status
2. Check Supabase status: https://status.supabase.com
3. Check Stripe status: https://status.stripe.com

**Support:**
- Vercel: https://vercel.com/support (live chat)
- Supabase: https://supabase.com/support (community)
- Stripe: https://support.stripe.com (24/7)

---

## 📝 Monitoring Checklist Template

**Date:** __________

### Morning Check (___:___ AM)
- [ ] Sentry: _____ errors (target: 0)
- [ ] Uptime: _____ % (target: 100%)
- [ ] Visitors (24h): _____
- [ ] Payments (24h): _____
- [ ] Emails sent: _____ delivered: _____ %

**Issues found:**
- _________________________________
- _________________________________

**Actions taken:**
- _________________________________
- _________________________________

### Evening Check (___:___ PM)
- [ ] Sentry: _____ new errors
- [ ] Total visitors: _____
- [ ] Total conversions: _____
- [ ] Payment success rate: _____ %

**Notes:**
_________________________________
_________________________________

---

**🎉 You're live! Monitor closely for 48 hours, then relax and grow! 🚀**
