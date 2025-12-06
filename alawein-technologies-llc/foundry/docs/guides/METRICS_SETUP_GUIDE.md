# Complete Metrics Setup Guide

**Step-by-step guide to implementing metrics tracking for your startup.**

From zero to complete analytics infrastructure in one week.

---

## 🎯 What to Measure by Stage

### Validation Stage (Weeks 1-4)

**Essential metrics:**
- [ ] # of interviews completed
- [ ] % identifying problem
- [ ] # of pre-sales
- [ ] Average pre-sale price
- [ ] Common objections

**How to track:**
- Google Sheet with interview data
- Spreadsheet formulas for calculations
- Manual updates daily

**Tools needed:** Google Sheets (free)

**Time setup:** 30 minutes

---

### MVP Build Stage (Weeks 5-20)

**Essential metrics:**
- [ ] Build progress (% complete)
- [ ] Code coverage (target: 80%+)
- [ ] Bug count (should decrease)
- [ ] Pre-sale customer feedback
- [ ] Feature completion rate

**Growth metrics:**
- [ ] Waitlist signups
- [ ] Email list growth
- [ ] Twitter followers
- [ ] Newsletter opens (if applicable)

**How to track:**
- GitHub metrics (built-in)
- Analytics (if website up)
- Spreadsheet for feedback

**Tools needed:**
- GitHub (free)
- Google Analytics (free)
- Google Sheets (free)

**Time setup:** 1-2 hours

---

### Launch Stage (Weeks 21-28)

**Crucial metrics:**
- [ ] Website visitors
- [ ] Signup conversion
- [ ] Trial signups
- [ ] Paid conversions
- [ ] Revenue (daily, weekly, monthly)
- [ ] Customer retention
- [ ] Support tickets
- [ ] Feature usage

**How to track:**
- Google Analytics for website
- Custom dashboard for app metrics
- CRM for customer tracking
- Spreadsheet for revenue
- Email for support tickets

**Tools needed:**
- Google Analytics (free)
- Amplitude or Mixpanel (free tier)
- Stripe (free)
- Spreadsheet (free)

**Time setup:** 2-4 hours

---

### Scale Stage (Weeks 29-52+)

**Business metrics:**
- [ ] Monthly Recurring Revenue (MRR)
- [ ] Annual Recurring Revenue (ARR)
- [ ] Customer Acquisition Cost (CAC)
- [ ] Lifetime Value (LTV)
- [ ] Churn rate
- [ ] Net Retention
- [ ] Growth rate
- [ ] Burn rate (if fundraised)

**Product metrics:**
- [ ] Daily Active Users (DAU)
- [ ] Monthly Active Users (MAU)
- [ ] Feature adoption
- [ ] User engagement
- [ ] User segments/cohorts
- [ ] Retention by cohort

**Team metrics:**
- [ ] Team size
- [ ] Hiring progress
- [ ] Turnover
- [ ] Productivity

**How to track:**
- Dedicated BI tool (Metabase, Tableau)
- Custom dashboards
- Weekly email reports
- Monthly board deck

**Tools needed:**
- Metabase (open source, free)
- or Tableau (paid)
- or Amplitude (paid)
- or Mixpanel (paid)
- Spreadsheet (always useful)

**Time setup:** 4-8 hours

---

## 🛠️ Tool-by-Tool Setup

### Google Analytics (FREE - Website Metrics)

**Setup time:** 30 minutes

**What it tracks:**
- Website visitors
- Pageviews
- Bounce rate
- Conversion funnel
- Traffic sources
- User behavior

**How to set it up:**

1. Go to analytics.google.com
2. Sign in with Google account
3. Click "Start measuring"
4. Enter website URL
5. Copy tracking code (provided)
6. Paste in website `<head>` tag
7. Wait 24 hours for data

**Key reports:**
- Audience Overview (who visits?)
- Acquisition (where do they come from?)
- Behavior (what do they do?)
- Conversions (what's your goal?)

**Pro tip:** Set up conversion goals (signup, demo request, etc.)

---

### Stripe (FREE - Payment Tracking)

**Setup time:** 1 hour

**What it tracks:**
- All payments
- Revenue
- Recurring charges
- Chargebacks
- Failed payments
- Customer data

**How to set it up:**

1. Go to stripe.com
2. Sign up for account
3. Add business info
4. Get API keys
5. Integrate into website/app
6. (Or use Stripe-hosted checkout)

**Key reports:**
- Revenue by date
- Customer list
- Subscription status
- Failed payments
- Refunds

**Pro tip:** Use Stripe Dashboard directly initially, then integrate deeply

---

### Google Sheets (FREE - Custom Tracking)

**Setup time:** 1-2 hours

**What it tracks:**
- Anything you want!
- User feedback
- Customer interviews
- Key decisions
- Custom metrics

**How to set it up:**

1. Create new spreadsheet
2. Add column headers
3. Create formulas
4. Setup daily/weekly updates
5. Add charts/graphs
6. Share with team

**Example tabs:**
- Daily Metrics (DAU, revenue, signups)
- Weekly Review (7-day summary)
- Monthly Dashboard (month overview)
- Customer Feedback (what users say)
- Roadmap Tracking (feature progress)

**Pro tip:** Automate updates with scripts (IFTTT, Zapier)

---

### Amplitude (FREE TIER - Product Analytics)

**Setup time:** 2 hours

**What it tracks:**
- User actions (buttons clicked, pages viewed)
- User journeys
- Feature adoption
- Retention cohorts
- Churn analysis
- User segmentation

**How to set it up:**

1. Go to amplitude.com
2. Sign up for free account
3. Create project
4. Install SDK in your app
5. Track custom events
6. Build dashboards

**Key reports:**
- User insights (who are your users?)
- Retention (do they come back?)
- Funnel (where do users drop off?)
- Cohort (grouped user behavior)

**Pro tip:** Start tracking events from day 1 of MVP

---

### Mixpanel (FREE TIER - Alternative to Amplitude)

**Setup time:** 2 hours

**Similar to Amplitude:**
- Product analytics
- User tracking
- Funnel analysis
- Retention tracking
- Cohort analysis

**Why choose Mixpanel over Amplitude:**
- Better for mobile apps
- More intuitive reporting
- Better free tier

**How to set up:**
Same process as Amplitude (sign up → install → track)

---

### Metabase (FREE - BI Dashboard)

**Setup time:** 3-4 hours

**What it does:**
- Connect to your database
- Create custom queries
- Build dashboards
- Share reports

**How to set it up:**

1. Download Metabase (opensource)
2. Install locally or on server
3. Connect to your database
4. Create dashboards
5. Add users

**What you can create:**
- Revenue dashboard (daily MRR)
- User growth dashboard (DAU/MAU trends)
- Cohort retention dashboard
- Custom metrics dashboard

**Pro tip:** Do this once you have real database (after MVP launch)

---

## 📊 Sample Dashboards

### Validation Stage Dashboard

```
┌─────────────────────────────────────┐
│   VALIDATION SPRINT PROGRESS         │
├─────────────────────────────────────┤
│                                       │
│  Interviews Completed: 15/20         │
│  % Identifying Problem: 70%           │
│  Pre-Sales: 8/10                      │
│                                       │
│  Common Objections:                   │
│  1. Too expensive (35%)              │
│  2. Need more features (25%)         │
│  3. Not sure about fit (20%)         │
│                                       │
│  Next Step: 5 more interviews        │
└─────────────────────────────────────┘
```

**How to create:**
- Google Sheet with these cells
- Manual daily updates
- Share with team

---

### Launch Stage Dashboard

```
┌──────────────────────────────────────┐
│        LAUNCH DAY METRICS              │
├──────────────────────────────────────┤
│                                        │
│  Website Visitors: 850                 │
│  Signup Rate: 12% (102 signups)       │
│  Trial Starts: 45                      │
│  Paid Conversions: 8 ($792 MRR)       │
│                                        │
│  Product Hunt:                         │
│  Upvotes: 347                          │
│  Rank: #8                              │
│                                        │
│  Support Tickets: 12                   │
│  Response Time: <2 hours              │
│                                        │
└──────────────────────────────────────┘
```

**How to create:**
- Google Sheets with formulas
- Connect to Google Analytics, Stripe, etc.
- Update hourly during launch day

---

### Scale Stage Dashboard

```
┌───────────────────────────────────────┐
│     MONTHLY BUSINESS DASHBOARD         │
├───────────────────────────────────────┤
│                                         │
│  MRR: $12,450 (↑ 8% vs last month)    │
│  ARR: $149,400                         │
│  Customers: 120 (↑ 15 this month)     │
│                                         │
│  Metrics:                               │
│  CAC: $850                              │
│  LTV: $2,850                            │
│  LTV:CAC Ratio: 3.35:1 ✅              │
│  Monthly Churn: 3.2%                   │
│  Net Retention: 104%                   │
│                                         │
│  Growth Rate: 12% MoM                  │
│  Burn Rate: $5,200/month (if raised)  │
│  Runway: 8.2 months (if raised)       │
│                                         │
│  Team: 4 people                        │
│  Cost per Person: $3,100              │
│                                         │
└───────────────────────────────────────┘
```

**How to create:**
- Metabase + custom database queries
- Automated weekly/monthly generation
- Email distribution to team

---

## 📈 Key Formulas

### CAC (Customer Acquisition Cost)

```
CAC = Total Marketing Spend / New Customers
CAC = $1,000 / 10 customers = $100 CAC
```

**Target:** CAC < $50 for B2C

---

### LTV (Customer Lifetime Value)

```
LTV = (Average Monthly Revenue per Customer) × (Average Customer Lifetime in Months)
LTV = $100/month × 12 months = $1,200 LTV
```

**Better formula:**
```
LTV = (ARPU × Gross Margin) / Monthly Churn Rate
LTV = ($100 × 0.8) / 0.05 = $1,600 LTV
```

**Target:** LTV > CAC × 3

---

### MRR (Monthly Recurring Revenue)

```
MRR = (Number of Customers) × (Average Monthly Price)
MRR = 50 customers × $99/month = $4,950 MRR
```

**Track weekly:**
- This week vs. last week
- Growth rate

---

### Retention Rate

```
Retention = (Users Still Here / Users at Start) × 100
Retention = (95 / 100) × 100 = 95% monthly retention
```

**Target:** >80% monthly retention

---

## 🚀 One-Week Metrics Setup Plan

**Goal:** Have complete metrics tracking infrastructure ready

### Day 1: Foundation

- [ ] Setup Google Analytics (30 min)
- [ ] Setup Stripe (30 min)
- [ ] Create Google Sheets dashboard (30 min)
- [ ] Share with team
- **Total: 1.5 hours**

### Day 2: Product Metrics

- [ ] Choose Amplitude OR Mixpanel (free)
- [ ] Integrate into your app/website
- [ ] Setup 5 key events to track
- [ ] Build basic retention dashboard
- **Total: 2-3 hours**

### Day 3: Custom Tracking

- [ ] Create detailed Google Sheet dashboards
- [ ] Add formulas for key metrics
- [ ] Setup daily auto-update (if possible)
- [ ] Create weekly review template
- **Total: 2-3 hours**

### Day 4: Reporting

- [ ] Design weekly metrics report
- [ ] Schedule email send (Friday 5pm)
- [ ] Create monthly board deck template
- [ ] Setup alert for critical metrics
- **Total: 1-2 hours**

### Day 5: Testing

- [ ] Review all dashboards
- [ ] Test data flow (add test customer)
- [ ] Verify calculations
- [ ] Train team on reading metrics
- **Total: 1-2 hours**

### By End of Week

✅ Google Analytics tracking website
✅ Stripe tracking revenue
✅ Amplitude tracking user behavior
✅ Google Sheets dashboards created
✅ Weekly reporting system set up
✅ Team trained on metrics

---

## 📊 Metrics Review Schedule

### Daily (5 min)

Check in morning:
- Revenue (new customers?)
- Website visitors
- Any alerts/critical issues

### Weekly (1 hour Friday afternoon)

Full review:
- Revenue & growth
- User retention
- Feature adoption
- Support tickets
- Team productivity

Create report & email team

### Monthly (2 hours)

Strategic review:
- Progress vs. targets
- Cohort analysis
- Improvement areas
- Next month strategy
- Board deck (if applicable)

---

## 🎯 Pro Tips

1. **Start simple** - Don't over-engineer metrics
2. **Automate early** - Manual updates get skipped
3. **Daily check-in** - 5 min morning review
4. **Weekly meeting** - Discuss metrics together
5. **Share results** - Transparent = motivating
6. **Set targets** - What are you aiming for?
7. **React to data** - Metrics should drive decisions
8. **Celebrate wins** - Share good metrics!

---

**Ready to track everything?** Setup starts with Google Analytics today! 📊

*See METRICS_DASHBOARD.md for interpretation guide.*
