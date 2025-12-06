# 📊 ORCHEX Metrics Dashboard
## Measure What Matters. Track Progress. Make Data-Driven Decisions.

**Your complete guide to tracking startup health.**

---

## 🎯 The 5 Core Metrics (Track Weekly)

These 5 numbers tell you everything about your business health:

### 1. MRR (Monthly Recurring Revenue)
**Formula:** Sum of all active subscriptions × price
```
Example:
10 users @ $9/mo  = $90
5 users @ $29/mo  = $145
2 teams @ $99/mo  = $198
-------------------------
MRR = $433
```

**Target Growth:** 20% month-over-month
**Health Check:**
- 🟢 Growing >15% MoM = Excellent
- 🟡 Growing 5-15% MoM = Good
- 🔴 Growing <5% MoM = Needs attention

---

### 2. Active Users
**Formula:** Users who logged in in past 30 days
```
Total users: 500
Active (30 days): 320
Inactive: 180

Active User Rate = 320/500 = 64%
```

**Target:** >40% active rate
**Health Check:**
- 🟢 >60% = Excellent engagement
- 🟡 40-60% = Decent engagement
- 🔴 <40% = Low engagement (product issue)

---

### 3. Conversion Rate (Free → Paid)
**Formula:** Paying customers / Total signups × 100%
```
Total signups: 500
Paying customers: 35

Conversion Rate = 35/500 = 7%
```

**Target:** 7% (industry average: 2-5%)
**Health Check:**
- 🟢 >10% = Excellent product-market fit
- 🟡 5-10% = Good
- 🔴 <5% = Pricing or value prop issue

---

### 4. Churn Rate (Monthly)
**Formula:** Customers who canceled / Total customers × 100%
```
Customers start of month: 35
Customers canceled: 2

Churn Rate = 2/35 = 5.7%
```

**Target:** <5% monthly churn
**Health Check:**
- 🟢 <3% = Excellent retention
- 🟡 3-7% = Acceptable
- 🔴 >7% = Product/support issue

---

### 5. CAC (Customer Acquisition Cost)
**Formula:** Total marketing/sales spend / New customers
```
Monthly ad spend: $500
Sales time value: $200
Total: $700

New customers: 35

CAC = $700/35 = $20
```

**Target:** CAC < LTV/3 (LTV = $345, so CAC should be <$115)
**Health Check:**
- 🟢 LTV:CAC >3:1 = Great unit economics
- 🟡 LTV:CAC 1:1 to 3:1 = Breakeven to profitable
- 🔴 LTV:CAC <1:1 = Losing money on each customer

---

## 📈 The Complete Metrics Tracker (Copy This Spreadsheet)

### Weekly Snapshot
Track these every Monday morning:

| Week | MRR | New MRR | Churn | Net MRR Growth | Total Users | New Users | Active Users | Paying Users | Conv. Rate | Churn Rate |
|------|-----|---------|-------|----------------|-------------|-----------|--------------|--------------|------------|-----------|
| 1 | $130 | $130 | $0 | $130 | 50 | 50 | 30 | 2 | 4% | 0% |
| 2 | $270 | $140 | $0 | $140 | 150 | 100 | 90 | 5 | 3.3% | 0% |
| 3 | $450 | $200 | $20 | $180 | 350 | 200 | 210 | 10 | 2.9% | 4% |
| 4 | $680 | $260 | $30 | $230 | 600 | 250 | 360 | 18 | 3% | 4.5% |

**How to use:**
1. Copy this to Google Sheets
2. Add a row every Monday
3. Create charts (MRR over time, users over time)
4. Review trends: What's going up? What's going down?

---

## 💰 Revenue Tracking Template

### Monthly Revenue Breakdown

```
Month: November 2025

REVENUE
├── Subscriptions
│   ├── Basic ($9/mo):  15 users × $9  = $135
│   ├── Pro ($29/mo):   8 users × $29  = $232
│   └── Teams ($99/mo): 2 teams × $99  = $198
│   TOTAL SUBSCRIPTIONS: $565
│
├── One-Time Sales
│   └── Annual plans: 1 × $290 = $290
│
└── Other
    └── Affiliate commission: $45

TOTAL REVENUE: $900

COSTS
├── Infrastructure
│   ├── Hosting (Vercel + Railway): $50
│   ├── Database (Supabase): $25
│   ├── OpenAI API: $150
│   └── Anthropic API: $75
│   SUBTOTAL: $300
│
├── Tools/SaaS
│   ├── Stripe: $27 (3% of revenue)
│   ├── Email (SendGrid): $15
│   └── Analytics: $10
│   SUBTOTAL: $52
│
└── Marketing
    ├── Ads: $100
    └── Content writer: $200
    SUBTOTAL: $300

TOTAL COSTS: $652

NET PROFIT: $248
PROFIT MARGIN: 27.5%
```

**Health Check:**
- 🟢 Profit margin >30% = Very healthy
- 🟡 Profit margin 10-30% = Sustainable
- 🔴 Profit margin <10% = Need to optimize costs or raise prices

---

## 👥 User Funnel Analysis

### The Complete Funnel

```
STAGE 1: AWARENESS
├── Website Visitors: 10,000
└── Conversion to signup: 5% = 500 signups

STAGE 2: ACTIVATION
├── Signups: 500
├── Completed onboarding: 70% = 350
└── Used product 1x: 60% = 300

STAGE 3: ENGAGEMENT
├── Used product 1x: 300
├── Used product 3x: 40% = 120
└── Active (30 days): 35% = 105

STAGE 4: CONVERSION
├── Active users: 105
├── Saw paywall: 80% = 84
├── Started checkout: 50% = 42
└── Completed purchase: 80% = 34

STAGE 5: RETENTION
├── Paid users: 34
├── Still active (Month 2): 90% = 31
└── Renewed subscription: 95% = 29

OVERALL CONVERSION: 0.3% (34/10,000)
```

**Where to optimize:**
1. Find biggest drop-off
2. Fix that stage first
3. Move to next biggest drop-off

**Example:** If 500 sign up but only 350 complete onboarding:
→ Fix onboarding experience (simplify steps, add progress bar, etc.)

---

## 📊 North Star Metric

**Your ONE metric that matters most:**

### For Phase 1 (Weeks 1-16):
**North Star:** Weekly Active Users (WAU)

**Why:** More active users → More conversions → More revenue

**How to increase:**
1. Improve onboarding (activate more signups)
2. Add engagement loops (bring users back)
3. Build features users want (increase utility)

### For Phase 2+ (Month 5+):
**North Star:** Net Revenue Retention (NRR)

**Formula:** (MRR start + expansion - churn) / MRR start × 100%

**Why:** Shows if existing customers are spending more over time

**Target:** >100% (means revenue grows even without new customers)

---

## 🎯 Goal Setting Framework (OKRs)

### Quarter 1 (Months 1-3): LAUNCH

**Objective:** Successfully launch and achieve product-market fit

**Key Results:**
- KR1: Reach $15K MRR
- KR2: Achieve 7% free-to-paid conversion rate
- KR3: Maintain <5% monthly churn
- KR4: Get 50+ testimonials/reviews

**Tracking:**
- Weekly: Check MRR, conversion, churn
- Monthly: Survey users for NPS
- End of quarter: Retrospective

---

### Quarter 2 (Months 4-6): OPTIMIZE

**Objective:** Improve unit economics and double down on growth

**Key Results:**
- KR1: Increase MRR to $50K
- KR2: Reduce CAC from $20 to $15
- KR3: Increase LTV from $345 to $450
- KR4: Launch 1 new product (Phase 2)

---

### Quarter 3 (Months 7-9): SCALE

**Objective:** Scale to $100K MRR profitably

**Key Results:**
- KR1: Reach $100K MRR
- KR2: Maintain >25% profit margin
- KR3: Launch 3rd product
- KR4: Hire 2 team members

---

### Quarter 4 (Months 10-12): EXPAND

**Objective:** Diversify revenue and build moat

**Key Results:**
- KR1: Reach $200K MRR
- KR2: Get 10 enterprise customers (>$500/mo each)
- KR3: Launch API/integrations
- KR4: Raise Series A or reach profitability

---

## 📈 Advanced Metrics (Track Monthly)

### 1. Lifetime Value (LTV)
**Formula:** ARPU × Average Lifetime (months) × Gross Margin
```
ARPU (Average Revenue Per User): $18/mo
Average Lifetime: 24 months
Gross Margin: 80%

LTV = $18 × 24 × 0.80 = $345
```

**Target:** LTV > CAC × 3

---

### 2. Payback Period
**Formula:** CAC / (ARPU × Gross Margin)
```
CAC: $20
ARPU: $18
Gross Margin: 80%

Payback Period = $20 / ($18 × 0.80) = 1.4 months
```

**Target:** <12 months
**Health Check:**
- 🟢 <6 months = Excellent (can scale fast)
- 🟡 6-12 months = Good
- 🔴 >12 months = Too long (cash flow issues)

---

### 3. Net Promoter Score (NPS)
**Formula:** % Promoters - % Detractors
```
Survey users: "How likely are you to recommend us? (0-10)"

Promoters (9-10): 60%
Passives (7-8): 30%
Detractors (0-6): 10%

NPS = 60% - 10% = 50
```

**Target:** >40
**Health Check:**
- 🟢 >50 = Excellent (users love you)
- 🟡 20-50 = Good
- 🔴 <20 = Product issues

**How to measure:** Email survey monthly to 100 random users

---

### 4. Viral Coefficient (K-Factor)
**Formula:** (Invites sent / User) × (Conversion rate of invites)
```
Average user sends: 3 invites
Conversion rate: 20%

K = 3 × 0.20 = 0.6
```

**Target:** >1 (viral growth)
**Reality:** Most products: 0.3-0.7 (need other growth channels)

---

### 5. Feature Adoption Rate
**Formula:** Users who used feature / Total users × 100%
```
Total users: 500
Used "Nightmare Mode": 150

Adoption Rate = 150/500 = 30%
```

**Use case:** Identify which features drive retention
**Action:** Double down on highly-adopted features

---

## 🚨 Red Flags (Act Immediately If You See These)

### 🔴 MRR Declining 2 Weeks in a Row
**Cause:** Churn > New customers
**Action:**
1. Talk to churned customers (why did they leave?)
2. Improve onboarding (activate more users)
3. Fix critical product issues immediately

---

### 🔴 Conversion Rate Dropping Below 3%
**Cause:** Pricing too high OR value prop unclear
**Action:**
1. A/B test pricing (lower by 20%)
2. Improve product demo/onboarding
3. Add free trial (remove friction)

---

### 🔴 Active User Rate <30%
**Cause:** Users sign up but don't find value
**Action:**
1. Interview 10 inactive users
2. Improve onboarding tutorial
3. Add email re-engagement campaign

---

### 🔴 CAC > LTV
**Cause:** Spending more to acquire than customer is worth
**Action:**
1. STOP paid ads immediately
2. Focus on organic channels
3. Increase LTV (raise prices, reduce churn)

---

### 🔴 Churn Rate >10%/month
**Cause:** Product not solving problem OR support issues
**Action:**
1. Call every churning customer (exit interviews)
2. Identify top 3 reasons
3. Fix those 3 things in next sprint

---

## 📅 Weekly Review Ritual (Monday Morning)

### Step 1: Update Numbers (15 min)
- [ ] Update MRR in spreadsheet
- [ ] Count total users, active users
- [ ] Calculate conversion rate
- [ ] Check churn (who canceled?)
- [ ] Calculate CAC (spend / new customers)

### Step 2: Analyze Trends (15 min)
- [ ] What went up? (celebrate 🎉)
- [ ] What went down? (investigate 🔍)
- [ ] Are we on track for quarterly goals?
- [ ] Any red flags?

### Step 3: Set Weekly Goals (10 min)
- [ ] What's the ONE metric to improve this week?
- [ ] What 3 actions will move that metric?
- [ ] Who's responsible?
- [ ] How will we measure success?

### Step 4: Share Publicly (5 min)
- [ ] Tweet: "Week X update: [Key metric] → [New number]"
- [ ] Internal team update (if you have a team)
- [ ] Accountability partner check-in

**Total time:** 45 minutes/week
**ROI:** Priceless (stay on track, catch issues early)

---

## 📊 Dashboard Tools (Choose One)

### Free Options:
- **Google Sheets** (manual but flexible)
- **Notion** (nice UI, templates available)
- **Airtable** (database + spreadsheet hybrid)

### Paid Tools:
- **Baremetrics** ($50/mo) - Auto-connects to Stripe
- **ChartMogul** ($100/mo) - Advanced analytics
- **Mixpanel** ($25/mo) - User behavior tracking
- **Amplitude** (Free tier) - Product analytics

**Recommendation for Phase 1:** Start with Google Sheets (free, learn the metrics). Upgrade to paid tool when >$10K MRR.

---

## 🎯 Metrics Cheat Sheet (Print This)

```
┌─────────────────────────────────────────────────────────────┐
│              YOUR STARTUP HEALTH AT A GLANCE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MRR:                    $______    (↑ target: 20%/mo)     │
│  Active Users:           ______     (↑ target: 40%+ rate)  │
│  Conversion Rate:        ______%    (↑ target: 7%)         │
│  Monthly Churn:          ______%    (↓ target: <5%)        │
│  CAC:                    $______    (↓ target: <$60)       │
│                                                             │
│  LTV:CAC Ratio:          ___:1      (↑ target: >3:1)       │
│  NPS:                    ______     (↑ target: >40)        │
│  Profit Margin:          ______%    (↑ target: >25%)       │
│                                                             │
│  Status: [  Healthy  ] [ Needs Attention ] [ Critical ]    │
│                                                             │
│  Next Week's Focus: _________________________________       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏆 Success Milestones to Celebrate

### 🎉 First Customer
**Metric:** 1 paying customer
**Celebrate:** Screenshot Stripe notification, tweet it, frame it

### 🎉 $1K MRR
**Metric:** $1,000 MRR
**Celebrate:** You've validated people will pay. Dinner out!

### 🎉 $10K MRR
**Metric:** $10,000 MRR
**Celebrate:** Sustainable income. Quit day job if you haven't!

### 🎉 $100K MRR
**Metric:** $100,000 MRR ($1.2M ARR)
**Celebrate:** Real business. Team offsite!

### 🎉 100 Customers
**Metric:** 100 paying customers
**Celebrate:** Send thank you notes to all 100

### 🎉 Break-Even
**Metric:** Profit margin >0% for 2 months straight
**Celebrate:** No more personal savings needed!

### 🎉 Profitability
**Metric:** $10K+ profit/month for 3 months straight
**Celebrate:** Give yourself a raise. Hire help.

---

## 💡 The One Metric That Rules Them All

**Revenue solves everything.**

- Product bugs? Hire engineers.
- Support overwhelmed? Hire support.
- Need marketing? Hire marketers.
- Want to quit day job? Revenue.

**Focus on MRR growth above all else in Phase 1.**

Everything else is a supporting metric.

---

## 📝 Weekly Metrics Update Template (For Social Sharing)

```
Week X Update 📊

MRR: $X,XXX (↑ X% from last week)
Users: XXX (↑ XX new this week)
Active Rate: XX%
Conversion: X.X%

What worked:
• [Thing 1]
• [Thing 2]

What didn't:
• [Challenge]

Focus this week:
• [Goal]

[Chart/graph screenshot]

#BuildInPublic #SaaS #Startup
```

**Post this every Monday** (transparency builds trust + accountability)

---

## 🎯 Your Action Items (This Week)

### Monday (Today):
- [ ] Create Google Sheet with metrics tracker
- [ ] Fill in current numbers
- [ ] Identify your weakest metric
- [ ] Set goal for next week

### Tuesday-Friday:
- [ ] Work on improving weakest metric
- [ ] Track progress daily

### Next Monday:
- [ ] Update metrics
- [ ] Did the weak metric improve?
- [ ] If yes: Celebrate + tackle next metric
- [ ] If no: Try different approach

---

## 🔥 Remember

**"What gets measured gets improved."**

Track your metrics religiously.
Make data-driven decisions.
Celebrate wins.
Fix issues fast.

**Now go track some metrics!** 📊🚀
