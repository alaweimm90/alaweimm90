# 📊 Metrics Tracking Spreadsheets - Copy & Use

**Spreadsheet templates for every phase. Track what matters, ignore vanity metrics.**

---

## 📑 TABLE OF CONTENTS

1. [Validation Metrics Tracker](#1-validation-metrics-tracker)
2. [Pre-Sale Conversion Tracker](#2-pre-sale-conversion-tracker)
3. [MVP Metrics Dashboard](#3-mvp-metrics-dashboard)
4. [SaaS Metrics Dashboard (Post-Launch)](#4-saas-metrics-dashboard)
5. [Cohort Analysis Tracker](#5-cohort-analysis-tracker)
6. [Unit Economics Calculator](#6-unit-economics-calculator)

---

## 1. VALIDATION METRICS TRACKER

**Use during:** Week 1-4 of validation sprint

**Purpose:** Track interview progress and validation signals

### Spreadsheet Structure:

#### TAB 1: Prospect Tracking
```
| Name | Email | Source | Date Sent | Opened? | Responded? | Scheduled? | Interview Date | Completed? | Pain Score /10 | Notes | Referrals Given |
|------|-------|--------|-----------|---------|------------|------------|----------------|------------|----------------|-------|-----------------|
| John Doe | john@uni.edu | Google Scholar | 1/15/24 | Yes | Yes | Yes | 1/18/24 9am | Yes | 8 | Great fit, works at MIT | 2 |
| Jane Smith | jane@corp.com | LinkedIn | 1/15/24 | Yes | No | - | - | - | - | Follow up 1/20 | 0 |
| Mike Johnson | mike@lab.org | Referral (John) | 1/18/24 | Yes | Yes | Yes | 1/22/24 2pm | No | - | Scheduled | 0 |
```

**Formulas:**
- **Response Rate:** `=COUNTIF(F:F,"Yes")/COUNTA(A:A)-1`
- **Interview Completion Rate:** `=COUNTIF(I:I,"Yes")/COUNTIF(G:G,"Yes")`
- **Average Pain Score:** `=AVERAGE(J:J)`

---

#### TAB 2: Interview Summary
```
| Interview # | Date | Name | Role | Pain Point 1 | Pain Point 2 | Pain Point 3 | Pain Severity /10 | Current Tools | Monthly Spend $ | Would Pay $/mo | Key Quote | Follow-Up Needed? |
|-------------|------|------|------|--------------|--------------|--------------|-------------------|---------------|-----------------|----------------|-----------|-------------------|
| 1 | 1/18 | John Doe | Prof, MIT | Statistical validation | Peer review anxiety | Time-consuming | 8 | SPSS ($50), Excel | $50 | $25-50 | "I spend 10 hours per paper just checking stats" | Yes - show mockup |
| 2 | 1/19 | Jane Smith | PhD Student | Can't afford tools | Methodological uncertainty | - | 6 | R (free), Google Scholar | $0 | $10-15 | "Free tools are clunky" | Maybe |
```

**Formulas:**
- **Consistent Pain (Pain 1):** `=COUNTIF(E:E,"Statistical validation")/COUNTA(A:A)-1` (repeat for each pain)
- **Average Stated Willingness to Pay:** `=AVERAGE(K:K)`

---

#### TAB 3: Validation Dashboard

```
KEY METRICS                              TARGET    ACTUAL   STATUS
================================================  ======    ======  =======
Emails Sent                              30        35       ✅
Response Rate                            20%       34%      ✅
Interviews Scheduled                     10        8        🟡
Interviews Completed                     10        7        🟡
Show-Up Rate                             70%       88%      ✅
Average Pain Severity                    7/10      7.4/10   ✅

CONSISTENCY METRICS
================================================
Top Pain Point Consistency               70%       71% (5/7)  ✅
Pain Point: Statistical validation

Second Pain Point Consistency            50%       57% (4/7)  ✅
Pain Point: Peer review anxiety

Third Pain Point Consistency             30%       29% (2/7)  🔴
Pain Point: Time-consuming

FINANCIAL SIGNALS
================================================
Currently Paying for Solutions           60%       71% (5/7)  ✅
Average Monthly Spend                    -         $35/person
Stated Willingness to Pay (avg)          -         $30/mo
Stated Willingness to Pay (range)        -         $10-50/mo

DECISION SIGNALS
================================================
Strong Pain (8+/10)                      50%       57% (4/7)  ✅
Would Use This (unprompted)              -         43% (3/7)  🟡
Referrals Received                       10        6          🟡

WEEK 4 DECISION                          STATUS: CAUTIOUS GO
```

**Decision Logic:**
- **STRONG GO** if: Interviews ≥10, Pain ≥7/10, Consistency ≥70%, Paying ≥60%
- **CAUTIOUS GO** if: Interviews 7-9, Pain 6-7/10, Consistency 50-69%
- **PIVOT** if: Pain <6/10 OR Consistency <50%
- **STOP** if: Can't schedule interviews OR No one paying for solutions

---

## 2. PRE-SALE CONVERSION TRACKER

**Use during:** Week 3 (pre-sale experiment)

**Purpose:** Track conversion from visitors → signups → purchases

### Spreadsheet Structure:

#### TAB 1: Traffic & Conversion Funnel
```
| Date | Traffic Source | Visitors | Email Signups | Conversion to Signup % | Pre-Orders | Conversion to Purchase % | Revenue $ | Notes |
|------|----------------|----------|---------------|------------------------|------------|-------------------------|-----------|-------|
| 1/22 | Email (interviewees) | 18 | 15 | 83% | 8 | 53% | $792 | Strong! |
| 1/23 | Reddit r/AskAcademia | 127 | 8 | 6% | 1 | 13% | $99 | Low conversion |
| 1/24 | Twitter | 43 | 4 | 9% | 0 | 0% | $0 | Need better targeting |
| 1/25 | Facebook Ads | 234 | 12 | 5% | 2 | 17% | $198 | $100 ad spend |
```

**Formulas:**
- **Total Visitors:** `=SUM(C:C)`
- **Total Signups:** `=SUM(D:D)`
- **Overall Signup Rate:** `=SUM(D:D)/SUM(C:C)`
- **Total Pre-Orders:** `=SUM(F:F)`
- **Overall Conversion Rate:** `=SUM(F:F)/SUM(D:D)`
- **Total Revenue:** `=SUM(H:H)`

---

#### TAB 2: Individual Pre-Sale Customers
```
| Order # | Date | Name | Email | Source | Price Paid $ | Payment Method | Status | Onboarded? | Notes |
|---------|------|------|-------|--------|--------------|----------------|--------|------------|-------|
| 1 | 1/22 | John Doe | john@uni.edu | Interviewee | $99 | Stripe | Paid | No | Send build updates |
| 2 | 1/22 | Jane Smith | jane@corp.com | Reddit | $99 | Stripe | Paid | No | Beta access Week 5 |
```

**Email these people weekly!**

---

#### TAB 3: Pre-Sale Dashboard
```
PRE-SALE METRICS                         TARGET    ACTUAL   STATUS
================================================  ======    ======  =======
Total Visitors                           500       422      🟡
Email Signups                            50        39       🟡
Signup Rate                              10%       9.2%     🟡
Pre-Orders                               10        11       ✅✅
Conversion Rate (signup→purchase)        20%       28%      ✅✅
Total Revenue                            $990      $1,089   ✅✅

BY SOURCE
================================================
Interviewees: 8/18 (44% conversion)               ✅ Best source
Reddit: 1/127 (0.8% conversion)                   🔴 Poor
Twitter: 0/43 (0% conversion)                     🔴 Poor
Facebook Ads: 2/234 (0.9% conversion)             🔴 Poor, $100 spent

DECISION: STRONG GO (11 pre-sales ≥ 10 target)
→ Start building immediately
→ Email all customers with timeline
→ Focus acquisition on warm leads (interviews/referrals)
```

---

## 3. MVP METRICS DASHBOARD

**Use during:** Weeks 5-12 (beta & early users)

**Purpose:** Track engagement, retention, and product-market fit signals

### Spreadsheet Structure:

#### TAB 1: Weekly Active Users (WAU)
```
| Week | New Users | Total Users | Active Users (used 1x) | Active % | Power Users (used 5x+) | Power User % | Churned Users | Churn Rate % |
|------|-----------|-------------|------------------------|----------|------------------------|--------------|---------------|--------------|
| 1 | 10 | 10 | 8 | 80% | 3 | 30% | 0 | 0% |
| 2 | 5 | 15 | 11 | 73% | 5 | 33% | 2 | 13% |
| 3 | 8 | 23 | 17 | 74% | 9 | 39% | 1 | 4% |
```

**Formulas:**
- **Active %:** `=D2/C2`
- **Power User %:** `=F2/C2`
- **Churn Rate:** `=H2/C1` (churned / total from previous week)

**Targets:**
- **Active %:** >50% (if lower, product isn't sticky)
- **Power User %:** >20% (these are your champions)
- **Churn Rate:** <10%/week (if higher, urgent problem)

---

#### TAB 2: Feature Usage
```
| Feature | Total Uses | Unique Users | % of Users Using | Avg Uses per User | Power Users (5x+ use) | Notes |
|---------|------------|--------------|------------------|-------------------|----------------------|-------|
| Statistical Critic | 45 | 18 | 78% | 2.5 | 6 | Core feature, high engagement |
| Export Report | 12 | 10 | 43% | 1.2 | 0 | Used once, then forgotten |
| Share Results | 3 | 3 | 13% | 1.0 | 0 | Low usage, consider cutting |
```

**Insights:**
- Features used by <30% of users → Improve discoverability OR cut
- Features with 5+ uses → Core value, double down
- Features used once then never → Onboarding problem

---

#### TAB 3: User Feedback Log
```
| Date | User | Feedback Type | Issue/Request | Priority | Status | Notes |
|------|------|---------------|---------------|----------|--------|-------|
| 1/20 | John | Bug | Export crashes on large files | High | Fixed 1/22 | Affected 40% of users |
| 1/21 | Jane | Feature Request | Add PDF export | Medium | Backlog | Requested by 3 users |
| 1/22 | Mike | Positive | "This saved me 10 hours!" | - | - | Use as testimonial |
```

**Track:**
- Bug frequency (if same bug reported 3x → top priority)
- Feature requests (if 3+ users request → add to roadmap)
- Positive feedback (use for testimonials)

---

#### TAB 4: MVP Dashboard
```
MVP HEALTH METRICS                       TARGET    ACTUAL   STATUS
================================================  ======    ======  =======
Total Users                              50        23       🟡
Weekly Active Users (WAU)                35        17       🟡
Week-over-Week Retention                 50%       74%      ✅✅
Power Users (5x+ use)                    10        9        ✅

ENGAGEMENT
================================================
Avg Sessions per User per Week           3         4.2      ✅
Avg Time per Session                     10 min    12 min   ✅
Core Feature Usage                       70%       78%      ✅

NPS (NET PROMOTER SCORE)
================================================
Promoters (9-10)                         -         8 (47%)  🟡
Passives (7-8)                           -         6 (35%)
Detractors (0-6)                         -         3 (18%)
NPS Score                                30        29       🟡

FEEDBACK
================================================
Critical Bugs                            0         1        🟡
Feature Requests (3+ users)              -         2        ↗
Positive Testimonials                    5         4        🟡

DECISION: ITERATE & GROW
→ Retention good (74% WoW)
→ Engagement strong (4.2 sessions/week)
→ NPS needs improvement (target 40+)
→ Fix critical bug, add top 2 requested features
→ Grow to 50 users, then reassess
```

---

## 4. SAAS METRICS DASHBOARD (Post-Launch)

**Use during:** Month 3+ (scaling phase)

**Purpose:** Track revenue, growth, and unit economics

### Spreadsheet Structure:

#### TAB 1: Monthly Revenue
```
| Month | New Customers | Churned Customers | Total Customers | MRR (Monthly Recurring Revenue) $ | MRR Growth % | ARR (Annual) $ | Notes |
|-------|---------------|-------------------|-----------------|-----------------------------------|--------------|----------------|-------|
| Jan | 15 | 0 | 15 | $375 | - | $4,500 | Launch month |
| Feb | 23 | 2 | 36 | $900 | 140% | $10,800 | Strong growth |
| Mar | 31 | 4 | 63 | $1,575 | 75% | $18,900 | Growth slowing but healthy |
| Apr | 28 | 6 | 85 | $2,125 | 35% | $25,500 | Churn increasing (monitor) |
```

**Formulas:**
- **Total Customers:** `=D2+B3-C3` (previous + new - churned)
- **MRR:** `=D3*25` (assuming $25/user/month avg)
- **MRR Growth %:** `=(E3-E2)/E2`
- **ARR:** `=E3*12`

**Targets:**
- **MRR Growth:** 15-20% month-over-month (early stage)
- **Churn:** <5% per month (if >10%, red flag)

---

#### TAB 2: Customer Acquisition
```
| Month | Marketing Spend $ | New Customers | CAC (Cost per Acquisition) $ | Payback Period (months) | Notes |
|-------|-------------------|---------------|------------------------------|-------------------------|-------|
| Jan | $0 | 15 | $0 | 0 | Pre-sale customers |
| Feb | $200 | 23 | $8.70 | 0.35 | Reddit ads |
| Mar | $500 | 31 | $16.13 | 0.65 | Facebook ads |
| Apr | $1,000 | 28 | $35.71 | 1.43 | Scaling spend |
```

**Formulas:**
- **CAC:** `=B2/C2` (spend / new customers)
- **Payback Period:** `=D2/25` (CAC / monthly revenue per customer)

**Targets:**
- **CAC:** <$50 for $25/mo product (2-month payback)
- **Payback Period:** <3 months (if longer, acquisition too expensive)

---

#### TAB 3: Unit Economics
```
UNIT ECONOMICS                           CALCULATION              ACTUAL   TARGET  STATUS
==========================================  =======================  =======  ======  ======
Average Revenue Per User (ARPU) $/mo     Total MRR / Customers    $25      $25     ✅
Customer Lifetime (months)               1 / Monthly Churn Rate   20       20      ✅
Lifetime Value (LTV) $                   ARPU × Lifetime × 0.7    $350     $300    ✅
Customer Acquisition Cost (CAC) $        Marketing / New Cust     $30      <$50    ✅
LTV:CAC Ratio                            LTV / CAC                11.7:1   >3:1    ✅✅

PROFITABILITY
==========================================
Gross Margin                             Revenue - COGS           70%      >70%    ✅
Monthly Burn Rate $                      Expenses - Revenue       $3,000   -       →
Runway (months)                          Cash / Burn Rate         8        >6      ✅
Break-Even MRR $                         Fixed Costs / 0.7        $7,000   -       →
```

**Decision Logic:**
- **LTV:CAC <3:1** → Acquisition too expensive, fix before scaling
- **LTV:CAC 3-5:1** → Healthy, can scale sustainably
- **LTV:CAC >10:1** → Excellent, scale aggressively
- **Gross Margin <70%** → Cost structure unsustainable
- **Runway <6 months** → Need to fundraise or reach profitability fast

---

#### TAB 4: Growth Dashboard
```
GROWTH METRICS (Current Month)           TARGET    ACTUAL   STATUS
================================================  ======    ======  =======
MRR                                      $5,000    $2,125   🟡
MRR Growth Rate                          15%       35%      ✅✅
Total Customers                          200       85       🟡
New Customers                            30        28       🟡
Churn Rate                               <5%       7%       🟡
Net Revenue Retention                    >100%     95%      🟡

EFFICIENCY METRICS
================================================
CAC                                      <$50      $36      ✅
LTV                                      >$300     $350     ✅
LTV:CAC                                  >3:1      9.7:1    ✅✅
Payback Period                           <3 mo     1.4 mo   ✅✅
Customer Concentration (top 5)           <20%      12%      ✅

LEADING INDICATORS
================================================
Website Visitors (monthly)               10,000    3,400    🟡
Trial Signups                            200       67       🟡
Trial→Paid Conversion                    20%       32%      ✅
Referral Rate                            15%       8%       🟡

DECISION: SCALE ACQUISITION
→ Unit economics excellent (9.7:1 LTV:CAC)
→ Churn slightly high but acceptable (7%)
→ Strong organic conversion (32% trial→paid)
→ Increase marketing spend from $1K to $2K next month
→ Focus on reducing churn (target <5%)
```

---

## 5. COHORT ANALYSIS TRACKER

**Use during:** Month 3+ (understanding retention)

**Purpose:** See how different cohorts retain over time

### Spreadsheet Structure:

```
COHORT RETENTION (% of users still active)

Cohort (Sign-Up Month) | Users | Month 0 | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
========================|=======|=========|=========|=========|=========|=========|=========|=========|
January 2024            | 15    | 100%    | 87%     | 80%     | 73%     | 67%     | 60%     | 53%     |
February 2024           | 23    | 100%    | 83%     | 70%     | 61%     | 52%     | 43%     | -       |
March 2024              | 31    | 100%    | 77%     | 61%     | 48%     | -       | -       | -       |
April 2024              | 28    | 100%    | 71%     | 54%     | -       | -       | -       | -       |
```

**Insights:**
- **Month 1 retention <50%** → Product doesn't solve problem (critical issue)
- **Month 3 retention <40%** → Weak value, users try then leave
- **Month 6 retention >40%** → Strong product-market fit
- **Improving cohorts** → Product getting better (good!)
- **Declining cohorts** → Product getting worse (bad!)

**Example Analysis:**
> "January cohort: 53% still active after 6 months = good
> April cohort: Only 54% active after 2 months = worse than January
> → Something broke in the product or onboarding. Investigate!"

---

## 6. UNIT ECONOMICS CALCULATOR

**Use when:** Making decisions about pricing, spending, fundraising

**Purpose:** Quick calculator to model different scenarios

### Spreadsheet Structure:

```
INPUTS (Edit These)                      BASE CASE    SCENARIO A   SCENARIO B
============================================ ===========  ===========  ===========
Monthly Price per User $                 $25          $50          $15
Monthly Churn Rate %                     5%           4%           7%
Customer Acquisition Cost (CAC) $        $30          $50          $20
Gross Margin %                           70%          70%          70%

CALCULATED OUTPUTS
============================================
Customer Lifetime (months)               20           25           14.3
Lifetime Value (LTV) $                   $350         $875         $150
LTV:CAC Ratio                            11.7:1       17.5:1       7.5:1
Payback Period (months)                  1.2          1.0          1.3

DECISION
==============================================
BASE CASE:  Excellent (11.7:1) - Scale now ✅
SCENARIO A: Incredible (17.5:1) - Double price? 🤔
SCENARIO B: Good (7.5:1) - Lower price risky 🟡
```

**Use this to:**
- Test different pricing ($10 vs $25 vs $50)
- Model impact of reducing churn (5% → 3%)
- Calculate how much you can spend on CAC
- Decide if you should raise funding

---

## 📊 HOW TO USE THESE SPREADSHEETS

### Step 1: Copy Template
1. Open Google Sheets or Excel
2. Create new spreadsheet
3. Copy relevant section above
4. Customize column headers for your business

### Step 2: Update Daily/Weekly
- **Validation:** Update after each interview
- **Pre-Sale:** Update daily during campaign
- **MVP:** Update weekly (every Friday)
- **SaaS:** Update monthly (first of month)

### Step 3: Review in Standups
**Daily (during validation):**
- How many emails sent today?
- How many interviews scheduled?

**Weekly (during MVP):**
- What's our retention this week?
- What's the #1 user complaint?

**Monthly (during scaling):**
- Did we hit MRR growth target?
- Is CAC increasing or decreasing?
- What's our runway?

### Step 4: Make Data-Driven Decisions
**Use red flags to trigger action:**
- Churn >10% → Stop scaling, fix retention
- LTV:CAC <3:1 → Stop ads, improve conversion
- NPS <20 → Product doesn't solve problem
- Month 1 retention <50% → Critical product issue

---

## 🎯 METRIC PRIORITIES BY PHASE

### Validation Phase (Week 1-4)
**Only track:**
1. Interviews completed
2. Pain severity (average)
3. Consistency of problem
4. Stated willingness to pay

**Ignore:** Everything else. Don't track website visitors or social media likes yet.

### Pre-Sale Phase (Week 3)
**Add:**
1. Conversion rate (signup → purchase)
2. Pre-orders
3. Total revenue

**Ignore:** Lifetime value, churn (no customers yet)

### MVP Phase (Week 5-12)
**Add:**
1. Weekly active users
2. Retention (week-over-week)
3. Feature usage
4. NPS

**Ignore:** MRR growth rate (too early, too volatile)

### Scaling Phase (Month 3+)
**Track everything:**
1. MRR & growth rate
2. CAC & LTV
3. Churn rate
4. Unit economics

**This is when you need the full dashboard.**

---

## 🚨 COMMON MISTAKES

### ❌ Tracking Too Many Metrics
**Problem:** Spreadsheet with 50 columns, never updated
**Fix:** Track 5 key metrics max per phase

### ❌ Vanity Metrics
**Problem:** "We have 10,000 Twitter followers!"
**Fix:** Focus on revenue, retention, not followers

### ❌ Not Updating Regularly
**Problem:** Spreadsheet from 3 months ago
**Fix:** Set recurring calendar reminder (Friday 4pm)

### ❌ No Targets
**Problem:** Don't know if 7% churn is good or bad
**Fix:** Set targets based on industry benchmarks (included above)

### ❌ Ignoring Red Flags
**Problem:** Churn at 15% for 3 months, still scaling ads
**Fix:** Use decision logic to trigger immediate action

---

## 📈 BENCHMARK TARGETS (SaaS)

**Use these to set your targets:**

| Metric | Bad | Acceptable | Good | Excellent |
|--------|-----|------------|------|-----------|
| **MRR Growth (monthly)** | <5% | 5-10% | 10-20% | >20% |
| **Churn (monthly)** | >10% | 5-10% | 2-5% | <2% |
| **LTV:CAC** | <3:1 | 3-5:1 | 5-10:1 | >10:1 |
| **NPS** | <0 | 0-30 | 30-50 | >50 |
| **Payback Period** | >6 mo | 3-6 mo | 1-3 mo | <1 mo |
| **Gross Margin** | <50% | 50-70% | 70-85% | >85% |
| **Month 1 Retention** | <40% | 40-60% | 60-80% | >80% |

---

## 🎯 YOUR NEXT ACTION

1. **Copy the spreadsheet for your phase:**
   - Validation → Copy Tab 1 & 3
   - Pre-Sale → Copy full Pre-Sale tracker
   - MVP → Copy full MVP dashboard
   - Scaling → Copy full SaaS dashboard

2. **Add your first data point TODAY:**
   - Log first email sent
   - Log first interview
   - Log first user

3. **Set weekly review:**
   - Every Friday 4pm
   - 15 minutes
   - Update dashboard, note trends

**Track what matters. Ignore the rest.** 📊

---

**Built for builders who measure progress objectively.** 📈

*Now go track your metrics!* 🚀
