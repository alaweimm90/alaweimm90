# benchbarrier – Execution Pack (CRM, Events/Programs, Automations, Commissions)

This document operationalizes **benchbarrier** as a performance brand with events, assessments, and programs, plus your special commission structure.

---

## 1. CRM Contact Schema

### 1.1 Core Identity

- `full_name` – Name (text)
- `email` – Email (email)
- `phone` – Phone (phone)
- `country` – Country (dropdown/text)

### 1.2 Performance Profile

- `athlete_type` – Athlete Type (dropdown)
  - individual
  - team
  - coach
  - organization
- `sport_or_domain` – Sport / Performance Domain (text)
- `team_or_gym_name` – Team / Gym / Organization (text)
- `current_barrier` – Current Performance Barrier (long text)
- `performance_goal` – Performance Goal (long text)

### 1.3 Funnel & Segmentation

- `source_channel` – Source Channel (dropdown)
  - instagram
  - youtube
  - tiktok
  - referral
  - web
  - event
  - podcast
  - other
- `campaign_tag` – Campaign / Offer Tag (text)
- `joined_event` – Last Event / Clinic ID (text or relation)
- `joined_program` – Current Program ID or Level (text/dropdown)
- `engagement_score` – Engagement Score (number)
- `lifecycle_stage` – Lifecycle Stage (dropdown)
  - lead
  - event_attendee
  - assessment_client
  - program_client
  - alumni

---

## 2. Pipelines

Use **two pipelines** for clarity: one for events/clinics, one for structured programs.

### 2.1 Pipeline: `benchbarrier_events_and_clinics`

Stages:

1. `Registered`
2. `Confirmed`
3. `Attended`
4. `Upsell Offered`
5. `Upgraded`
6. `No Upgrade`

Usage:

- Tracks a contact’s journey through a specific event or clinic.
- Each deal can represent a registration for a particular event.


### 2.2 Pipeline: `benchbarrier_program_enrollment`

Stages:

1. `Applied`
2. `Accepted`
3. `Onboarding`
4. `Active Program`
5. `Renewed`
6. `Completed`

Usage:

- Tracks entry and lifecycle in longer-term programs or coaching packages.

---

## 3. Key Automations

### 3.1 Workflow A – Event Registration → Reminders & Check-in

**Trigger:**

- New event registration form submitted (with `brand_id = benchbarrier`).

**Actions:**

1. Create/update contact with:
   - `lifecycle_stage >= event_attendee`.
   - `joined_event` set to event ID or name.
2. Create a deal in `benchbarrier_events_and_clinics`:
   - Stage: `Registered`.
   - Amount: event fee (if paid upfront), or 0 for free events.
3. Send confirmation email:
   - Event details: date, time, location/link, what to bring.
4. Schedule reminders:
   - 7 days before (if applicable).
   - 24 hours before.
   - 3 hours before.
5. Internal:
   - Add to attendee list for that event.

**Post-Event:**

- Upon marking attendance:
  - If `attended = true` → move deal to `Attended`.
  - Trigger follow-up sequence `BB_EVENT_FOLLOWUP_01`.


### 3.2 Workflow B – Event Follow-up → Program Upsell

**Trigger:**

- Deal in `benchbarrier_events_and_clinics` moves to `Attended`.

**Actions:**

1. Enroll contact in sequence `BB_EVENT_FOLLOWUP_01`:
   - Email 1 (same day): thank you + recap + highlight key wins.
   - Email 2 (Day 1–2): personalized assessment offer or program recommendation.
   - Email 3 (Day 3–5): case study / testimonial + clear program CTA.
2. If contact clicks key upsell link or books a call:
   - Create a deal in `benchbarrier_program_enrollment` at stage `Applied` or `Accepted` (depending on flow).
   - Tag contact with `interested_in_program`.
3. If contact purchases program (via payment event):
   - Move program deal to `Accepted` or `Onboarding`.
   - Set `lifecycle_stage = program_client`.


### 3.3 Workflow C – Assessment Application → Acceptance & Payment

**Trigger:**

- Assessment application form submitted.

**Fields to capture:**

- `sport_or_domain`, `current_barrier`, `performance_goal`, `athlete_type`, `team_or_gym_name`.

**Actions:**

1. Create/update contact with performance fields.
2. Create program-enrollment deal:
   - Pipeline: `benchbarrier_program_enrollment`.
   - Stage: `Applied`.
3. Auto-score or route to Rio/John:
   - Add internal fields like `assessment_priority` (low/med/high).
   - Notify relevant owner.
4. If **accepted**:
   - Move deal to `Accepted`.
   - Send payment link email.
5. If payment completed (via Stripe/PayPal event):
   - Move deal to `Onboarding`.
   - Set `lifecycle_stage = assessment_client` or `program_client`.
   - Enroll in `BB_ONBOARDING_01` sequence.


### 3.4 Workflow D – Program Lifecycle & Renewal

**Triggers:**

- Deal moves to `Active Program`.
- Program duration or end date is reached.

**Actions when `Active Program` starts:**

1. Enroll in `BB_PROGRAM_ACTIVE_01` (check-ins, expectations, resources).
2. Create recurring tasks for check-ins (weekly/bi-weekly).

**Actions near program end:**

1. 2–3 weeks before end, create task “Discuss renewal/next level”.
2. Send 2–3-email renewal sequence.
3. If renewed:
   - Move deal to `Renewed`.
4. If not renewed but completed:
   - Move to `Completed`.
   - Tag `alumni`.

---

## 4. Email Sequence Outlines

### 4.1 `BB_EVENT_CONFIRMATION_01`

Sent immediately after registration.

- Subject example: “You’re in – details for [Event Name]”.
- Content:
  - Confirmation + gratitude.
  - Event logistics: time, place/Zoom link, what to bring.
  - Short statement of what they can expect to achieve.


### 4.2 `BB_EVENT_FOLLOWUP_01`

**Email 1 – "How to lock in your gains from [Event]" (same day)**

- Recap key principles.
- Simple at-home or follow-up protocol.
- Soft invite to program/assessment.

**Email 2 – "Where most athletes get stuck again" (Day 1–2)**

- Explain why implementation drops after events.
- Introduce ongoing support/program as solution.
- Clear CTA to book assessment or join program.

**Email 3 – "Case study: breaking the barrier" (Day 3–5)**

- Short story / anonymized example.
- Strong CTA for program enrollment.


### 4.3 `BB_ONBOARDING_01`

For new assessment/program clients.

**Email 1 – "Welcome to benchbarrier – next steps"**

- What happens in the first 7 days.
- How to prepare: training logs, videos, baseline tests.

**Email 2 – "How we measure progress"**

- Explain metrics (strength, power, speed, recovery, etc. depending on domain).
- Overview of check-ins and adjustments.

**Email 3 – "How to get the most from coaching"**

- Response time expectations.
- Communication channels.
- Examples of ideal client behaviors.

---

## 5. Commissions & Revenue Reporting (benchbarrier)

benchbarrier has a **special commission structure for Meshal**:

- **5%** on **all** sales (global sales share).
- **20%** on revenue from **marketing, sales, promotions, social media, and broadcasting** sources.

These rules are already encoded conceptually in `portfolio-config.yaml`. Here is how to report and visualize them.

### 5.1 Source Channel Mapping

For each `revenue_event` (Stripe/PayPal transaction), capture or infer:

- `brand_id = benchbarrier`
- `source_channel` (enum, mapped to marketing/sales/social/broadcast where relevant)

Examples to treat as **20% bonus eligible**:

- `source_channel` in:
  - `marketing`
  - `sales`
  - `promotions`
  - `social_media`
  - `broadcasting`

### 5.2 Commission Calculation Logic (Summary)

For each benchbarrier `revenue_event` with `amount`:

1. **Base commission (5%)** – applies to **all** benchbarrier events:
   - `base_commission = amount * 0.05`
2. **Bonus commission (20%)** – applies when source is marketing/sales/social/broadcast:
   - If `source_channel` ∈ {marketing, sales, promotions, social_media, broadcasting}:
     - `bonus_commission = amount * 0.20`
   - Else:
     - `bonus_commission = 0`
3. **Total commission for Meshal on that transaction**:
   - `total_commission = base_commission + bonus_commission`

These can be written into `commission_entries` as:

- One or two rows per `revenue_event` (base + bonus), or combined into one row with `rate_percent = effective_rate`.


### 5.3 Dashboards for Commissions

Create a **benchbarrier commissions** dashboard with at least:

- Total benchbarrier revenue for period (gross and net).
- Total commission to Meshal:
  - Base 5% portion.
  - Extra 20% marketing/sales/social/broadcast portion.
- Effective commission rate (% of total revenue).
- Breakdown by **source_channel** and **product/program**.

This relies on:

- `revenue_events` filtered by `brand_id = benchbarrier`.
- Joined or derived `commission_entries` data for Meshal.


### 5.4 Example Metrics (Monthly)

- `bb_total_revenue` – sum of `revenue_events.amount` for benchbarrier.
- `bb_meshal_base_commission` – sum of 5% rows.
- `bb_meshal_bonus_commission` – sum of 20% rows.
- `bb_meshal_total_commission` – base + bonus.
- `bb_commission_rate_percent` – `(bb_meshal_total_commission / bb_total_revenue) * 100`.

---

## 6. Metrics & Dashboards for benchbarrier

Key monthly KPIs:

- `event_registrations` – number of contacts registered for events.
- `event_attendance_rate` – attendees / registrations.
- `event_to_program_conversion` – # who attend and then join a program / attendees.
- `program_clients` – number of active program clients.
- `avg_revenue_per_event` – total revenue per event.
- Meshal commission totals (see 5.4).

Optional breakdowns:

- By sport or domain.
- By athlete_type (individual vs team vs organization).
- By source_channel (e.g. IG vs YouTube vs referrals).

---

## 7. Implementation Notes

- Start by mirroring the **repz setup process**:
  1. Create contact fields for performance profile and funnel.
  2. Create both pipelines and stages.
  3. Implement core workflows (event registration, event follow-up, assessment, program lifecycle).
  4. Ensure Stripe/PayPal events for benchbarrier set `brand_id` and `source_channel` correctly.
- Once wired, configure BI dashboards or CRM reports with the metrics and commission views defined above.
