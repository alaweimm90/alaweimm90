# calla – Execution Pack (CRM, Consult Flows, Aftercare, Metrics)

This document operationalizes **calla** as a high-touch, trust-centered service brand.

---

## 1. CRM Contact Schema

### 1.1 Core Identity

- `full_name` – Name (text)
- `email` – Email (email)
- `phone` – Phone/WhatsApp (phone)
- `country` – Country (dropdown/text)
- `preferred_language` – Preferred Language (dropdown/text)

### 1.2 Relationship & Context

- `referral_source` – Who/where referred them (text)
- `preferred_contact_channel` – Preferred Contact Channel (dropdown)
  - phone
  - whatsapp
  - email
  - in_person
- `primary_concern` – Main reason they reached out (long text)
- `background_context` – Important life/context notes (long text)
- `sensitivity_notes` – Anything to be careful/respectful about (private, long text)

### 1.3 Service & Lifecycle

- `service_type` – Service of interest (dropdown; values defined by your mother)
- `urgency_level` – How urgent is the need? (dropdown)
  - low
  - medium
  - high
- `consult_scheduled_at` – Date/time of upcoming consult (datetime)
- `client_status` – Client Status (dropdown)
  - lead
  - consult_scheduled
  - active_client
  - paused_client
  - past_client

---

## 2. Simple Pipeline (Optional)

If your CRM uses pipelines, define one light pipeline.

### 2.1 Pipeline: `calla_consult_and_care`

Stages:

1. `New Inquiry`
2. `Consult Scheduled`
3. `Consult Completed`
4. `Became Client`
5. `Did Not Become Client`
6. `Ongoing Care`
7. `Completed / Transitioned`

This is mostly to visualize flow; many actions will be manual or semi-automated.

---

## 3. Key Automations

### 3.1 Workflow A – Warm Welcome for New Inquiry

**Trigger:**

- New contact created with brand = calla OR filled “Request a Consult” form.

**Actions:**

1. Set `client_status = lead`.
2. Send warm, gentle welcome email `CALLA_WELCOME_01`:
   - Acknowledge message.
   - Affirm that reaching out was a good step.
   - Briefly outline next steps (you or your mother will respond personally).
3. Internal notification:
   - Send WhatsApp/Slack/Email to Meshal’s mother (and optionally you).
   - Create task “Review inquiry & respond personally within 24 hours”.


### 3.2 Workflow B – Consult Booking & Reminders

Assuming a scheduling link (Calendly/OnceHub/etc.) or manual booking.

**Trigger:**

- Consult booked for calla.

**Actions:**

1. Set `client_status = consult_scheduled`.
2. Set `consult_scheduled_at`.
3. Send confirmation email `CALLA_CONSULT_CONFIRM_01`:
   - Date, time, time zone.
   - How the conversation will happen (phone/Zoom/etc.).
   - Gentle reassurance of what to expect.
4. Schedule reminders:
   - 24 hours before.
   - 3 hours before.


### 3.3 Workflow C – Post-Consult Branching

**Trigger:**

- Consult completed; outcome recorded in CRM:
  - `became_client` (yes/no), plus optional notes.

**If became client (yes):**

1. Set `client_status = active_client`.
2. Enroll in `CALLA_ONBOARDING_01` sequence:
   - Gentle “thank you + what happens next” email.
3. Create initial care plan as internal note (manual but anchored in CRM).

**If did not become client (no):**

1. Set stage to `Did Not Become Client`.
2. Optionally tag reason (not ready, financial, not right fit, etc.).
3. Enroll in light-touch nurture `CALLA_NURTURE_01` (e.g. 3–4 emails over 3–6 months with helpful resources, not salesy).


### 3.4 Workflow D – Aftercare for Active Clients

Most interactions can and should be personal, but some anchors can be automated.

**Trigger:**

- `client_status` changes to `active_client` or new “session completed”.

**Actions:**

1. Optional automated email after a session:
   - Simple check-in: “How are you feeling after our last conversation?”
   - Option to share reflections or questions.
2. Schedule follow-up reminders:
   - “Check in with [Name]” 1–2 weeks after important sessions.
3. For long-term clients, periodic survey or reflection prompts.


### 3.5 Workflow E – Completion & Gentle Follow-up

**Trigger:**

- Client finishes a defined package or decides to pause.

**Actions:**

1. Set `client_status = past_client` or `paused_client`.
2. Enroll in `CALLA_COMPLETION_01`:
   - Gratitude + reflection.
   - Offer a way to return in the future.
3. Optional annual or semi-annual gentle check-in (“thinking of you” style).

---

## 4. Email Sequence Outlines

### 4.1 `CALLA_WELCOME_01` (New Inquiry)

- Tone: soft, reassuring, not pushy.
- Content elements:
  - Thank them for reaching out.
  - One or two lines to normalize their concern or situation.
  - Simple description of what happens next (personal reply, possible consult).
  - Clear note that they’re free to reply directly to the email/WhatsApp.


### 4.2 `CALLA_CONSULT_CONFIRM_01`

- Confirm time and channel.
- Encourage them to bring any notes/questions.
- Normalize any nervousness.


### 4.3 `CALLA_ONBOARDING_01` (New Client)

- Email 1 – “Thank you for trusting calla”
  - Gratitude.
  - Outline structure: sessions, how to reschedule, how to reach out between sessions.
- Email 2 – “How we’ll work together”
  - Short bullet list of how you and your mother show up in the relationship.
  - What they can do to get the most support.


### 4.4 `CALLA_NURTURE_01` (Non-Client / Not Yet Ready)

A few spaced-out emails over months:

- Gentle teachings or reflections.
- Occasional reminder that support is available if/when they’re ready.
- No heavy calls to action.


### 4.5 `CALLA_COMPLETION_01`

- Gratitude.
- Reflect on progress or themes (manually personalized where possible).
- Invite them to check in in the future or share reflections.

---

## 5. Metrics & Dashboards for calla

Keep reporting lightweight but meaningful.

Monthly/quarterly KPIs:

- `new_inquiries` – number of new leads.
- `consults_booked` – number of consults scheduled.
- `consult_show_rate` – consults completed / consults booked.
- `consult_to_client_conversion` – # who become clients / consults completed.
- `active_clients` – number of active clients in a given month.
- `client_retention` – percentage of clients who continue from period to period.

Optionally:

- Breakdown by referral_source.
- Breakdown by service_type.

---

## 6. Implementation Notes

- The priority for calla is **experience quality** over automation volume.
- Start with:
  1. Contact fields focused on relationship context and sensitivity.
  2. A very light pipeline for visualization.
  3. Warm welcome, consult confirmation, and simple post-consult automations.
- Add more automation only where it **reduces friction** (reminders, simple check-ins), not where it would make the experience feel cold or generic.
