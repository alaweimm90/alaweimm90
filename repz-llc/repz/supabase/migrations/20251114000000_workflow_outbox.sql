create table if not exists workflows (
  id uuid default gen_random_uuid() primary key,
  type text not null,
  status text not null,
  attempts integer default 0,
  trace_id text unique,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists workflows_trace_id_idx on workflows(trace_id);

create table if not exists workflow_steps (
  id uuid default gen_random_uuid() primary key,
  workflow_id uuid references workflows(id) on delete cascade,
  step text not null,
  status text not null,
  attempts integer default 0,
  error_code text,
  error_message text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists workflow_steps_workflow_step_idx on workflow_steps(workflow_id, step);

create table if not exists outbox (
  id uuid default gen_random_uuid() primary key,
  event_type text not null,
  payload jsonb not null,
  trace_id text,
  published boolean default false,
  created_at timestamptz default now()
);

create index if not exists outbox_published_event_idx on outbox(published, event_type);
