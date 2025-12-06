create table if not exists analytics_snapshots (
  id uuid default gen_random_uuid() primary key,
  trace_id text,
  timeframe text not null,
  payload jsonb not null,
  created_at timestamptz default now()
);

create index if not exists analytics_snapshots_trace_idx on analytics_snapshots(trace_id);
