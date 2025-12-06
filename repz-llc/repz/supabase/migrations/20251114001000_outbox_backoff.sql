alter table outbox add column if not exists attempts integer default 0;
alter table outbox add column if not exists next_run_at timestamptz;
alter table outbox add column if not exists last_error text;
create index if not exists outbox_next_run_idx on outbox(next_run_at) where published = false;
