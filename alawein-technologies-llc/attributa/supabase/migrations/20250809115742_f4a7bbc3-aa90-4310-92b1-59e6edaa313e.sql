-- Create table for per-user API keys (fresh)
create table public.user_api_keys (
  user_id uuid primary key references auth.users(id) on delete cascade,
  openai_key text,
  anthropic_key text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Enable Row Level Security
alter table public.user_api_keys enable row level security;

-- Timestamp trigger function
create or replace function public.update_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- Trigger
create trigger trg_user_api_keys_updated_at
before update on public.user_api_keys
for each row execute function public.update_updated_at();

-- Policies
create policy "Users can view their own API keys"
  on public.user_api_keys
  for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert their own API keys"
  on public.user_api_keys
  for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "Users can update their own API keys"
  on public.user_api_keys
  for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Users can delete their own API keys"
  on public.user_api_keys
  for delete
  to authenticated
  using (auth.uid() = user_id);