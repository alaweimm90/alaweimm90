-- Policies for user_api_keys (re-create without IF NOT EXISTS)
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