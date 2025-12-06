create unique index if not exists analytics_snapshots_trace_time_idx
  on analytics_snapshots(trace_id, timeframe);