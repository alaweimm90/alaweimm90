import fs from 'node:fs'
import path from 'node:path'
import type { SupabaseClient } from '@supabase/supabase-js'
import type { AgentTask } from '@/agents/turingo/types'

type FnMap = Record<string, string>

function repoRoot(): string { return path.resolve(process.cwd(), '../../../../') }

function mapping(): FnMap {
  const p = path.join(repoRoot(), 'CONFIG', 'workflows', 'transports.supabase.json')
  if (!fs.existsSync(p)) return {}
  const raw = fs.readFileSync(p, 'utf-8')
  const json = JSON.parse(raw)
  return json.map || {}
}

export async function runSupabase(client: SupabaseClient, tasks: AgentTask[]) {
  const results = [] as { name: string; ok: boolean; data?: unknown; error?: string }[]
  const map = mapping()
  for (const t of tasks) {
    const fn = map[t.name] || 'workflow-status'
    const r = await client.functions.invoke(fn as any, { body: t.input as any })
    if (r.error) {
      results.push({ name: t.name, ok: false, error: r.error.message })
    } else {
      results.push({ name: t.name, ok: true, data: r.data })
    }
  }
  return results
}