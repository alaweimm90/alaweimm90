import fs from 'node:fs'
import path from 'node:path'
import type { AgentTask } from '@/agents/turingo/types'
import { request } from '@/lib/net/http'

function repoRoot(): string { return path.resolve(process.cwd(), '../../../../') }

function readMcpConfig() {
  const p = path.join(repoRoot(), '.mcp', 'config.json')
  if (!fs.existsSync(p)) return null as any
  const raw = fs.readFileSync(p, 'utf-8')
  return JSON.parse(raw) as {
    servers: Record<string, { transport: string; url: string; headers?: Record<string, string>; authToken?: string }>
  }
}

export async function runMcp(
  tasks: AgentTask[],
  opts?: { timeoutMs?: number; retries?: number; backoffMs?: number }
) {
  const cfg = readMcpConfig()
  const server = cfg?.servers?.supabase?.url || ''
  const headersFromCfg = cfg?.servers?.supabase?.headers || {}
  const tokenFromCfg = cfg?.servers?.supabase?.authToken || ''
  const tokenFromEnv = process.env.MCP_TOKEN || ''
  const authToken = tokenFromEnv || tokenFromCfg
  const results = [] as { name: string; ok: boolean; data?: unknown; error?: string }[]
  for (const t of tasks) {
    if (!server) { results.push({ name: t.name, ok: false, error: 'mcp_server_missing' }); continue }
    const traceId = typeof t.input?.traceId === 'string' ? String(t.input.traceId) : undefined
    const ts = Date.now().toString()
    const headers: Record<string, string> = {
      ...headersFromCfg,
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      ...(traceId ? { 'x-trace-id': traceId } : {}),
      'x-run-ts': ts,
    }
    const r = await request(server, { agent: t.name, input: t.input }, { ...opts, headers })
    if (r.ok) results.push({ name: t.name, ok: true, data: r.data })
    else results.push({ name: t.name, ok: false, error: r.error })
  }
  return results
}