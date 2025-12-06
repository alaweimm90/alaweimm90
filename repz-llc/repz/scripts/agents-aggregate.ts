import fs from 'node:fs'
import path from 'node:path'

function repoRoot() { return path.resolve(process.cwd(), '../../../../') }

function listJson(dir: string) {
  if (!fs.existsSync(dir)) return [] as string[]
  return fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => path.join(dir, f))
}

function read(file: string) { return JSON.parse(fs.readFileSync(file, 'utf-8')) }

function aggregate(dir: string) {
  const files = listJson(dir)
  const runs = files.map(read)
  const total = runs.length
  const byWorkflow = new Map<string, number>()
  for (const r of runs) byWorkflow.set(r.workflow, (byWorkflow.get(r.workflow) || 0) + 1)
  const summary = {
    total,
    workflows: Array.from(byWorkflow.entries()).map(([workflow, count]) => ({ workflow, count })),
    outputs: files
  }
  const out = path.join(dir, 'dashboard.json')
  fs.writeFileSync(out, JSON.stringify(summary, null, 2))
  console.log(JSON.stringify(summary, null, 2))
}

const argDir = process.argv[2] || path.join(repoRoot(), 'outputs', 'agents', 'global')
aggregate(argDir)