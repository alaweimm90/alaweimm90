import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'

const policy = JSON.parse(readFileSync('.metaHub/repo-policy.json', 'utf8'))

function staged() {
  const out = execSync('git diff --cached --name-status', { encoding: 'utf8' })
  return out.split('\n').filter(Boolean).map(line => {
    const [status, ...rest] = line.trim().split(/\s+/)
    const file = rest.pop()
    return { status, file }
  })
}

const isAddLike = s => s.startsWith('A') || s.startsWith('C') || s.startsWith('R')
const root = f => f.split('/')[0]
const ext = f => (f.match(/\.[^./]+$/)?.[0] ?? '')

const ALLOW_DOCS = process.env.ALLOW_DOCS === 'true'
const violations = []

for (const { status, file } of staged()) {
  if (!isAddLike(status)) continue

  // Block new docs unless allowed
  const inBlockedDir = policy.blockedDirs.some(d => file.startsWith(`${d}/`))
  const isBlockedExt = policy.blockedExtensions.includes(ext(file))
  const allowlisted = policy.allowlist.includes(file)
  if (!ALLOW_DOCS && (inBlockedDir || isBlockedExt) && !allowlisted) {
    violations.push(`blocked-doc: ${file}`)
  }

  // Enforce allowed roots
  const r = root(file)
  if (!policy.allowedRoots.includes(r) && !policy.rootAllowlist.includes(file)) {
    violations.push(`disallowed-root: ${file}`)
  }
}

if (violations.length) {
  console.error('Structure violations:')
  for (const v of violations) console.error(' -', v)
  process.exit(1)
}
