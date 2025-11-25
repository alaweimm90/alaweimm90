import { execSync } from 'node:child_process'

function getStaged() {
  const out = execSync('git diff --cached --name-status', { encoding: 'utf8' })
  return out.split('\n').filter(Boolean).map(line => {
    const [status, ...rest] = line.trim().split(/\s+/)
    const file = rest.pop()
    return { status, file }
  })
}

const blockedPatterns = [
  /^docs\//,
  /^\.metaHub\/docs\//,
  /\.md$/,
  /\.adoc$/
]

const allowlist = new Set([
  'README.md',
  '.metaHub/docs/README.md'
])

const allowEnv = process.env.ALLOW_DOCS === 'true'
const staged = getStaged()
const violations = []

for (const { status, file } of staged) {
  if (allowEnv) continue
  if (allowlist.has(file)) continue
  const isAdd = status.startsWith('A') || status.startsWith('C') || status.startsWith('R')
  const matches = blockedPatterns.some(rx => rx.test(file))
  if (matches && isAdd) violations.push(file)
}

if (violations.length) {
  console.error('Blocked doc additions:', violations.join(', '))
  process.exit(1)
}
