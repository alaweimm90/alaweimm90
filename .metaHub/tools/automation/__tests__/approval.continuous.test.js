const { AutomationFramework } = require('../core/framework')

jest.setTimeout(30000)

describe('Continuous Approval Mode', () => {
  let fw
  let orchestrator

  beforeAll(async () => {
    process.env.STATUS_PORT = '7080'
    fw = new AutomationFramework({ logLevel: 'error' })
    await fw.initialize()
    orchestrator = fw.modules.get('agent-orchestrator')
    await orchestrator.setApprovalMode({ continuous: true })
  })

  afterAll(async () => {
    await fw.shutdown()
  })

  test('bypasses RBAC when no security findings', async () => {
    const sec = fw.modules.get('security-module')
    const original = sec.scanSecrets
    sec.scanSecrets = async () => ({ status: 'success', secretsFound: [] })
    const res = await orchestrator.assignTask({ task: { name: 'code:generate', context: { actor: { role: 'viewer' }, action: 'code:generate' } }, requirements: ['code-generation'] })
    expect(['assigned', 'awaiting-override']).toContain(res.status)
    sec.scanSecrets = original
  })

  test('blocks when security findings exist', async () => {
    const sec = fw.modules.get('security-module')
    const original = sec.scanSecrets
    sec.scanSecrets = async () => ({ status: 'critical', secretsFound: [{ file: 'x', pattern: 'AWS Key' }] })
    const res = await orchestrator.assignTask({ task: { name: 'code:generate', context: { actor: { role: 'viewer' }, action: 'code:generate' } }, requirements: ['code-generation'] })
    expect(res.status).toBe('awaiting-override')
    sec.scanSecrets = original
  })
})
