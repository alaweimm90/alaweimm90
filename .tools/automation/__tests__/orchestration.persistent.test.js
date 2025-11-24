const { AutomationFramework } = require('../core/framework')

jest.setTimeout(30000)

describe('Persistent State & Domain Agents', () => {
  let fw
  let orchestrator

  beforeAll(async () => {
    fw = new AutomationFramework({ logLevel: 'error' })
    await fw.initialize()
    orchestrator = fw.modules.get('agent-orchestrator')
  })

  afterAll(async () => {
    await fw.shutdown()
  })

  test('persists workflow state on steps', async () => {
    const wf = { name: 'persist', steps: [ { name:'dp', requirements:['data-processing'], task:{ name:'dp', payload:[1,2] } }, { name:'api', requirements:['api-integration'], task:{ name:'api' }, dependsOn:['dp'] } ] }
    const res = await orchestrator.orchestrateWorkflow({ workflow: wf })
    expect(res.status).toBe('completed')
    const list = fw.stateStore.listWorkflows()
    expect(list.find(w => w.name === 'persist')).toBeTruthy()
  })

  test('handles long-running task', async () => {
    const r = await orchestrator.assignTask({ task: { name:'long', duration: 200 }, requirements: ['long-running'] })
    expect(['assigned','awaiting-override']).toContain(r.status)
  })

  test('scheduler triggers task', async () => {
    const sched = fw.modules.get('scheduler')
    const s = await sched.schedule({ name:'ping', intervalMs: 50, task: { name:'api', requirements:['api-integration'] } })
    expect(s.status).toBe('scheduled')
    const listed = await sched.list()
    expect(listed.jobs).toContain('ping')
    await sched.cancel({ name:'ping' })
  })
})

