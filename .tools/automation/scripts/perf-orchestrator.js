const { AutomationFramework } = require('../core/framework')

async function sleep(ms) {
  return new Promise(resolve => { setTimeout(resolve, ms) })
}

async function main() {
  process.env.STATUS_PORT = process.env.STATUS_PORT || '7081'
  const fw = new AutomationFramework({ logLevel: 'error' })
  await fw.initialize()
  const orchestrator = fw.modules.get('agent-orchestrator')

  const total = Number(process.env.PERF_TOTAL || '50')
  const executions = []
  const start = Date.now()

  for (let i = 0; i < total; i++) {
    const kind = i % 4
    let task
    let requirements
    if (kind === 0) {
      task = { name: `dp-${i}`, payload: Array.from({ length: 10 }, (_, j) => j) }
      requirements = ['data-processing']
    } else if (kind === 1) {
      task = { name: `api-${i}` }
      requirements = ['api-integration']
    } else if (kind === 2) {
      task = { name: `evt-${i}` }
      requirements = ['event-driven']
    } else {
      task = { name: `long-${i}`, duration: 50 }
      requirements = ['long-running']
    }
    const res = await orchestrator.assignTask({ task, requirements })
    executions.push(res.executionId)
  }

  const timeoutAt = Date.now() + Number(process.env.PERF_TIMEOUT || '15000')
  while (true) {
    const completed = executions.filter(id => {
      const e = fw.stateStore.getExecution(id)
      return e && e.status === 'completed'
    }).length
    const failed = executions.filter(id => {
      const e = fw.stateStore.getExecution(id)
      return e && e.status === 'failed'
    }).length
    if (completed + failed >= executions.length) break
    if (Date.now() > timeoutAt) break
    await sleep(50)
  }

  const end = Date.now()
  const agentMetrics = {}
  for (const [name, m] of orchestrator.agentMetrics) {
    agentMetrics[name] = m
  }

  const result = {
    total,
    durationMs: end - start,
    completed: executions.filter(id => fw.stateStore.getExecution(id)?.status === 'completed').length,
    failed: executions.filter(id => fw.stateStore.getExecution(id)?.status === 'failed').length,
    agentMetrics,
    memory: {
      rss: process.memoryUsage().rss,
      heapTotal: process.memoryUsage().heapTotal,
      heapUsed: process.memoryUsage().heapUsed,
    },
  }

  process.stdout.write(JSON.stringify(result))
  await fw.shutdown()
}

main().catch(e => { process.stderr.write(e.message); process.exit(1) })
