const { BaseModule } = require('../core/framework')

class SchedulerModule extends BaseModule {
  constructor(framework) {
    super(framework)
    this.name = 'scheduler'
    this.jobs = new Map()
    this.registerTask('schedule', this.schedule.bind(this))
    this.registerTask('cancel', this.cancel.bind(this))
    this.registerTask('list', this.list.bind(this))
  }

  async init() { await super.init() }

  async schedule(params) {
    const { name, intervalMs, task } = params
    if (this.jobs.has(name)) return { status: 'exists' }
    const timer = setInterval(async () => {
      try {
        await this.framework.modules.get('agent-orchestrator').assignTask({ task, requirements: task.requirements || [] })
      } catch (e) {
        this.logger.warn('scheduled_task_error', { name, error: e.message })
      }
    }, intervalMs)
    this.jobs.set(name, timer)
    return { status: 'scheduled', name }
  }

  async cancel(params) {
    const { name } = params
    const t = this.jobs.get(name)
    if (!t) return { status: 'not-found' }
    clearInterval(t)
    this.jobs.delete(name)
    return { status: 'cancelled', name }
  }

  async list() {
    return { jobs: Array.from(this.jobs.keys()) }
  }
}

module.exports = SchedulerModule
