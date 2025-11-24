const { join } = require('path')
const { promises: fs } = require('fs')

class StateStore {
  constructor(baseDir) {
    this.baseDir = baseDir
    this.file = join(baseDir, 'state', 'store.jsonl')
    this.cache = { workflows: {}, executions: {} }
  }

  async init() {
    await fs.mkdir(join(this.baseDir, 'state'), { recursive: true })
    try {
      const data = await fs.readFile(this.file, 'utf8')
      for (const line of data.split(/\r?\n/)) {
        if (!line) continue
        const evt = JSON.parse(line)
        if (evt.type === 'workflow') this.cache.workflows[evt.id] = evt.data
        if (evt.type === 'execution') this.cache.executions[evt.id] = evt.data
      }
    } catch (e) {
      if (e && e.code !== 'ENOENT') {
        throw e
      }
    }
  }
  
  async write(entry) {
    await fs.appendFile(this.file, `${JSON.stringify(entry)}\n`)
  }
  
  async setWorkflow(id, data) {
    this.cache.workflows[id] = data
    await this.write({ type: 'workflow', id, data })
  }
  
  async setExecution(id, data) {
    this.cache.executions[id] = data
    await this.write({ type: 'execution', id, data })
  }
  
  getWorkflow(id) {
    return this.cache.workflows[id]
  }
  
  getExecution(id) {
    return this.cache.executions[id]
  }
  
  listWorkflows() {
    return Object.entries(this.cache.workflows).map(([id, data]) => ({ id, ...data }))
  }
  
  listExecutions() {
    return Object.entries(this.cache.executions).map(([id, data]) => ({ id, ...data }))
  }
}

module.exports = { StateStore }
