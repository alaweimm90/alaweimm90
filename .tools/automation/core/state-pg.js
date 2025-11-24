const { Client } = require('pg')

class PgStateStore {
  constructor(url) {
    this.client = new Client({ connectionString: url })
    this.cache = { workflows: {}, executions: {} }
  }

  async init() {
    await this.client.connect()
    await this.client.query(`CREATE TABLE IF NOT EXISTS workflows (
      id TEXT PRIMARY KEY,
      name TEXT,
      status TEXT,
      context JSONB,
      updated_at TIMESTAMP DEFAULT NOW()
    )`)
    await this.client.query(`CREATE TABLE IF NOT EXISTS executions (
      id TEXT PRIMARY KEY,
      agent TEXT,
      task JSONB,
      status TEXT,
      start_time BIGINT,
      end_time BIGINT,
      updated_at TIMESTAMP DEFAULT NOW()
    )`)
  }

  async setWorkflow(id, data) {
    await this.client.query(
      `INSERT INTO workflows(id, name, status, context, updated_at)
       VALUES($1,$2,$3,$4,NOW())
       ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name, status=EXCLUDED.status, context=EXCLUDED.context, updated_at=NOW()`,
      [id, data.name, data.status, JSON.stringify(data.context || {})]
    )
    this.cache.workflows[id] = data
  }

  async setExecution(id, data) {
    await this.client.query(
      `INSERT INTO executions(id, agent, task, status, start_time, end_time, updated_at)
       VALUES($1,$2,$3,$4,$5,$6,NOW())
       ON CONFLICT(id) DO UPDATE SET agent=EXCLUDED.agent, task=EXCLUDED.task, status=EXCLUDED.status, start_time=EXCLUDED.start_time, end_time=EXCLUDED.end_time, updated_at=NOW()`,
      [id, data.agent, JSON.stringify(data.task || {}), data.status, data.startTime || null, data.endTime || null]
    )
    this.cache.executions[id] = data
  }
  
  getWorkflow(id) { return this.cache.workflows[id] }

  getExecution(id) { return this.cache.executions[id] }

  listWorkflows() { return Object.entries(this.cache.workflows).map(([id, data]) => ({ id, ...data })) }

  listExecutions() { return Object.entries(this.cache.executions).map(([id, data]) => ({ id, ...data })) }

  async shutdown() { await this.client.end() }
}

module.exports = { PgStateStore }
