/**
 * Core Automation Framework
 * Modular, extensible automation system with robust error handling and logging
 */

const fs = require('fs').promises;
const path = require('path');
const { EventEmitter } = require('events');
let winston;
const { AuditTrail, ExecutionTracker, StatusServer } = require('./execution');
const { StateStore } = require('./state')

/**
 * Core Framework Class
 * Central orchestration system for all automation tasks
 */
class AutomationFramework extends EventEmitter {
  constructor(config = {}) {
    super();

    this.config = {
      name: 'GitHub Monorepo Automation Framework',
      version: '2.0.0',
      logLevel: config.logLevel || 'info',
      logDir: config.logDir || path.join(__dirname, '../logs'),
      modulesDir: config.modulesDir || path.join(__dirname, '../modules'),
      pluginsDir: config.pluginsDir || path.join(__dirname, '../plugins'),
      enableMetrics: config.enableMetrics !== false,
      enableHealthCheck: config.enableHealthCheck !== false,
      retryPolicy: {
        maxRetries: config.maxRetries || 3,
        retryDelay: config.retryDelay || 1000,
        backoffMultiplier: config.backoffMultiplier || 2,
      },
      ...config,
    };

    this.modules = new Map();
    this.plugins = new Map();
    this.tasks = new Map();
    this.agents = new Map();
    this.metrics = {
      tasksExecuted: 0,
      tasksSucceeded: 0,
      tasksFailed: 0,
      totalExecutionTime: 0,
    };

    this.initialized = false;
    this.logger = null;

    this.statusServer = null;
    this.auditTrail = null;
    this.executionTracker = null;

    this.setupLogging();
  }

  /**
   * Setup comprehensive logging system
   */
  setupLogging() {
    let combine, fmtTimestamp, printf, colorize, json

    // Custom format for console output (defined only when winston is available)
    let consoleFormat

    try {
      winston = require('winston')
      ;({ combine, timestamp: fmtTimestamp, printf, colorize, json } = winston.format)
      // Console formatter
      consoleFormat = printf(({ level, message, timestamp: time, ...metadata }) => {
        let msg = `${time} [${level}]: ${message}`;
        if (Object.keys(metadata).length > 0) {
          msg += ` ${JSON.stringify(metadata)}`;
        }
        return msg;
      });

      // Create logger instance
      this.logger = winston.createLogger({
        level: this.config.logLevel,
        format: combine(fmtTimestamp({ format: 'YYYY-MM-DD HH:mm:ss' }), json()),
        transports: [
          new winston.transports.Console({
            format: combine(colorize(), fmtTimestamp({ format: 'YYYY-MM-DD HH:mm:ss' }), consoleFormat),
          }),
          new winston.transports.File({
            filename: path.join(this.config.logDir, 'automation.log'),
            maxsize: 5242880,
            maxFiles: 5,
          }),
          new winston.transports.File({
            filename: path.join(this.config.logDir, 'errors.log'),
            level: 'error',
            maxsize: 5242880,
            maxFiles: 5,
          }),
        ],
        exceptionHandlers: [
          new winston.transports.File({
            filename: path.join(this.config.logDir, 'exceptions.log'),
          }),
        ],
        rejectionHandlers: [
          new winston.transports.File({
            filename: path.join(this.config.logDir, 'rejections.log'),
          }),
        ],
      });
    } catch (e) {
      // Fallback lightweight logger
      const noop = () => {}
      this.logger = {
        info: console.log,
        warn: console.warn,
        error: console.error,
        debug: this.config.logLevel === 'debug' ? console.debug : noop,
      }
    }

    // Create log directory if it doesn't exist
    this.ensureLogDirectory();

    const statusPort = Number(process.env.STATUS_PORT || '7070');
    this.statusServer = new StatusServer({ port: statusPort, logger: console });
    this.statusServer.start();
    this.auditTrail = new AuditTrail(this.config.logDir);
    this.executionTracker = new ExecutionTracker({
      logger: this.logger,
      audit: this.auditTrail,
      broadcast: e => this.statusServer.broadcast(e),
    });
    const pgUrl = process.env.STATE_PG_URL
    if (pgUrl) {
      const { PgStateStore } = require('./state-pg')
      this.stateStore = new PgStateStore(pgUrl)
    } else {
      this.stateStore = new StateStore(this.config.logDir)
    }
  }

  /**
   * Ensure log directory exists
   */
  async ensureLogDirectory() {
    try {
      await fs.mkdir(this.config.logDir, { recursive: true });
    } catch (error) {
      console.error(`Failed to create log directory: ${error.message}`);
    }
  }

  /**
   * Initialize the framework
   */
  async initialize() {
    if (this.initialized) {
      this.logger.warn('Framework already initialized');
      return;
    }

    this.logger.info('Initializing Automation Framework', {
      version: this.config.version,
      modules: this.config.modulesDir,
      plugins: this.config.pluginsDir,
    });

    try {
      // Load core modules
      await this.loadModules();

      // Load plugins
      await this.loadPlugins();

      // Initialize agents
      await this.initializeAgents();

      // Setup health check
      if (this.config.enableHealthCheck) {
        this.setupHealthCheck();
      }

      this.initialized = true;
      this.emit('initialized');
      this.logger.info('Framework initialized successfully');
      await this.executionTracker.emit('framework_initialized', {
        modules: this.modules.size,
        plugins: this.plugins.size,
      });
    } catch (error) {
      this.logger.error('Framework initialization failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Load automation modules
   */
  async loadModules() {
    this.logger.info('Loading automation modules');

    try {
      const files = await fs.readdir(this.config.modulesDir);
      const moduleFiles = files.filter(f => f.endsWith('.js'));

      for (const file of moduleFiles) {
        const modulePath = path.join(this.config.modulesDir, file);
        const moduleName = path.basename(file, '.js');

        try {
          const ModuleClass = require(modulePath);
          const moduleInstance = new ModuleClass(this);

          this.modules.set(moduleName, moduleInstance);
          this.logger.info(`Module loaded: ${moduleName}`);

          if (typeof moduleInstance.init === 'function') {
            await moduleInstance.init();
          }

          // Auto-register module tasks
          if (moduleInstance.tasks) {
            for (const [taskName, taskHandler] of Object.entries(moduleInstance.tasks)) {
              this.registerTask(`${moduleName}:${taskName}`, taskHandler.bind(moduleInstance));
            }
          }
        } catch (error) {
          this.logger.error(`Failed to load module ${moduleName}`, { error: error.message });
        }
      }

      this.logger.info(`Loaded ${this.modules.size} modules`);
    } catch (error) {
      if (error.code === 'ENOENT') {
        this.logger.warn('Modules directory not found, creating it');
        await fs.mkdir(this.config.modulesDir, { recursive: true });
      } else {
        throw error;
      }
    }
  }

  /**
   * Load plugins
   */
  async loadPlugins() {
    this.logger.info('Loading plugins');

    try {
      const files = await fs.readdir(this.config.pluginsDir);
      const pluginFiles = files.filter(f => f.endsWith('.js'));

      for (const file of pluginFiles) {
        const pluginPath = path.join(this.config.pluginsDir, file);
        const pluginName = path.basename(file, '.js');

        try {
          const PluginClass = require(pluginPath);
          const pluginInstance = new PluginClass(this);

          this.plugins.set(pluginName, pluginInstance);

          // Initialize plugin if it has an init method
          if (typeof pluginInstance.init === 'function') {
            await pluginInstance.init();
          }

          this.logger.info(`Plugin loaded: ${pluginName}`);
        } catch (error) {
          this.logger.error(`Failed to load plugin ${pluginName}`, { error: error.message });
        }
      }

      this.logger.info(`Loaded ${this.plugins.size} plugins`);
    } catch (error) {
      if (error.code === 'ENOENT') {
        this.logger.warn('Plugins directory not found, creating it');
        await fs.mkdir(this.config.pluginsDir, { recursive: true });
      } else {
        throw error;
      }
    }
  }

  /**
   * Register a task
   * @param name
   * @param handler
   */
  registerTask(name, handler) {
    if (this.tasks.has(name)) {
      this.logger.warn(`Task ${name} already registered, overwriting`);
    }

    this.tasks.set(name, handler);
    this.logger.debug(`Task registered: ${name}`);
  }

  /**
   * Execute a task with retry logic and error handling
   * @param taskName
   * @param params
   * @param options
   */
  async executeTask(taskName, params = {}, options = {}) {
    if (!this.tasks.has(taskName)) {
      const error = new Error(`Task not found: ${taskName}`);
      this.logger.error(error.message);
      throw error;
    }

    const startTime = Date.now();
    const retryPolicy = { ...this.config.retryPolicy, ...options.retry };
    let lastError;

    this.logger.info(`Executing task: ${taskName}`, { params });
    await this.executionTracker.emit('task_start', { taskName, params });
    this.emit('task:start', { taskName, params });

    for (let attempt = 1; attempt <= retryPolicy.maxRetries; attempt++) {
      try {
        const handler = this.tasks.get(taskName);
        const result = await handler(params, { logger: this.logger, framework: this });

        const executionTime = Date.now() - startTime;

        // Update metrics
        if (this.config.enableMetrics) {
          this.metrics.tasksExecuted++;
          this.metrics.tasksSucceeded++;
          this.metrics.totalExecutionTime += executionTime;
        }

        this.logger.info(`Task completed: ${taskName}`, {
          executionTime: `${executionTime}ms`,
          attempt,
        });
        await this.executionTracker.emit('task_success', {
          taskName,
          result,
          executionTime,
          attempt,
        });
        this.emit('task:success', { taskName, result, executionTime });
        return result;
      } catch (error) {
        lastError = error;
        this.logger.error(
          `Task failed (attempt ${attempt}/${retryPolicy.maxRetries}): ${taskName}`,
          {
            error: error.message,
            stack: error.stack,
          }
        );
        await this.executionTracker.emit('task_error', { taskName, error: error.message, attempt });

        if (attempt < retryPolicy.maxRetries) {
          const delay =
            retryPolicy.retryDelay * retryPolicy.backoffMultiplier**(attempt - 1);
          this.logger.info(`Retrying in ${delay}ms...`);
          await this.sleep(delay);
        }
      }
    }

    // All retries failed
    if (this.config.enableMetrics) {
      this.metrics.tasksExecuted++;
      this.metrics.tasksFailed++;
    }

    this.emit('task:failure', { taskName, error: lastError });
    await this.executionTracker.emit('task_failure', { taskName, error: lastError?.message });
    throw lastError;
  }

  /**
   * Execute multiple tasks in parallel
   * @param tasks
   */
  async executeParallel(tasks) {
    this.logger.info(`Executing ${tasks.length} tasks in parallel`);

    const promises = tasks.map(({ name, params, options }) =>
      this.executeTask(name, params, options).catch(error => ({
        task: name,
        error: error.message,
        failed: true,
      }))
    );

    const results = await Promise.all(promises);

    const failed = results.filter(r => r && r.failed);
    if (failed.length > 0) {
      this.logger.warn(`${failed.length} tasks failed in parallel execution`, { failed });
    }

    return results;
  }

  /**
   * Execute tasks in sequence
   * @param tasks
   */
  async executeSequence(tasks) {
    this.logger.info(`Executing ${tasks.length} tasks in sequence`);
    const results = [];

    for (const { name, params, options } of tasks) {
      try {
        const result = await this.executeTask(name, params, options);
        results.push({ task: name, result, success: true });
      } catch (error) {
        results.push({ task: name, error: error.message, success: false });

        // Stop on failure unless specified otherwise
        if (!options?.continueOnFailure) {
          this.logger.error('Sequence execution stopped due to failure');
          break;
        }
      }
    }

    return results;
  }

  /**
   * Initialize agent system
   */
  async initializeAgents() {
    this.logger.info('Initializing agent system');

    // Agent orchestration will be implemented in separate module
    this.emit('agents:initializing');
  }

  /**
   * Register an agent
   * @param name
   * @param agent
   */
  registerAgent(name, agent) {
    if (this.agents.has(name)) {
      this.logger.warn(`Agent ${name} already registered, overwriting`);
    }

    this.agents.set(name, agent);
    this.logger.info(`Agent registered: ${name}`);
    this.emit('agent:registered', { name, agent });
  }

  /**
   * Setup health check endpoint
   */
  setupHealthCheck() {
    this.healthCheckInterval = setInterval(() => {
      const health = this.getHealthStatus();
      this.emit('health:check', health);
      this.executionTracker.emit('health_check', health);

      if (health.status !== 'healthy') {
        this.logger.warn('Health check detected issues', { health });
      }
    }, 60000); // Check every minute
  }

  /**
   * Get health status
   */
  getHealthStatus() {
    const status = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      modules: {
        loaded: this.modules.size,
        active: Array.from(this.modules.keys()),
      },
      plugins: {
        loaded: this.plugins.size,
        active: Array.from(this.plugins.keys()),
      },
      tasks: {
        registered: this.tasks.size,
        executed: this.metrics.tasksExecuted,
        succeeded: this.metrics.tasksSucceeded,
        failed: this.metrics.tasksFailed,
      },
      agents: {
        registered: this.agents.size,
        active: Array.from(this.agents.keys()),
      },
      memory: process.memoryUsage(),
    };

    // Check for issues
    if (this.metrics.tasksFailed > this.metrics.tasksSucceeded * 0.1) {
      status.status = 'degraded';
      status.issues = ['High task failure rate'];
    }

    if (status.memory.heapUsed > status.memory.heapTotal * 0.9) {
      status.status = 'critical';
      status.issues = [...(status.issues || []), 'High memory usage'];
    }

    return status;
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down framework');

    // Clear intervals
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    // Shutdown agents
    for (const [name, agent] of this.agents) {
      if (typeof agent.shutdown === 'function') {
        try {
          await agent.shutdown();
          this.logger.info(`Agent shutdown: ${name}`);
        } catch (error) {
          this.logger.error(`Failed to shutdown agent ${name}`, { error: error.message });
        }
      }
    }

    // Shutdown modules
    for (const [name, module] of this.modules) {
      if (typeof module.shutdown === 'function') {
        try {
          await module.shutdown();
          this.logger.info(`Module shutdown: ${name}`);
        } catch (error) {
          this.logger.error(`Failed to shutdown module ${name}`, { error: error.message });
        }
      }
    }

    // Final metrics
    this.logger.info('Final metrics', this.metrics);
    await this.executionTracker.emit('framework_shutdown', { metrics: this.metrics });
    if (this.statusServer) this.statusServer.stop();

    this.emit('shutdown');
    this.initialized = false;
  }

  /**
   * Utility: Sleep function
   * @param ms
   */
  sleep(ms) {
    return new Promise(resolve => { setTimeout(resolve, ms) });
  }
}

/**
 * Base Module Class
 * All modules should extend this class
 */
class BaseModule {
  constructor(framework) {
    this.framework = framework;
    this.logger = framework.logger;
    this.name = this.constructor.name;
    this.tasks = {};
  }

  /**
   * Initialize module
   */
  async init() {
    this.logger.info(`Initializing module: ${this.name}`);
  }

  /**
   * Shutdown module
   */
  async shutdown() {
    this.logger.info(`Shutting down module: ${this.name}`);
  }

  /**
   * Register a task handler
   * @param name
   * @param handler
   */
  registerTask(name, handler) {
    this.tasks[name] = handler;
  }
}

/**
 * Base Agent Class
 * All agents should extend this class
 */
class BaseAgent extends EventEmitter {
  constructor(name, framework) {
    super();
    this.name = name;
    this.framework = framework;
    this.logger = framework.logger;
    this.status = 'idle';
    this.taskQueue = [];
    this.currentTask = null;
  }

  /**
   * Initialize agent
   */
  async init() {
    this.logger.info(`Initializing agent: ${this.name}`);
    this.status = 'ready';
    this.emit('initialized');
  }

  /**
   * Execute a task
   * @param task
   */
  async execute(task) {
    this.status = 'busy';
    this.currentTask = task;
    this.emit('task:start', task);

    try {
      const result = await this.processTask(task);
      this.emit('task:complete', { task, result });
      return result;
    } catch (error) {
      this.emit('task:error', { task, error });
      throw error;
    } finally {
      this.status = 'ready';
      this.currentTask = null;
    }
  }

  /**
   * Process task - to be overridden by specific agents
   * @param task
   */
  async processTask(task) {
    throw new Error(`Agent ${this.name} must implement processTask method`);
  }

  /**
   * Shutdown agent
   */
  async shutdown() {
    this.logger.info(`Shutting down agent: ${this.name}`);
    this.status = 'shutdown';
    this.emit('shutdown');
  }
}

// Export the framework and base classes
module.exports = {
  AutomationFramework,
  BaseModule,
  BaseAgent,
};
