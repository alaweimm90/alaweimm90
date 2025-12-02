#!/usr/bin/env node

/**
 * ATLAS Demo Logger
 * Comprehensive logging system for demo operations
 */

const fs = require('fs');
const path = require('path');

class AtlasLogger {
  constructor(logDir = null) {
    this.logDir = logDir || path.join(__dirname, '..', 'logs');
    this.ensureLogDir();

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.logFile = path.join(this.logDir, `atlas-demo-${timestamp}.log`);
    this.metricsFile = path.join(this.logDir, `metrics-${timestamp}.json`);

    this.metrics = {
      startTime: new Date().toISOString(),
      operations: [],
      performance: {},
      errors: [],
      summary: {},
    };
  }

  ensureLogDir() {
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
  }

  log(level, message, metadata = {}) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level: level.toUpperCase(),
      message,
      ...metadata,
    };

    // Console output with colors
    const colors = {
      INFO: '\x1b[36m', // Cyan
      SUCCESS: '\x1b[32m', // Green
      WARNING: '\x1b[33m', // Yellow
      ERROR: '\x1b[31m', // Red
      DEBUG: '\x1b[35m', // Magenta
      RESET: '\x1b[0m',
    };

    const color = colors[level.toUpperCase()] || colors.RESET;
    console.log(`${color}[${timestamp}] ${level.toUpperCase()}: ${message}${colors.RESET}`);

    // File output
    const logLine = JSON.stringify(logEntry) + '\n';
    fs.appendFileSync(this.logFile, logLine);

    // Track metrics
    if (level === 'ERROR') {
      this.metrics.errors.push(logEntry);
    }
  }

  info(message, metadata = {}) {
    this.log('INFO', message, metadata);
  }

  success(message, metadata = {}) {
    this.log('SUCCESS', message, metadata);
  }

  warning(message, metadata = {}) {
    this.log('WARNING', message, metadata);
  }

  error(message, metadata = {}) {
    this.log('ERROR', message, metadata);
  }

  debug(message, metadata = {}) {
    this.log('DEBUG', message, metadata);
  }

  startOperation(operationName, metadata = {}) {
    const operation = {
      name: operationName,
      startTime: new Date().toISOString(),
      status: 'running',
      metadata,
    };

    this.metrics.operations.push(operation);
    this.info(`Starting operation: ${operationName}`, metadata);

    return operation;
  }

  endOperation(operationName, success = true, result = {}) {
    const operation = this.metrics.operations.find(
      (op) => op.name === operationName && op.status === 'running'
    );

    if (operation) {
      operation.endTime = new Date().toISOString();
      operation.status = success ? 'completed' : 'failed';
      operation.result = result;
      operation.duration = new Date(operation.endTime) - new Date(operation.startTime);

      const level = success ? 'SUCCESS' : 'ERROR';
      const message = `Operation ${operationName} ${operation.status} in ${operation.duration}ms`;

      this.log(level, message, { operation, result });
    }
  }

  recordMetric(name, value, unit = '') {
    this.metrics.performance[name] = {
      value,
      unit,
      timestamp: new Date().toISOString(),
    };

    this.debug(`Recorded metric: ${name} = ${value}${unit ? ' ' + unit : ''}`);
  }

  recordAnalysisResult(repository, analysisType, result) {
    if (!this.metrics.analysis) {
      this.metrics.analysis = {};
    }

    if (!this.metrics.analysis[repository]) {
      this.metrics.analysis[repository] = {};
    }

    this.metrics.analysis[repository][analysisType] = {
      result,
      timestamp: new Date().toISOString(),
    };

    this.info(`Analysis result recorded for ${repository}:${analysisType}`, {
      repository,
      analysisType,
      keyMetrics: this.extractKeyMetrics(result),
    });
  }

  extractKeyMetrics(result) {
    const metrics = {};

    if (result.complexityScore !== undefined) {
      metrics.complexityScore = result.complexityScore;
    }
    if (result.chaosLevel !== undefined) {
      metrics.chaosLevel = result.chaosLevel;
    }
    if (result.maintainabilityIndex !== undefined) {
      metrics.maintainabilityIndex = result.maintainabilityIndex;
    }
    if (result.filesAnalyzed !== undefined) {
      metrics.filesAnalyzed = result.filesAnalyzed;
    }
    if (result.totalLines !== undefined) {
      metrics.totalLines = result.totalLines;
    }

    return metrics;
  }

  finalize() {
    this.metrics.endTime = new Date().toISOString();
    this.metrics.duration = new Date(this.metrics.endTime) - new Date(this.metrics.startTime);

    // Calculate summary
    this.metrics.summary = {
      totalOperations: this.metrics.operations.length,
      successfulOperations: this.metrics.operations.filter((op) => op.status === 'completed')
        .length,
      failedOperations: this.metrics.operations.filter((op) => op.status === 'failed').length,
      totalErrors: this.metrics.errors.length,
      totalDuration: this.metrics.duration,
    };

    // Save metrics to file
    fs.writeFileSync(this.metricsFile, JSON.stringify(this.metrics, null, 2));

    this.success('Demo logging finalized', {
      summary: this.metrics.summary,
      logFile: this.logFile,
      metricsFile: this.metricsFile,
    });
  }

  getSummary() {
    return {
      operations: this.metrics.operations.length,
      successful: this.metrics.operations.filter((op) => op.status === 'completed').length,
      failed: this.metrics.operations.filter((op) => op.status === 'failed').length,
      errors: this.metrics.errors.length,
      duration: this.metrics.duration,
    };
  }
}

module.exports = AtlasLogger;

// CLI usage
if (require.main === module) {
  const logger = new AtlasLogger();

  // Example usage
  logger.info('Demo logging system initialized');
  logger.startOperation('test-operation');
  logger.recordMetric('test-metric', 42, 'units');
  logger.endOperation('test-operation', true, { result: 'success' });
  logger.finalize();

  console.log('Logger test completed. Check logs directory for output files.');
}
