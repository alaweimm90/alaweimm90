/**
 * Tests for logger utilities
 */

import { createLogger } from '../logger';

describe('Logger Utilities', () => {
  describe('createLogger', () => {
    it('should create a logger with default options', () => {
      const logger = createLogger();
      expect(logger).toBeDefined();
      expect(logger.info).toBeInstanceOf(Function);
      expect(logger.error).toBeInstanceOf(Function);
      expect(logger.warn).toBeInstanceOf(Function);
      expect(logger.debug).toBeInstanceOf(Function);
    });

    it('should create a logger with custom service name', () => {
      const logger = createLogger({ service: 'test-service' });
      expect(logger).toBeDefined();
    });

    it('should create a logger with custom log level', () => {
      const logger = createLogger({ level: 'debug' });
      expect(logger).toBeDefined();
    });

    it('should create a silent logger', () => {
      const logger = createLogger({ silent: true });
      expect(logger).toBeDefined();
      // Silent logger should not output, but methods should still exist
      expect(() => logger.info('test')).not.toThrow();
    });
  });
});
