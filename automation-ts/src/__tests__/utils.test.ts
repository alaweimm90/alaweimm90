import { getAutomationPath } from '../utils/file';

describe('File Utilities', () => {
  describe('getAutomationPath', () => {
    it('should return a path string', () => {
      const result = getAutomationPath();
      expect(typeof result).toBe('string');
    });

    it('should return automation path from env if set', () => {
      const originalEnv = process.env.AUTOMATION_PATH;
      process.env.AUTOMATION_PATH = '/custom/path';

      const result = getAutomationPath();
      expect(result).toBe('/custom/path');

      // Restore
      if (originalEnv) {
        process.env.AUTOMATION_PATH = originalEnv;
      } else {
        delete process.env.AUTOMATION_PATH;
      }
    });
  });
});
