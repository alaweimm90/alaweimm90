import { describe, it, expect } from 'vitest';

// Example unit test for the testing framework
describe('Testing Framework Setup', () => {
  it('should have testing framework properly configured', () => {
    expect(true).toBe(true);
  });

  it('should support TypeScript', () => {
    const testValue: string = 'TypeScript works';
    expect(testValue).toBe('TypeScript works');
  });

  it('should support async/await', async () => {
    const asyncTest = async (): Promise<string> => {
      return 'Async support works';
    };

    const result = await asyncTest();
    expect(result).toBe('Async support works');
  });
});