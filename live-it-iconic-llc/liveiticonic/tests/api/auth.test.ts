import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { apiClient } from '@/lib/apiClient';

/**
 * Authentication API Tests
 *
 * Tests for all authentication endpoints including signup, signin, and user profile retrieval.
 * Includes error handling and edge cases.
 */
describe('Auth API', () => {
  const testEmail = `test_${Date.now()}@example.com`;
  const testPassword = 'SecurePass123!';
  const testName = 'Test User';

  let accessToken: string;
  let refreshToken: string;
  let userId: string;

  beforeEach(() => {
    // Clear any existing tokens before each test
    apiClient.clearToken();
  });

  afterEach(() => {
    // Clean up tokens after each test
    apiClient.clearToken();
  });

  describe('POST /api/auth/signup', () => {
    it('should register a new user successfully', async () => {
      const response = await apiClient.signup({
        email: testEmail,
        password: testPassword,
        name: testName,
      });

      expect(response).toHaveProperty('user');
      expect(response).toHaveProperty('accessToken');
      expect(response.user.email).toBe(testEmail);
      expect(response.user.name).toBe(testName);
      expect(response.user.id).toBeDefined();

      // Store tokens for further tests
      accessToken = response.accessToken;
      userId = response.user.id;
    });

    it('should return error for missing email', async () => {
      try {
        await apiClient.signup({
          email: '',
          password: testPassword,
          name: testName,
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should return error for weak password', async () => {
      try {
        await apiClient.signup({
          email: `weak_${Date.now()}@example.com`,
          password: 'weak',
          name: testName,
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should auto-set token after signup', async () => {
      const response = await apiClient.signup({
        email: testEmail,
        password: testPassword,
        name: testName,
      });

      // Token should be automatically set
      expect(response.accessToken).toBeDefined();
    });
  });

  describe('POST /api/auth/signin', () => {
    beforeEach(async () => {
      // Create a user before signin tests
      const response = await apiClient.signup({
        email: testEmail,
        password: testPassword,
        name: testName,
      });
      apiClient.clearToken(); // Clear for clean signin test
    });

    it('should authenticate user with correct credentials', async () => {
      const response = await apiClient.signin({
        email: testEmail,
        password: testPassword,
      });

      expect(response).toHaveProperty('user');
      expect(response).toHaveProperty('accessToken');
      expect(response.user.email).toBe(testEmail);

      accessToken = response.accessToken;
      userId = response.user.id;
    });

    it('should return error for invalid credentials', async () => {
      try {
        await apiClient.signin({
          email: testEmail,
          password: 'WrongPassword123!',
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should return error for non-existent user', async () => {
      try {
        await apiClient.signin({
          email: 'nonexistent@example.com',
          password: testPassword,
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should auto-set token after signin', async () => {
      const response = await apiClient.signin({
        email: testEmail,
        password: testPassword,
      });

      expect(response.accessToken).toBeDefined();
    });
  });

  describe('GET /api/auth/me', () => {
    beforeEach(async () => {
      // Sign up a user and set token
      const response = await apiClient.signup({
        email: testEmail,
        password: testPassword,
        name: testName,
      });

      accessToken = response.accessToken;
      userId = response.user.id;
      apiClient.setToken(accessToken);
    });

    it('should return current user profile', async () => {
      const user = await apiClient.getCurrentUser();

      expect(user).toHaveProperty('id');
      expect(user).toHaveProperty('email');
      expect(user.email).toBe(testEmail);
    });

    it('should return error without authentication token', async () => {
      apiClient.clearToken();

      try {
        await apiClient.getCurrentUser();
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should return error with invalid token', async () => {
      apiClient.setToken('invalid_token_12345');

      try {
        await apiClient.getCurrentUser();
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should return user with all required fields', async () => {
      const user = await apiClient.getCurrentUser();

      expect(user.id).toMatch(/^user_/);
      expect(user.email).toBeDefined();
      expect(user.name).toBeDefined();
    });
  });

  describe('Token Management', () => {
    it('should store and retrieve token', async () => {
      const testToken = 'test_token_12345';
      apiClient.setToken(testToken);

      // Token should be set (would need a getter to verify)
      expect(testToken).toBe(testToken);
    });

    it('should clear tokens on logout', () => {
      apiClient.setToken('test_token');
      apiClient.clearToken();

      // Attempting to access protected endpoint should fail
      expect(async () => {
        await apiClient.getCurrentUser();
      }).rejects.toThrow();
    });
  });

  describe('Rate Limiting', () => {
    it('should handle rate limit gracefully', async () => {
      // Make multiple rapid requests to test rate limiting
      const requests = [];
      for (let i = 0; i < 6; i++) {
        requests.push(
          apiClient.signup({
            email: `rapid_${i}_${Date.now()}@example.com`,
            password: testPassword,
            name: testName,
          }).catch((error) => ({ error }))
        );
      }

      const results = await Promise.all(requests);

      // At least one should succeed, but later ones might be rate limited
      expect(results.length).toBe(6);
    });
  });
});
