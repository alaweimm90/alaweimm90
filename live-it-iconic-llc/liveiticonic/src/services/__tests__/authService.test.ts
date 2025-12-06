/**
 * Auth Service Tests
 *
 * Comprehensive test suite for authentication service
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { authService, type User } from '../authService';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('AuthService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('signUp', () => {
    it('should create new user account successfully', async () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'newuser@example.com',
        name: 'New User',
        createdAt: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: mockUser,
          token: 'auth-token-123',
        }),
      });

      const result = await authService.signUp({
        email: 'newuser@example.com',
        password: 'SecurePass123!',
        name: 'New User',
      });

      expect(result.user).toEqual(mockUser);
      expect(result.token).toBe('auth-token-123');
      expect(localStorageMock.getItem('auth_token')).toBe('auth-token-123');

      expect(mockFetch).toHaveBeenCalledWith('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'newuser@example.com',
          password: 'SecurePass123!',
          name: 'New User',
        }),
      });
    });

    it('should create account without name (optional)', async () => {
      const mockUser: User = {
        id: 'user-456',
        email: 'noname@example.com',
        createdAt: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: mockUser,
          token: 'token-456',
        }),
      });

      const result = await authService.signUp({
        email: 'noname@example.com',
        password: 'Password123!',
      });

      expect(result.user.name).toBeUndefined();
      expect(result.user.email).toBe('noname@example.com');
    });

    it('should throw error if email already exists', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 409,
        json: async () => ({
          message: 'Email already exists',
        }),
      });

      await expect(
        authService.signUp({
          email: 'existing@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow('Email already exists');
    });

    it('should throw error if password is weak', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Password must be at least 8 characters',
        }),
      });

      await expect(
        authService.signUp({
          email: 'user@example.com',
          password: 'weak',
        })
      ).rejects.toThrow('Password must be at least 8 characters');
    });

    it('should throw error if email format is invalid', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          message: 'Invalid email format',
        }),
      });

      await expect(
        authService.signUp({
          email: 'invalid-email',
          password: 'Password123!',
        })
      ).rejects.toThrow('Invalid email format');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      await expect(
        authService.signUp({
          email: 'user@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow('Network error');
    });

    it('should handle generic API failure with fallback error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({}), // No error message
      });

      await expect(
        authService.signUp({
          email: 'user@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow('Sign up failed');
    });

    it('should not store token if API returns no token', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: {
            id: 'user-789',
            email: 'notoken@example.com',
          },
          // token missing
        }),
      });

      const result = await authService.signUp({
        email: 'notoken@example.com',
        password: 'Password123!',
      });

      expect(result.user.id).toBe('user-789');
      expect(localStorageMock.getItem('auth_token')).toBeNull();
    });
  });

  describe('signIn', () => {
    it('should authenticate existing user successfully', async () => {
      const mockUser: User = {
        id: 'user-abc',
        email: 'existing@example.com',
        name: 'Existing User',
        createdAt: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: mockUser,
          token: 'signin-token-abc',
        }),
      });

      const result = await authService.signIn({
        email: 'existing@example.com',
        password: 'Password123!',
      });

      expect(result.user).toEqual(mockUser);
      expect(result.token).toBe('signin-token-abc');
      expect(localStorageMock.getItem('auth_token')).toBe('signin-token-abc');

      expect(mockFetch).toHaveBeenCalledWith('/api/auth/signin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'existing@example.com',
          password: 'Password123!',
        }),
      });
    });

    it('should throw error for invalid credentials', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 401,
        json: async () => ({
          message: 'Invalid email or password',
        }),
      });

      await expect(
        authService.signIn({
          email: 'user@example.com',
          password: 'WrongPassword',
        })
      ).rejects.toThrow('Invalid email or password');
    });

    it('should throw error if user not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          message: 'User not found',
        }),
      });

      await expect(
        authService.signIn({
          email: 'nonexistent@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow('User not found');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network timeout'));

      await expect(
        authService.signIn({
          email: 'user@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow('Network timeout');
    });

    it('should handle generic sign in failure', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({}),
      });

      await expect(
        authService.signIn({
          email: 'user@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow('Sign in failed');
    });

    it('should replace existing token on new sign in', async () => {
      // Set old token
      localStorageMock.setItem('auth_token', 'old-token');

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: {
            id: 'user-xyz',
            email: 'user@example.com',
          },
          token: 'new-token-xyz',
        }),
      });

      await authService.signIn({
        email: 'user@example.com',
        password: 'Password123!',
      });

      expect(localStorageMock.getItem('auth_token')).toBe('new-token-xyz');
    });
  });

  describe('signOut', () => {
    it('should sign out user successfully', async () => {
      // Set token
      localStorageMock.setItem('auth_token', 'active-token');

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      await authService.signOut();

      expect(localStorageMock.getItem('auth_token')).toBeNull();
      expect(mockFetch).toHaveBeenCalledWith('/api/auth/signout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should remove token even if API call fails', async () => {
      localStorageMock.setItem('auth_token', 'active-token');

      mockFetch.mockRejectedValue(new Error('API error'));

      await authService.signOut();

      // Token should still be removed
      expect(localStorageMock.getItem('auth_token')).toBeNull();
    });

    it('should not throw if localStorage is empty', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      await expect(authService.signOut()).resolves.toBeUndefined();
    });

    it('should handle network errors gracefully', async () => {
      localStorageMock.setItem('auth_token', 'token-to-remove');

      mockFetch.mockRejectedValue(new Error('Network error'));

      // Should not throw
      await expect(authService.signOut()).resolves.toBeUndefined();

      // Token should be removed
      expect(localStorageMock.getItem('auth_token')).toBeNull();
    });
  });

  describe('getCurrentUser', () => {
    it('should return user when valid token exists', async () => {
      localStorageMock.setItem('auth_token', 'valid-token-123');

      const mockUser: User = {
        id: 'user-current',
        email: 'current@example.com',
        name: 'Current User',
        createdAt: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: mockUser,
        }),
      });

      const result = await authService.getCurrentUser();

      expect(result).toEqual(mockUser);
      expect(mockFetch).toHaveBeenCalledWith('/api/auth/me', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer valid-token-123',
        },
      });
    });

    it('should return null when no token exists', async () => {
      // No token in localStorage
      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('should return null when token is invalid', async () => {
      localStorageMock.setItem('auth_token', 'invalid-token');

      mockFetch.mockResolvedValue({
        ok: false,
        status: 401,
        json: async () => ({
          message: 'Invalid token',
        }),
      });

      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
    });

    it('should return null when token is expired', async () => {
      localStorageMock.setItem('auth_token', 'expired-token');

      mockFetch.mockResolvedValue({
        ok: false,
        status: 401,
        json: async () => ({
          message: 'Token expired',
        }),
      });

      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
    });

    it('should handle network errors gracefully', async () => {
      localStorageMock.setItem('auth_token', 'some-token');

      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
    });

    it('should return null if API response has no user', async () => {
      localStorageMock.setItem('auth_token', 'token-123');

      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({}), // No user property
      });

      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
    });

    it('should handle API errors gracefully', async () => {
      localStorageMock.setItem('auth_token', 'token-456');

      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          message: 'Internal server error',
        }),
      });

      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
    });
  });

  describe('token storage', () => {
    it('should persist token across service calls', async () => {
      // Sign in
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: {
            id: 'user-persistent',
            email: 'persistent@example.com',
          },
          token: 'persistent-token-789',
        }),
      });

      await authService.signIn({
        email: 'persistent@example.com',
        password: 'Password123!',
      });

      expect(localStorageMock.getItem('auth_token')).toBe('persistent-token-789');

      // Get current user (should use stored token)
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          user: {
            id: 'user-persistent',
            email: 'persistent@example.com',
          },
        }),
      });

      const user = await authService.getCurrentUser();

      expect(user?.id).toBe('user-persistent');
      expect(mockFetch).toHaveBeenLastCalledWith(
        '/api/auth/me',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer persistent-token-789',
          }),
        })
      );
    });
  });

  describe('error handling', () => {
    it('should console.error on sign up failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('API down'));

      await expect(
        authService.signUp({
          email: 'test@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Sign up failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should console.error on sign in failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      mockFetch.mockRejectedValue(new Error('API down'));

      await expect(
        authService.signIn({
          email: 'test@example.com',
          password: 'Password123!',
        })
      ).rejects.toThrow();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Sign in failed:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });

    it('should console.error on getCurrentUser failure', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      localStorageMock.setItem('auth_token', 'token');
      mockFetch.mockRejectedValue(new Error('Network error'));

      const result = await authService.getCurrentUser();

      expect(result).toBeNull();
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to get current user:',
        expect.any(Error)
      );

      consoleErrorSpy.mockRestore();
    });
  });
});
