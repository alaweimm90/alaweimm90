/**
 * Authentication Service
 *
 * Handles user authentication, session management, and token storage.
 * Provides sign-up, sign-in, sign-out, and user state management functionality.
 * Uses Supabase Auth for backend authentication and localStorage for token persistence.
 *
 * @module authService
 *
 * @example
 * import { authService } from '@/services/authService';
 *
 * // Sign up new user
 * const { user, token } = await authService.signUp({
 *   email: 'user@example.com',
 *   password: 'securePassword123',
 *   name: 'John Doe'
 * });
 *
 * // Sign in existing user
 * const auth = await authService.signIn({
 *   email: 'user@example.com',
 *   password: 'securePassword123'
 * });
 *
 * // Get current user
 * const user = await authService.getCurrentUser();
 *
 * // Sign out
 * await authService.signOut();
 */

/**
 * User interface
 *
 * Represents an authenticated user in the system
 *
 * @interface User
 * @property {string} id - Unique user identifier
 * @property {string} email - User email address
 * @property {string} [name] - Optional user display name
 * @property {string} [createdAt] - Account creation timestamp
 */
export interface User {
  id: string;
  email: string;
  name?: string;
  createdAt?: string;
}

/**
 * Sign up parameters
 *
 * @interface SignUpParams
 * @property {string} email - Email address for new account
 * @property {string} password - Account password (min 8 characters)
 * @property {string} [name] - Optional display name
 */
interface SignUpParams {
  email: string;
  password: string;
  name?: string;
}

/**
 * Sign in parameters
 *
 * @interface SignInParams
 * @property {string} email - User email address
 * @property {string} password - User password
 */
interface SignInParams {
  email: string;
  password: string;
}

/**
 * Authentication Service
 *
 * @object authService
 *
 * @property {Function} signUp - Register new user account
 * @property {Function} signIn - Authenticate existing user
 * @property {Function} signOut - End user session
 * @property {Function} getCurrentUser - Get currently logged-in user
 */
export const authService = {
  /**
   * Sign up new user account
   *
   * Creates a new user account with email and password authentication.
   * Automatically stores authentication token in localStorage for persistence.
   *
   * @async
   * @param {SignUpParams} params - Sign-up credentials
   * @param {string} params.email - User email address
   * @param {string} params.password - Account password (minimum 8 characters recommended)
   * @param {string} [params.name] - Optional display name or full name
   *
   * @returns {Promise<{user: User, token: string}>} Newly created user and auth token
   *
   * @throws {Error} If email already exists, password is weak, or API fails
   * @throws {Error} If email format is invalid
   * @throws {Error} If network error occurs
   *
   * @example
   * try {
   *   const { user, token } = await authService.signUp({
   *     email: 'newuser@example.com',
   *     password: 'SecurePass123!',
   *     name: 'Jane Smith'
   *   });
   *   console.log('Account created:', user.email);
   * } catch (error) {
   *   console.error('Sign up failed:', error.message);
   * }
   *
   * @see {@link signIn} for existing user authentication
   */
  async signUp(params: SignUpParams): Promise<{ user: User; token: string }> {
    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: params.email,
          password: params.password,
          name: params.name,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Sign up failed');
      }

      const data = await response.json();
      // Store token in localStorage
      if (data.token) {
        localStorage.setItem('auth_token', data.token);
      }
      return data;
    } catch (error) {
      console.error('Sign up failed:', error);
      throw error;
    }
  },

  /**
   * Sign in existing user
   *
   * Authenticates user with email and password credentials.
   * Stores authentication token in localStorage for session persistence.
   *
   * @async
   * @param {SignInParams} params - Sign-in credentials
   * @param {string} params.email - Registered user email
   * @param {string} params.password - User password
   *
   * @returns {Promise<{user: User, token: string}>} Authenticated user and auth token
   *
   * @throws {Error} If credentials are invalid
   * @throws {Error} If user account not found
   * @throws {Error} If network error occurs
   *
   * @example
   * try {
   *   const { user, token } = await authService.signIn({
   *     email: 'user@example.com',
   *     password: 'password123'
   *   });
   *   console.log('Successfully logged in:', user.name);
   * } catch (error) {
   *   if (error.message.includes('Invalid')) {
   *     console.error('Invalid credentials');
   *   }
   * }
   *
   * @see {@link signUp} for new user registration
   * @see {@link getCurrentUser} to get authenticated user info
   */
  async signIn(params: SignInParams): Promise<{ user: User; token: string }> {
    try {
      const response = await fetch('/api/auth/signin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: params.email,
          password: params.password,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Sign in failed');
      }

      const data = await response.json();
      // Store token in localStorage
      if (data.token) {
        localStorage.setItem('auth_token', data.token);
      }
      return data;
    } catch (error) {
      console.error('Sign in failed:', error);
      throw error;
    }
  },

  /**
   * Sign out current user
   *
   * Ends the user session by invalidating the authentication token
   * and removing it from localStorage. Safe to call even if not logged in.
   *
   * @async
   * @returns {Promise<void>} Completes when sign out is done
   *
   * @throws {Error} Only if localStorage is unavailable (rare)
   *
   * @example
   * // Sign out user
   * await authService.signOut();
   * console.log('User logged out');
   *
   * // Token is automatically removed from localStorage
   * // User is redirected to login page in your app
   *
   * @see {@link signIn} to sign in again
   * @see {@link getCurrentUser} returns null after sign out
   */
  async signOut(): Promise<void> {
    try {
      await fetch('/api/auth/signout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      // Remove token from localStorage
      localStorage.removeItem('auth_token');
    } catch (error) {
      console.error('Sign out failed:', error);
      // Still remove token even if API call fails
      localStorage.removeItem('auth_token');
    }
  },

  /**
   * Get currently authenticated user
   *
   * Retrieves the current user information if a valid authentication token exists.
   * Returns null if user is not authenticated or token has expired.
   * Useful for checking authentication state and retrieving user info after page reload.
   *
   * @async
   * @returns {Promise<User|null>} Current user object if authenticated, null otherwise
   *
   * @throws {Error} Only in rare network conditions
   *
   * @example
   * // Check if user is logged in
   * const user = await authService.getCurrentUser();
   * if (user) {
   *   console.log('Logged in as:', user.email);
   *   console.log('User name:', user.name);
   * } else {
   *   console.log('Not authenticated');
   *   // Redirect to login page
   * }
   *
   * @example
   * // Initialize app with user info
   * const user = await authService.getCurrentUser();
   * if (user) {
   *   setCurrentUser(user);
   *   // Load user preferences, cart, etc.
   * }
   *
   * @see {@link signIn} to authenticate user
   * @see {@link signOut} to end session
   */
  async getCurrentUser(): Promise<User | null> {
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        return null;
      }

      const response = await fetch('/api/auth/me', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        return null;
      }

      const data = await response.json();
      return data.user || null;
    } catch (error) {
      console.error('Failed to get current user:', error);
      return null;
    }
  },
};
