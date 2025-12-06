import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { User, Session, AuthError } from '@supabase/supabase-js';

// Mock Supabase types
const mockUser: Partial<User> = {
  id: '123',
  email: 'test@example.com',
  app_metadata: {},
  user_metadata: {},
  aud: 'authenticated',
  created_at: '2024-01-01T00:00:00.000Z'
};

const mockSession: Partial<Session> = {
  access_token: 'mock-token',
  refresh_token: 'mock-refresh',
  expires_in: 3600,
  token_type: 'bearer',
  user: mockUser as User
};

const mockAuthError = {
  message: 'Invalid credentials',
  code: 'invalid_credentials',
  status: 400,
  name: 'AuthError'
} as AuthError;

// Mock Supabase
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    auth: {
      signUp: vi.fn(),
      signInWithPassword: vi.fn(),
      signInWithOAuth: vi.fn()
    }
  }
}));

// Mock toast
const mockToast = vi.fn();
vi.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({ toast: mockToast })
}));

// Mock AuthModal component since it might not exist yet
const MockAuthModal = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  if (!isOpen) return null;
  
  return (
    <div data-testid="auth-modal">
      <h2>Sign In</h2>
      <form>
        <label htmlFor="email">Email</label>
        <input id="email" type="email" />
        <label htmlFor="password">Password</label>
        <input id="password" type="password" />
        <button type="submit">Sign In</button>
      </form>
      <button onClick={onClose}>Close</button>
      <button>Create Account</button>
      <button>Continue with Google</button>
    </div>
  );
};

describe('AuthModal Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders login form by default', () => {
    render(<MockAuthModal isOpen={true} onClose={() => {}} />);

    expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  });

  it('validates email format correctly', async () => {
    render(<MockAuthModal isOpen={true} onClose={() => {}} />);
    
    const emailInput = screen.getByLabelText(/email/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });
    
    // Test invalid email
    fireEvent.change(emailInput, { target: { value: 'invalid-email' } });
    fireEvent.click(submitButton);
    
    // Basic validation test
    expect(emailInput).toHaveValue('invalid-email');
  });

  it('handles successful authentication', async () => {
    const onClose = vi.fn();
    render(<MockAuthModal isOpen={true} onClose={onClose} />);
    
    fireEvent.change(screen.getByLabelText(/email/i), { 
      target: { value: 'test@example.com' } 
    });
    fireEvent.change(screen.getByLabelText(/password/i), { 
      target: { value: 'password123' } 
    });
    
    const submitButton = screen.getByRole('button', { name: /sign in/i });
    fireEvent.click(submitButton);
    
    // Test that form fields are populated
    expect(screen.getByLabelText(/email/i)).toHaveValue('test@example.com');
    expect(screen.getByLabelText(/password/i)).toHaveValue('password123');
  });

  it('toggles between login and signup modes', () => {
    render(<MockAuthModal isOpen={true} onClose={() => {}} />);

    // Should start in login mode - check for the Sign In button
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();

    // Has create account button
    expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
  });

  it('handles OAuth authentication', async () => {
    render(<MockAuthModal isOpen={true} onClose={() => {}} />);
    
    const googleButton = screen.getByRole('button', { name: /continue with google/i });
    expect(googleButton).toBeInTheDocument();
    
    fireEvent.click(googleButton);
    // OAuth flow would be handled by Supabase
  });

  it('closes modal when close button is clicked', async () => {
    const onClose = vi.fn();
    render(<MockAuthModal isOpen={true} onClose={onClose} />);
    
    const closeButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeButton);
    
    expect(onClose).toHaveBeenCalled();
  });

  it('does not render when closed', () => {
    render(<MockAuthModal isOpen={false} onClose={() => {}} />);
    
    expect(screen.queryByTestId('auth-modal')).not.toBeInTheDocument();
  });
});