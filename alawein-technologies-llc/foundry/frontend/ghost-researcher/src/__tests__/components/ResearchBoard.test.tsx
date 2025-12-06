import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import { customRender, mockUser, mockProject, MockWebSocket } from '@/testing/utils/test-utils';
import ResearchBoard from '@/components/research/ResearchBoard';
import type { ResearchProject, Collaborator } from '@/types';

// Mock WebSocket
global.WebSocket = MockWebSocket as any;

describe('ResearchBoard', () => {
  const mockProject: ResearchProject = {
    id: 'project-1',
    title: 'Quantum Computing Research',
    description: 'Exploring quantum algorithms for optimization',
    status: 'active',
    owner: mockUser(),
    collaborators: [
      mockUser({ id: '2', name: 'Alice Johnson', role: 'researcher' }),
      mockUser({ id: '3', name: 'Bob Smith', role: 'reviewer' }),
    ],
    hypotheses: [
      {
        id: 'h1',
        title: 'Quantum supremacy in optimization',
        description: 'Quantum computers can solve NP-hard problems exponentially faster',
        status: 'testing',
        evidence: [],
        createdAt: new Date().toISOString(),
      },
    ],
    papers: [],
    datasets: [],
    tags: ['quantum', 'optimization'],
    visibility: 'team',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Rendering', () => {
    it('should render the research board with project details', () => {
      const { container } = customRender(
        <ResearchBoard project={mockProject} currentUser={mockUser()} />
      );

      expect(screen.getByText('Quantum Computing Research')).toBeInTheDocument();
      expect(screen.getByText('Exploring quantum algorithms for optimization')).toBeInTheDocument();
      expect(screen.getByText('Alice Johnson')).toBeInTheDocument();
      expect(screen.getByText('Bob Smith')).toBeInTheDocument();
    });

    it('should display hypothesis cards', () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      const hypothesisCard = screen.getByText('Quantum supremacy in optimization');
      expect(hypothesisCard).toBeInTheDocument();
      expect(screen.getByText('testing')).toBeInTheDocument();
    });

    it('should show collaboration status indicators', () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      const statusIndicators = screen.getAllByTestId('collaboration-status');
      expect(statusIndicators).toHaveLength(3); // Owner + 2 collaborators
    });
  });

  describe('Real-time Collaboration', () => {
    it('should establish WebSocket connection on mount', async () => {
      const { container } = customRender(
        <ResearchBoard project={mockProject} currentUser={mockUser()} />
      );

      await waitFor(() => {
        const ws = (global as any).mockWebSocketInstance;
        expect(ws).toBeDefined();
        expect(ws.readyState).toBe(WebSocket.OPEN);
      });
    });

    it('should handle real-time hypothesis updates', async () => {
      const { container, user } = customRender(
        <ResearchBoard project={mockProject} currentUser={mockUser()} />
      );

      // Simulate incoming WebSocket message
      const ws = (global as any).mockWebSocketInstance as MockWebSocket;
      ws.simulateMessage({
        type: 'hypothesis_update',
        data: {
          id: 'h1',
          title: 'Updated Quantum Hypothesis',
          status: 'validated',
        },
      });

      await waitFor(() => {
        expect(screen.getByText('Updated Quantum Hypothesis')).toBeInTheDocument();
        expect(screen.getByText('validated')).toBeInTheDocument();
      });
    });

    it('should show collaborator presence in real-time', async () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      const ws = (global as any).mockWebSocketInstance as MockWebSocket;
      ws.simulateMessage({
        type: 'user_joined',
        data: {
          user: mockUser({ id: '4', name: 'Carol White' }),
        },
      });

      await waitFor(() => {
        expect(screen.getByText('Carol White')).toBeInTheDocument();
        expect(screen.getByTestId('online-indicator-4')).toHaveClass('online');
      });
    });

    it('should handle WebSocket disconnection gracefully', async () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      const ws = (global as any).mockWebSocketInstance as MockWebSocket;
      ws.simulateError();

      await waitFor(() => {
        expect(screen.getByText('Connection lost. Attempting to reconnect...')).toBeInTheDocument();
      });

      // Simulate reconnection
      ws.readyState = WebSocket.OPEN;
      ws.simulateMessage({ type: 'connected' });

      await waitFor(() => {
        expect(screen.queryByText('Connection lost')).not.toBeInTheDocument();
      });
    });
  });

  describe('Interactions', () => {
    it('should allow adding new hypothesis', async () => {
      const onAddHypothesis = vi.fn();
      const { user } = customRender(
        <ResearchBoard
          project={mockProject}
          currentUser={mockUser()}
          onAddHypothesis={onAddHypothesis}
        />
      );

      const addButton = screen.getByRole('button', { name: /add hypothesis/i });
      await user.click(addButton);

      // Fill in the form
      const titleInput = screen.getByLabelText(/hypothesis title/i);
      const descInput = screen.getByLabelText(/description/i);

      await user.type(titleInput, 'New Quantum Theory');
      await user.type(descInput, 'A revolutionary approach to quantum computing');

      const submitButton = screen.getByRole('button', { name: /create hypothesis/i });
      await user.click(submitButton);

      expect(onAddHypothesis).toHaveBeenCalledWith({
        title: 'New Quantum Theory',
        description: 'A revolutionary approach to quantum computing',
        status: 'proposed',
      });
    });

    it('should handle paper upload', async () => {
      const onUploadPaper = vi.fn();
      const { user } = customRender(
        <ResearchBoard
          project={mockProject}
          currentUser={mockUser()}
          onUploadPaper={onUploadPaper}
        />
      );

      const file = new File(['paper content'], 'research.pdf', { type: 'application/pdf' });
      const input = screen.getByLabelText(/upload paper/i) as HTMLInputElement;

      await user.upload(input, file);

      expect(onUploadPaper).toHaveBeenCalledWith(expect.objectContaining({
        file,
        projectId: 'project-1',
      }));
    });

    it('should enable collaboration features for team members', async () => {
      const currentUser = mockUser({ id: '2' }); // Alice Johnson
      const { user } = customRender(
        <ResearchBoard project={mockProject} currentUser={currentUser} />
      );

      // Should see collaboration controls
      expect(screen.getByRole('button', { name: /invite collaborator/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /start discussion/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /share findings/i })).toBeInTheDocument();
    });

    it('should restrict actions for non-collaborators', () => {
      const nonCollaborator = mockUser({ id: '99', name: 'External User' });
      customRender(<ResearchBoard project={mockProject} currentUser={nonCollaborator} />);

      // Should not see sensitive controls
      expect(screen.queryByRole('button', { name: /add hypothesis/i })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /edit project/i })).not.toBeInTheDocument();
      expect(screen.getByText(/view only mode/i)).toBeInTheDocument();
    });
  });

  describe('Analytics and Insights', () => {
    it('should display research progress metrics', () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      const progressSection = screen.getByTestId('research-progress');
      expect(within(progressSection).getByText(/1 hypothesis/i)).toBeInTheDocument();
      expect(within(progressSection).getByText(/0 papers/i)).toBeInTheDocument();
      expect(within(progressSection).getByText(/3 collaborators/i)).toBeInTheDocument();
    });

    it('should show hypothesis validation status', () => {
      const projectWithValidation = {
        ...mockProject,
        hypotheses: [
          { ...mockProject.hypotheses[0], status: 'validated', confidence: 0.95 },
          { id: 'h2', title: 'Alternative approach', status: 'rejected', confidence: 0.2 },
          { id: 'h3', title: 'New theory', status: 'testing', confidence: 0.6 },
        ],
      };

      customRender(<ResearchBoard project={projectWithValidation} currentUser={mockUser()} />);

      const validationChart = screen.getByTestId('validation-chart');
      expect(validationChart).toBeInTheDocument();
      expect(within(validationChart).getByText('33%')).toBeInTheDocument(); // Validated
      expect(within(validationChart).getByText('33%')).toBeInTheDocument(); // Testing
      expect(within(validationChart).getByText('33%')).toBeInTheDocument(); // Rejected
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      expect(screen.getByRole('main', { name: /research board/i })).toBeInTheDocument();
      expect(screen.getByRole('region', { name: /collaborators/i })).toBeInTheDocument();
      expect(screen.getByRole('region', { name: /hypotheses/i })).toBeInTheDocument();
    });

    it('should support keyboard navigation', async () => {
      const { user } = customRender(
        <ResearchBoard project={mockProject} currentUser={mockUser()} />
      );

      // Tab through interactive elements
      await user.tab();
      expect(document.activeElement).toHaveAttribute('role', 'button');

      await user.tab();
      expect(document.activeElement).toHaveAttribute('data-testid', 'hypothesis-card');

      // Activate with Enter key
      await user.keyboard('{Enter}');
      expect(screen.getByRole('dialog')).toBeInTheDocument();

      // Close with Escape key
      await user.keyboard('{Escape}');
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('should announce status changes to screen readers', async () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      const ws = (global as any).mockWebSocketInstance as MockWebSocket;
      ws.simulateMessage({
        type: 'hypothesis_validated',
        data: { id: 'h1', title: 'Quantum supremacy proven' },
      });

      await waitFor(() => {
        const announcement = screen.getByRole('status');
        expect(announcement).toHaveTextContent('Hypothesis "Quantum supremacy proven" has been validated');
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      const { rerender } = customRender(
        <ResearchBoard project={mockProject} currentUser={mockUser()} error={null} />
      );

      const error = new Error('Failed to load project data');
      rerender(<ResearchBoard project={mockProject} currentUser={mockUser()} error={error} />);

      expect(screen.getByRole('alert')).toHaveTextContent('Failed to load project data');
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('should show offline mode when connection is lost', async () => {
      customRender(<ResearchBoard project={mockProject} currentUser={mockUser()} />);

      // Simulate going offline
      window.dispatchEvent(new Event('offline'));

      await waitFor(() => {
        expect(screen.getByText(/working offline/i)).toBeInTheDocument();
        expect(screen.getByText(/changes will sync when reconnected/i)).toBeInTheDocument();
      });

      // Simulate coming back online
      window.dispatchEvent(new Event('online'));

      await waitFor(() => {
        expect(screen.queryByText(/working offline/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Performance', () => {
    it('should render large hypothesis lists efficiently', () => {
      const manyHypotheses = Array.from({ length: 100 }, (_, i) => ({
        id: `h${i}`,
        title: `Hypothesis ${i}`,
        description: `Description for hypothesis ${i}`,
        status: i % 3 === 0 ? 'validated' : i % 3 === 1 ? 'testing' : 'proposed',
        createdAt: new Date().toISOString(),
      }));

      const largeProject = { ...mockProject, hypotheses: manyHypotheses };

      const startTime = performance.now();
      customRender(<ResearchBoard project={largeProject} currentUser={mockUser()} />);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100); // Should render in less than 100ms
      expect(screen.getByTestId('hypothesis-list')).toHaveAttribute('data-virtualized', 'true');
    });

    it('should debounce search input', async () => {
      const onSearch = vi.fn();
      const { user } = customRender(
        <ResearchBoard project={mockProject} currentUser={mockUser()} onSearch={onSearch} />
      );

      const searchInput = screen.getByPlaceholderText(/search hypotheses/i);
      await user.type(searchInput, 'quantum');

      // Should not call immediately
      expect(onSearch).not.toHaveBeenCalled();

      // Wait for debounce
      await waitFor(() => {
        expect(onSearch).toHaveBeenCalledWith('quantum');
      }, { timeout: 500 });

      expect(onSearch).toHaveBeenCalledTimes(1);
    });
  });
});