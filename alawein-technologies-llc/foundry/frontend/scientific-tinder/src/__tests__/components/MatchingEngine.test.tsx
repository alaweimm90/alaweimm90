import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor, within, fireEvent } from '@testing-library/react';
import { customRender, mockUser, mockProject } from '@/testing/utils/test-utils';
import MatchingEngine from '@/components/matching/MatchingEngine';
import type { Researcher, Match, CollaborationProfile } from '@/types';

describe('MatchingEngine', () => {
  const mockResearcher: Researcher = {
    ...mockUser(),
    expertise: ['Machine Learning', 'Quantum Computing'],
    interests: ['AI Safety', 'Cryptography'],
    publications: 25,
    hIndex: 12,
    collaborationHistory: [],
    preferredRoles: ['lead', 'contributor'],
    availability: 'full-time',
    timezone: 'UTC-5',
  };

  const mockMatches: Match[] = [
    {
      id: 'm1',
      researcher: {
        ...mockUser({ id: '2', name: 'Dr. Sarah Chen' }),
        expertise: ['Quantum Physics', 'Machine Learning'],
        interests: ['Quantum ML', 'AI Safety'],
        publications: 30,
        hIndex: 15,
      },
      matchScore: 0.92,
      commonInterests: ['AI Safety'],
      complementarySkills: ['Quantum Physics'],
      potentialProjects: ['Quantum-enhanced ML algorithms'],
    },
    {
      id: 'm2',
      researcher: {
        ...mockUser({ id: '3', name: 'Prof. James Wilson' }),
        expertise: ['Cryptography', 'Mathematics'],
        interests: ['Post-quantum cryptography'],
        publications: 45,
        hIndex: 20,
      },
      matchScore: 0.78,
      commonInterests: ['Cryptography'],
      complementarySkills: ['Mathematics'],
      potentialProjects: ['Quantum-resistant encryption'],
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Swipe Mechanics', () => {
    it('should render the current match card', () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      expect(screen.getByText('Dr. Sarah Chen')).toBeInTheDocument();
      expect(screen.getByText('92% Match')).toBeInTheDocument();
      expect(screen.getByText('Quantum Physics')).toBeInTheDocument();
      expect(screen.getByText('Machine Learning')).toBeInTheDocument();
    });

    it('should handle right swipe (accept)', async () => {
      const onAccept = vi.fn();
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          onAccept={onAccept}
        />
      );

      const card = screen.getByTestId('match-card');

      // Simulate swipe right
      fireEvent.touchStart(card, { touches: [{ clientX: 0, clientY: 0 }] });
      fireEvent.touchMove(card, { touches: [{ clientX: 200, clientY: 0 }] });
      fireEvent.touchEnd(card);

      await waitFor(() => {
        expect(onAccept).toHaveBeenCalledWith(mockMatches[0]);
        expect(screen.getByText('Prof. James Wilson')).toBeInTheDocument(); // Next card
      });
    });

    it('should handle left swipe (reject)', async () => {
      const onReject = vi.fn();
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          onReject={onReject}
        />
      );

      const card = screen.getByTestId('match-card');

      // Simulate swipe left
      fireEvent.touchStart(card, { touches: [{ clientX: 200, clientY: 0 }] });
      fireEvent.touchMove(card, { touches: [{ clientX: 0, clientY: 0 }] });
      fireEvent.touchEnd(card);

      await waitFor(() => {
        expect(onReject).toHaveBeenCalledWith(mockMatches[0]);
        expect(screen.getByText('Prof. James Wilson')).toBeInTheDocument(); // Next card
      });
    });

    it('should handle keyboard controls', async () => {
      const onAccept = vi.fn();
      const onReject = vi.fn();
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          onAccept={onAccept}
          onReject={onReject}
        />
      );

      // Press right arrow to accept
      await user.keyboard('{ArrowRight}');
      expect(onAccept).toHaveBeenCalledWith(mockMatches[0]);

      // Press left arrow to reject
      await user.keyboard('{ArrowLeft}');
      expect(onReject).toHaveBeenCalledWith(mockMatches[1]);
    });

    it('should handle button controls', async () => {
      const onAccept = vi.fn();
      const onReject = vi.fn();
      const onSuperLike = vi.fn();
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          onAccept={onAccept}
          onReject={onReject}
          onSuperLike={onSuperLike}
        />
      );

      // Click accept button
      await user.click(screen.getByRole('button', { name: /accept/i }));
      expect(onAccept).toHaveBeenCalledWith(mockMatches[0]);

      // Click super like button
      await user.click(screen.getByRole('button', { name: /super like/i }));
      expect(onSuperLike).toHaveBeenCalledWith(mockMatches[1]);
    });
  });

  describe('Match Algorithm', () => {
    it('should calculate match scores accurately', () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      const scoreDetails = screen.getByTestId('score-breakdown');
      expect(within(scoreDetails).getByText('Expertise Match: 85%')).toBeInTheDocument();
      expect(within(scoreDetails).getByText('Interest Alignment: 90%')).toBeInTheDocument();
      expect(within(scoreDetails).getByText('Collaboration Potential: 95%')).toBeInTheDocument();
    });

    it('should highlight complementary skills', () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      const complementarySection = screen.getByTestId('complementary-skills');
      expect(within(complementarySection).getByText('Quantum Physics')).toBeInTheDocument();
      expect(within(complementarySection).getByText('You bring: Machine Learning')).toBeInTheDocument();
      expect(within(complementarySection).getByText('They bring: Quantum Physics')).toBeInTheDocument();
    });

    it('should suggest collaboration opportunities', () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      const opportunities = screen.getByTestId('collaboration-opportunities');
      expect(within(opportunities).getByText('Quantum-enhanced ML algorithms')).toBeInTheDocument();
      expect(within(opportunities).getByText('Estimated Impact: High')).toBeInTheDocument();
    });
  });

  describe('Filtering and Preferences', () => {
    it('should apply expertise filters', async () => {
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      await user.click(screen.getByRole('button', { name: /filters/i }));

      const quantumFilter = screen.getByRole('checkbox', { name: /quantum physics/i });
      await user.click(quantumFilter);

      await waitFor(() => {
        const visibleCards = screen.getAllByTestId('match-card');
        expect(visibleCards).toHaveLength(1);
        expect(screen.getByText('Dr. Sarah Chen')).toBeInTheDocument();
      });
    });

    it('should apply availability filters', async () => {
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      await user.click(screen.getByRole('button', { name: /filters/i }));

      const fullTimeFilter = screen.getByRole('radio', { name: /full-time/i });
      await user.click(fullTimeFilter);

      await waitFor(() => {
        const cards = screen.getAllByTestId('match-card');
        cards.forEach(card => {
          expect(within(card).getByText(/full-time/i)).toBeInTheDocument();
        });
      });
    });

    it('should sort matches by different criteria', async () => {
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      const sortSelect = screen.getByRole('combobox', { name: /sort by/i });
      await user.selectOptions(sortSelect, 'publications');

      await waitFor(() => {
        const cards = screen.getAllByTestId('match-card');
        expect(within(cards[0]).getByText('Prof. James Wilson')).toBeInTheDocument(); // 45 publications
        expect(within(cards[1]).getByText('Dr. Sarah Chen')).toBeInTheDocument(); // 30 publications
      });
    });
  });

  describe('Match History and Analytics', () => {
    it('should track swipe history', async () => {
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          showHistory={true}
        />
      );

      // Make some swipes
      await user.click(screen.getByRole('button', { name: /accept/i }));
      await user.click(screen.getByRole('button', { name: /reject/i }));

      // Check history
      await user.click(screen.getByRole('tab', { name: /history/i }));

      const history = screen.getByTestId('match-history');
      expect(within(history).getByText('Dr. Sarah Chen - Accepted')).toBeInTheDocument();
      expect(within(history).getByText('Prof. James Wilson - Rejected')).toBeInTheDocument();
    });

    it('should display matching statistics', () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          stats={{
            totalSeen: 50,
            accepted: 15,
            rejected: 35,
            matchRate: 0.30,
            avgMatchScore: 0.75,
          }}
        />
      );

      const stats = screen.getByTestId('match-stats');
      expect(within(stats).getByText('30% Match Rate')).toBeInTheDocument();
      expect(within(stats).getByText('15 Accepted')).toBeInTheDocument();
      expect(within(stats).getByText('75% Avg Score')).toBeInTheDocument();
    });
  });

  describe('Collaboration Initiation', () => {
    it('should show mutual match notification', async () => {
      const { rerender } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      // Simulate mutual match
      rerender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          mutualMatch={{
            researcher: mockMatches[0].researcher,
            matchedAt: new Date().toISOString(),
          }}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole('dialog', { name: /it's a match!/i })).toBeInTheDocument();
        expect(screen.getByText('You and Dr. Sarah Chen matched!')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /start chatting/i })).toBeInTheDocument();
      });
    });

    it('should allow initiating collaboration after match', async () => {
      const onInitiateCollaboration = vi.fn();
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          mutualMatch={{
            researcher: mockMatches[0].researcher,
            matchedAt: new Date().toISOString(),
          }}
          onInitiateCollaboration={onInitiateCollaboration}
        />
      );

      await user.click(screen.getByRole('button', { name: /start collaboration/i }));

      // Fill project proposal
      const titleInput = screen.getByLabelText(/project title/i);
      const descInput = screen.getByLabelText(/project description/i);

      await user.type(titleInput, 'Quantum ML Research');
      await user.type(descInput, 'Exploring quantum computing for machine learning');

      await user.click(screen.getByRole('button', { name: /send proposal/i }));

      expect(onInitiateCollaboration).toHaveBeenCalledWith({
        partnerId: '2',
        projectTitle: 'Quantum ML Research',
        projectDescription: 'Exploring quantum computing for machine learning',
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels for swipe cards', () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      const card = screen.getByTestId('match-card');
      expect(card).toHaveAttribute('role', 'article');
      expect(card).toHaveAttribute('aria-label', expect.stringContaining('Dr. Sarah Chen'));
      expect(card).toHaveAttribute('aria-describedby', 'match-details-m1');
    });

    it('should announce swipe actions to screen readers', async () => {
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      await user.click(screen.getByRole('button', { name: /accept/i }));

      await waitFor(() => {
        const announcement = screen.getByRole('status');
        expect(announcement).toHaveTextContent('Accepted match with Dr. Sarah Chen');
      });
    });

    it('should support focus management during swiping', async () => {
      const { user } = customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
        />
      );

      const firstCard = screen.getByTestId('match-card');
      expect(document.activeElement).toBe(firstCard);

      await user.click(screen.getByRole('button', { name: /accept/i }));

      await waitFor(() => {
        const nextCard = screen.getByTestId('match-card');
        expect(document.activeElement).toBe(nextCard);
      });
    });
  });

  describe('Performance', () => {
    it('should efficiently handle large match queues', () => {
      const manyMatches = Array.from({ length: 100 }, (_, i) => ({
        id: `m${i}`,
        researcher: mockUser({ id: `${i}`, name: `Researcher ${i}` }),
        matchScore: Math.random(),
        commonInterests: ['Research'],
        complementarySkills: ['Skill'],
        potentialProjects: ['Project'],
      }));

      const startTime = performance.now();
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={manyMatches}
        />
      );
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100); // Should render in less than 100ms
      expect(screen.getByTestId('match-queue-size')).toHaveTextContent('100');
    });

    it('should preload next match cards', async () => {
      customRender(
        <MatchingEngine
          currentUser={mockResearcher}
          matches={mockMatches}
          preloadCount={2}
        />
      );

      // Check that next cards are preloaded but hidden
      const preloadedCards = screen.getAllByTestId(/match-card-preload/);
      expect(preloadedCards).toHaveLength(1); // Next card is preloaded
      expect(preloadedCards[0]).toHaveStyle({ visibility: 'hidden' });
    });
  });
});