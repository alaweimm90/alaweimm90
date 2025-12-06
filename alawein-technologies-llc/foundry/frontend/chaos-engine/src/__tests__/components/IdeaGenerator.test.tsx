import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import { customRender, mockUser, mockIdea } from '@/testing/utils/test-utils';
import IdeaGenerator from '@/components/generator/IdeaGenerator';
import type { Domain, Idea, GenerationConfig } from '@/types';

describe('IdeaGenerator', () => {
  const mockDomains: Domain[] = [
    { id: 'd1', name: 'Artificial Intelligence', description: 'Machine learning and AI', trends: ['LLMs', 'Computer Vision'] },
    { id: 'd2', name: 'Healthcare', description: 'Medical and health tech', trends: ['Telemedicine', 'Genomics'] },
    { id: 'd3', name: 'Finance', description: 'Financial technology', trends: ['DeFi', 'Digital Banking'] },
    { id: 'd4', name: 'Space Technology', description: 'Space exploration', trends: ['Satellites', 'Space Tourism'] },
  ];

  const mockConfig: GenerationConfig = {
    domainCount: 2,
    chaosLevel: 0.7,
    innovationBias: 0.8,
    feasibilityThreshold: 0.5,
    targetAudience: 'startup',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Idea Generation Process', () => {
    it('should display domain selection interface', () => {
      customRender(<IdeaGenerator domains={mockDomains} config={mockConfig} />);

      expect(screen.getByText('Select Domains to Collide')).toBeInTheDocument();
      mockDomains.forEach(domain => {
        expect(screen.getByText(domain.name)).toBeInTheDocument();
      });
    });

    it('should allow selecting multiple domains', async () => {
      const { user } = customRender(<IdeaGenerator domains={mockDomains} config={mockConfig} />);

      const aiCheckbox = screen.getByRole('checkbox', { name: /artificial intelligence/i });
      const healthCheckbox = screen.getByRole('checkbox', { name: /healthcare/i });

      await user.click(aiCheckbox);
      await user.click(healthCheckbox);

      expect(aiCheckbox).toBeChecked();
      expect(healthCheckbox).toBeChecked();
      expect(screen.getByText('2 domains selected')).toBeInTheDocument();
    });

    it('should generate ideas on collision', async () => {
      const onGenerate = vi.fn().mockResolvedValue([
        {
          id: 'idea1',
          title: 'AI-Powered Diagnostic Platform',
          description: 'Using machine learning for medical diagnosis',
          domains: ['AI', 'Healthcare'],
          score: 92,
          potential: 'high',
          feasibility: 'medium',
        },
      ]);

      const { user } = customRender(
        <IdeaGenerator domains={mockDomains} config={mockConfig} onGenerate={onGenerate} />
      );

      // Select domains
      await user.click(screen.getByRole('checkbox', { name: /artificial intelligence/i }));
      await user.click(screen.getByRole('checkbox', { name: /healthcare/i }));

      // Click generate
      await user.click(screen.getByRole('button', { name: /collide domains/i }));

      await waitFor(() => {
        expect(screen.getByText('AI-Powered Diagnostic Platform')).toBeInTheDocument();
        expect(screen.getByText('Score: 92')).toBeInTheDocument();
      });

      expect(onGenerate).toHaveBeenCalledWith({
        domains: expect.arrayContaining(['d1', 'd2']),
        config: mockConfig,
      });
    });

    it('should show generation animation during processing', async () => {
      const onGenerate = vi.fn().mockImplementation(() =>
        new Promise(resolve => setTimeout(() => resolve([mockIdea()]), 1000))
      );

      const { user } = customRender(
        <IdeaGenerator domains={mockDomains} config={mockConfig} onGenerate={onGenerate} />
      );

      await user.click(screen.getByRole('checkbox', { name: /artificial intelligence/i }));
      await user.click(screen.getByRole('checkbox', { name: /healthcare/i }));
      await user.click(screen.getByRole('button', { name: /collide domains/i }));

      // Check for loading state
      expect(screen.getByTestId('collision-animation')).toBeInTheDocument();
      expect(screen.getByText(/generating ideas/i)).toBeInTheDocument();

      await waitFor(() => {
        expect(screen.queryByTestId('collision-animation')).not.toBeInTheDocument();
      }, { timeout: 2000 });
    });
  });

  describe('Chaos Level Controls', () => {
    it('should display chaos level slider', () => {
      customRender(<IdeaGenerator domains={mockDomains} config={mockConfig} />);

      const slider = screen.getByRole('slider', { name: /chaos level/i });
      expect(slider).toBeInTheDocument();
      expect(slider).toHaveAttribute('value', '70'); // 0.7 * 100
    });

    it('should update generation based on chaos level', async () => {
      const onGenerate = vi.fn();
      const { user } = customRender(
        <IdeaGenerator domains={mockDomains} config={mockConfig} onGenerate={onGenerate} />
      );

      const slider = screen.getByRole('slider', { name: /chaos level/i });
      await user.clear(slider);
      await user.type(slider, '90');

      // Select domains and generate
      await user.click(screen.getByRole('checkbox', { name: /artificial intelligence/i }));
      await user.click(screen.getByRole('checkbox', { name: /space technology/i }));
      await user.click(screen.getByRole('button', { name: /collide domains/i }));

      expect(onGenerate).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({ chaosLevel: 0.9 }),
        })
      );
    });

    it('should show chaos level descriptions', () => {
      customRender(<IdeaGenerator domains={mockDomains} config={mockConfig} />);

      expect(screen.getByText(/high chaos/i)).toBeInTheDocument();
      expect(screen.getByText(/wild, unexpected combinations/i)).toBeInTheDocument();
    });
  });

  describe('Idea Refinement', () => {
    it('should allow refining generated ideas', async () => {
      const generatedIdea = {
        id: 'idea1',
        title: 'Space Finance Platform',
        description: 'Blockchain for space commerce',
        domains: ['Finance', 'Space Technology'],
        score: 85,
        potential: 'high',
        feasibility: 'low',
      };

      const onRefine = vi.fn().mockResolvedValue({
        ...generatedIdea,
        title: 'Orbital Asset Trading Platform',
        description: 'Decentralized marketplace for space-based assets',
        score: 90,
        feasibility: 'medium',
      });

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[generatedIdea]}
          onRefine={onRefine}
        />
      );

      const refineButton = screen.getByRole('button', { name: /refine idea/i });
      await user.click(refineButton);

      // Select refinement options
      const feasibilityCheckbox = screen.getByRole('checkbox', { name: /improve feasibility/i });
      await user.click(feasibilityCheckbox);

      await user.click(screen.getByRole('button', { name: /apply refinements/i }));

      await waitFor(() => {
        expect(screen.getByText('Orbital Asset Trading Platform')).toBeInTheDocument();
        expect(screen.getByText('Score: 90')).toBeInTheDocument();
      });
    });

    it('should suggest refinement strategies', async () => {
      const generatedIdea = {
        ...mockIdea(),
        feasibility: 'low',
        potential: 'high',
      };

      customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[generatedIdea]}
        />
      );

      const suggestions = screen.getByTestId('refinement-suggestions');
      expect(within(suggestions).getByText(/consider partnerships/i)).toBeInTheDocument();
      expect(within(suggestions).getByText(/phase implementation/i)).toBeInTheDocument();
      expect(within(suggestions).getByText(/reduce technical complexity/i)).toBeInTheDocument();
    });
  });

  describe('Idea Scoring and Analysis', () => {
    it('should display comprehensive scoring breakdown', () => {
      const idea = {
        ...mockIdea(),
        score: 88,
        scoreBreakdown: {
          innovation: 95,
          feasibility: 70,
          marketPotential: 90,
          uniqueness: 97,
        },
      };

      customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[idea]}
        />
      );

      const scoreCard = screen.getByTestId('score-breakdown');
      expect(within(scoreCard).getByText('Innovation: 95')).toBeInTheDocument();
      expect(within(scoreCard).getByText('Feasibility: 70')).toBeInTheDocument();
      expect(within(scoreCard).getByText('Market Potential: 90')).toBeInTheDocument();
      expect(within(scoreCard).getByText('Uniqueness: 97')).toBeInTheDocument();
    });

    it('should provide detailed analysis for each idea', () => {
      const idea = {
        ...mockIdea(),
        analysis: {
          strengths: ['Innovative approach', 'Large market'],
          weaknesses: ['High initial cost', 'Regulatory challenges'],
          opportunities: ['First mover advantage', 'Partnership potential'],
          threats: ['Competition', 'Technology risk'],
        },
      };

      customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[idea]}
        />
      );

      const analysis = screen.getByTestId('idea-analysis');
      expect(within(analysis).getByText('Innovative approach')).toBeInTheDocument();
      expect(within(analysis).getByText('High initial cost')).toBeInTheDocument();
      expect(within(analysis).getByText('First mover advantage')).toBeInTheDocument();
      expect(within(analysis).getByText('Competition')).toBeInTheDocument();
    });
  });

  describe('Export and Sharing', () => {
    it('should allow exporting ideas to different formats', async () => {
      const ideas = [mockIdea(), mockIdea({ id: '2', title: 'Another Idea' })];
      const onExport = vi.fn();

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={ideas}
          onExport={onExport}
        />
      );

      await user.click(screen.getByRole('button', { name: /export ideas/i }));

      const formatSelect = screen.getByRole('combobox', { name: /export format/i });
      await user.selectOptions(formatSelect, 'business-plan');

      await user.click(screen.getByRole('button', { name: /download/i }));

      expect(onExport).toHaveBeenCalledWith({
        ideas,
        format: 'business-plan',
      });
    });

    it('should generate pitch decks', async () => {
      const idea = mockIdea();
      const onGeneratePitch = vi.fn();

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[idea]}
          onGeneratePitch={onGeneratePitch}
        />
      );

      await user.click(screen.getByRole('button', { name: /generate pitch deck/i }));

      // Configure pitch options
      const audienceSelect = screen.getByRole('combobox', { name: /target audience/i });
      await user.selectOptions(audienceSelect, 'investors');

      await user.click(screen.getByRole('button', { name: /create pitch/i }));

      expect(onGeneratePitch).toHaveBeenCalledWith({
        idea,
        audience: 'investors',
      });
    });

    it('should enable idea sharing with collaborators', async () => {
      const idea = mockIdea();
      const onShare = vi.fn();

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[idea]}
          onShare={onShare}
        />
      );

      await user.click(screen.getByRole('button', { name: /share idea/i }));

      const emailInput = screen.getByLabelText(/recipient email/i);
      await user.type(emailInput, 'collaborator@example.com');

      await user.click(screen.getByRole('button', { name: /send/i }));

      expect(onShare).toHaveBeenCalledWith({
        idea,
        recipient: 'collaborator@example.com',
      });
    });
  });

  describe('Comparison and Voting', () => {
    it('should allow comparing multiple ideas', async () => {
      const ideas = [
        mockIdea({ id: '1', title: 'Idea A', score: 85 }),
        mockIdea({ id: '2', title: 'Idea B', score: 92 }),
        mockIdea({ id: '3', title: 'Idea C', score: 78 }),
      ];

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={ideas}
        />
      );

      await user.click(screen.getByRole('button', { name: /compare ideas/i }));

      const comparisonTable = screen.getByTestId('comparison-table');
      expect(within(comparisonTable).getByText('Idea A')).toBeInTheDocument();
      expect(within(comparisonTable).getByText('Idea B')).toBeInTheDocument();
      expect(within(comparisonTable).getByText('Idea C')).toBeInTheDocument();

      // Should highlight best scores
      const bestScore = within(comparisonTable).getByText('92');
      expect(bestScore).toHaveClass('highlight-best');
    });

    it('should enable voting on ideas', async () => {
      const idea = mockIdea({ votes: 10 });
      const onVote = vi.fn();

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={[idea]}
          onVote={onVote}
        />
      );

      const voteButton = screen.getByRole('button', { name: /upvote/i });
      await user.click(voteButton);

      expect(onVote).toHaveBeenCalledWith({
        ideaId: idea.id,
        voteType: 'up',
      });

      // Should update vote count
      await waitFor(() => {
        expect(screen.getByText('11 votes')).toBeInTheDocument();
      });
    });
  });

  describe('History and Iterations', () => {
    it('should track generation history', () => {
      const history = [
        {
          id: 'gen1',
          timestamp: new Date('2024-01-01').toISOString(),
          domains: ['AI', 'Healthcare'],
          ideasGenerated: 5,
        },
        {
          id: 'gen2',
          timestamp: new Date('2024-01-02').toISOString(),
          domains: ['Finance', 'Space'],
          ideasGenerated: 3,
        },
      ];

      customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generationHistory={history}
        />
      );

      const historyTab = screen.getByRole('tab', { name: /history/i });
      historyTab.click();

      const historyList = screen.getByTestId('generation-history');
      expect(within(historyList).getByText('AI + Healthcare')).toBeInTheDocument();
      expect(within(historyList).getByText('5 ideas generated')).toBeInTheDocument();
    });

    it('should allow reverting to previous configurations', async () => {
      const previousConfig = {
        ...mockConfig,
        chaosLevel: 0.5,
        domainCount: 3,
      };

      const onRevert = vi.fn();

      const { user } = customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          previousConfigs={[previousConfig]}
          onRevert={onRevert}
        />
      );

      await user.click(screen.getByRole('button', { name: /previous config/i }));

      expect(onRevert).toHaveBeenCalledWith(previousConfig);
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels for all interactive elements', () => {
      customRender(<IdeaGenerator domains={mockDomains} config={mockConfig} />);

      expect(screen.getByRole('slider', { name: /chaos level/i })).toHaveAttribute('aria-valuemin', '0');
      expect(screen.getByRole('slider', { name: /chaos level/i })).toHaveAttribute('aria-valuemax', '100');

      mockDomains.forEach(domain => {
        const checkbox = screen.getByRole('checkbox', { name: new RegExp(domain.name, 'i') });
        expect(checkbox).toHaveAttribute('aria-describedby', expect.stringContaining('description'));
      });
    });

    it('should announce generation results to screen readers', async () => {
      const onGenerate = vi.fn().mockResolvedValue([mockIdea()]);
      const { user } = customRender(
        <IdeaGenerator domains={mockDomains} config={mockConfig} onGenerate={onGenerate} />
      );

      await user.click(screen.getByRole('checkbox', { name: /artificial intelligence/i }));
      await user.click(screen.getByRole('checkbox', { name: /healthcare/i }));
      await user.click(screen.getByRole('button', { name: /collide domains/i }));

      await waitFor(() => {
        const announcement = screen.getByRole('status');
        expect(announcement).toHaveTextContent('Generated 1 new idea');
      });
    });
  });

  describe('Performance', () => {
    it('should handle rapid domain selections efficiently', async () => {
      const { user } = customRender(
        <IdeaGenerator domains={mockDomains} config={mockConfig} />
      );

      const startTime = performance.now();

      // Rapidly select and deselect domains
      for (let i = 0; i < 10; i++) {
        await user.click(screen.getByRole('checkbox', { name: /artificial intelligence/i }));
      }

      const endTime = performance.now();
      expect(endTime - startTime).toBeLessThan(500); // Should handle rapid clicks in less than 500ms
    });

    it('should paginate large idea lists', () => {
      const manyIdeas = Array.from({ length: 50 }, (_, i) =>
        mockIdea({ id: `idea${i}`, title: `Idea ${i}` })
      );

      customRender(
        <IdeaGenerator
          domains={mockDomains}
          config={mockConfig}
          generatedIdeas={manyIdeas}
        />
      );

      // Should show pagination controls
      expect(screen.getByRole('navigation', { name: /pagination/i })).toBeInTheDocument();
      expect(screen.getByText('Page 1 of 5')).toBeInTheDocument();
      expect(screen.getAllByTestId('idea-card')).toHaveLength(10); // 10 per page
    });
  });
});