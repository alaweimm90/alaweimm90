/**
 * LiveItIconic Launch Platform - Agent Exports
 *
 * Central export file for all specialized agents
 */

// Market Intelligence Agents
export { CompetitorAnalystAgent } from './market/CompetitorAnalyst';
export { TrendDetectorAgent } from './market/TrendDetector';
export { CustomerResearcherAgent } from './market/CustomerResearcher';
export { PricingStrategistAgent } from './market/PricingStrategist';
export { MarketSizerAgent } from './market/MarketSizer';

// Creative & Branding Agents
export { BrandArchitectAgent } from './creative/BrandArchitect';
export { CopyWriterAgent } from './creative/CopyWriter';
export { VisualDesignerAgent } from './creative/VisualDesigner';
export { VideoProducerAgent } from './creative/VideoProducer';
export { StoryTellerAgent } from './creative/StoryTeller';

// Execution Agents
export { CampaignManagerAgent } from './execution/CampaignManager';
export { SocialMediaStrategistAgent } from './execution/SocialMediaStrategist';
export { InfluencerOutreachAgent } from './execution/InfluencerOutreach';
export { PRManagerAgent } from './execution/PRManager';
export { EmailMarketerAgent } from './execution/EmailMarketer';
export { ContentDistributorAgent } from './execution/ContentDistributor';

// Optimization Agents
export { AnalyticsInterpreterAgent } from './optimization/AnalyticsInterpreter';
export { ConversionOptimizerAgent } from './optimization/ConversionOptimizer';
export { SEOSpecialistAgent } from './optimization/SEOSpecialist';
export { PaidAdsManagerAgent } from './optimization/PaidAdsManager';
export { FeedbackAnalyzerAgent } from './optimization/FeedbackAnalyzer';

// Supporting Agents
export { DataCollectorAgent } from './supporting/DataCollector';
export { QualityControllerAgent } from './supporting/QualityController';
export { ComplianceCheckerAgent } from './supporting/ComplianceChecker';
export { BudgetManagerAgent } from './supporting/BudgetManager';
export { RiskAssessorAgent } from './supporting/RiskAssessor';

// Export agent factory for easy initialization
import { CompetitorAnalystAgent } from './market/CompetitorAnalyst';
import { TrendDetectorAgent } from './market/TrendDetector';
import { CustomerResearcherAgent } from './market/CustomerResearcher';
import { PricingStrategistAgent } from './market/PricingStrategist';
import { MarketSizerAgent } from './market/MarketSizer';
import { BrandArchitectAgent } from './creative/BrandArchitect';
import { CopyWriterAgent } from './creative/CopyWriter';
import { VisualDesignerAgent } from './creative/VisualDesigner';
import { VideoProducerAgent } from './creative/VideoProducer';
import { StoryTellerAgent } from './creative/StoryTeller';
import { CampaignManagerAgent } from './execution/CampaignManager';
import { SocialMediaStrategistAgent } from './execution/SocialMediaStrategist';
import { InfluencerOutreachAgent } from './execution/InfluencerOutreach';
import { PRManagerAgent } from './execution/PRManager';
import { EmailMarketerAgent } from './execution/EmailMarketer';
import { ContentDistributorAgent } from './execution/ContentDistributor';
import { AnalyticsInterpreterAgent } from './optimization/AnalyticsInterpreter';
import { ConversionOptimizerAgent } from './optimization/ConversionOptimizer';
import { SEOSpecialistAgent } from './optimization/SEOSpecialist';
import { PaidAdsManagerAgent } from './optimization/PaidAdsManager';
import { FeedbackAnalyzerAgent } from './optimization/FeedbackAnalyzer';
import { DataCollectorAgent } from './supporting/DataCollector';
import { QualityControllerAgent } from './supporting/QualityController';
import { ComplianceCheckerAgent } from './supporting/ComplianceChecker';
import { BudgetManagerAgent } from './supporting/BudgetManager';
import { RiskAssessorAgent } from './supporting/RiskAssessor';
import { AgentType } from '../types';
import { BaseAgent } from '../core/BaseAgent';

export class AgentFactory {
  /**
   * Create an agent by type
   */
  static createAgent(type: AgentType, customId?: string): BaseAgent | null {
    switch (type) {
      // Market Intelligence
      case AgentType.COMPETITOR_ANALYST:
        return new CompetitorAnalystAgent(customId);
      case AgentType.TREND_DETECTOR:
        return new TrendDetectorAgent(customId);
      case AgentType.CUSTOMER_RESEARCHER:
        return new CustomerResearcherAgent(customId);
      case AgentType.PRICING_STRATEGIST:
        return new PricingStrategistAgent(customId);
      case AgentType.MARKET_SIZER:
        return new MarketSizerAgent(customId);

      // Creative & Branding
      case AgentType.BRAND_ARCHITECT:
        return new BrandArchitectAgent(customId);
      case AgentType.COPYWRITER:
        return new CopyWriterAgent(customId);
      case AgentType.VISUAL_DESIGNER:
        return new VisualDesignerAgent(customId);
      case AgentType.VIDEO_PRODUCER:
        return new VideoProducerAgent(customId);
      case AgentType.STORYTELLER:
        return new StoryTellerAgent(customId);

      // Execution
      case AgentType.CAMPAIGN_MANAGER:
        return new CampaignManagerAgent(customId);
      case AgentType.SOCIAL_MEDIA_STRATEGIST:
        return new SocialMediaStrategistAgent(customId);
      case AgentType.INFLUENCER_OUTREACH:
        return new InfluencerOutreachAgent(customId);
      case AgentType.PR_MANAGER:
        return new PRManagerAgent(customId);
      case AgentType.EMAIL_MARKETER:
        return new EmailMarketerAgent(customId);
      case AgentType.CONTENT_DISTRIBUTOR:
        return new ContentDistributorAgent(customId);

      // Optimization
      case AgentType.ANALYTICS_INTERPRETER:
        return new AnalyticsInterpreterAgent(customId);
      case AgentType.CONVERSION_OPTIMIZER:
        return new ConversionOptimizerAgent(customId);
      case AgentType.SEO_SPECIALIST:
        return new SEOSpecialistAgent(customId);
      case AgentType.PAID_ADS_MANAGER:
        return new PaidAdsManagerAgent(customId);
      case AgentType.FEEDBACK_ANALYZER:
        return new FeedbackAnalyzerAgent(customId);

      // Supporting
      case AgentType.DATA_COLLECTOR:
        return new DataCollectorAgent(customId);
      case AgentType.QUALITY_CONTROLLER:
        return new QualityControllerAgent(customId);
      case AgentType.COMPLIANCE_CHECKER:
        return new ComplianceCheckerAgent(customId);
      case AgentType.BUDGET_MANAGER:
        return new BudgetManagerAgent(customId);
      case AgentType.RISK_ASSESSOR:
        return new RiskAssessorAgent(customId);

      default:
        console.warn(`Agent type not yet implemented: ${type}`);
        return null;
    }
  }

  /**
   * Create all available agents
   */
  static createAllAgents(): BaseAgent[] {
    const agents: BaseAgent[] = [];

    // Market Intelligence
    agents.push(new CompetitorAnalystAgent());
    agents.push(new TrendDetectorAgent());
    agents.push(new CustomerResearcherAgent());
    agents.push(new PricingStrategistAgent());
    agents.push(new MarketSizerAgent());

    // Creative & Branding
    agents.push(new BrandArchitectAgent());
    agents.push(new CopyWriterAgent());
    agents.push(new VisualDesignerAgent());
    agents.push(new VideoProducerAgent());
    agents.push(new StoryTellerAgent());

    // Execution
    agents.push(new CampaignManagerAgent());
    agents.push(new SocialMediaStrategistAgent());
    agents.push(new InfluencerOutreachAgent());
    agents.push(new PRManagerAgent());
    agents.push(new EmailMarketerAgent());
    agents.push(new ContentDistributorAgent());

    // Optimization
    agents.push(new AnalyticsInterpreterAgent());
    agents.push(new ConversionOptimizerAgent());
    agents.push(new SEOSpecialistAgent());
    agents.push(new PaidAdsManagerAgent());
    agents.push(new FeedbackAnalyzerAgent());

    // Supporting
    agents.push(new DataCollectorAgent());
    agents.push(new QualityControllerAgent());
    agents.push(new ComplianceCheckerAgent());
    agents.push(new BudgetManagerAgent());
    agents.push(new RiskAssessorAgent());

    return agents;
  }

  /**
   * Create agents for a specific phase
   */
  static createAgentsForPhase(phase: 'analysis' | 'creative' | 'execution' | 'optimization' | 'supporting'): BaseAgent[] {
    const agents: BaseAgent[] = [];

    switch (phase) {
      case 'analysis':
        agents.push(new CompetitorAnalystAgent());
        agents.push(new TrendDetectorAgent());
        agents.push(new CustomerResearcherAgent());
        agents.push(new PricingStrategistAgent());
        agents.push(new MarketSizerAgent());
        break;

      case 'creative':
        agents.push(new BrandArchitectAgent());
        agents.push(new CopyWriterAgent());
        agents.push(new VisualDesignerAgent());
        agents.push(new VideoProducerAgent());
        agents.push(new StoryTellerAgent());
        break;

      case 'execution':
        agents.push(new CampaignManagerAgent());
        agents.push(new SocialMediaStrategistAgent());
        agents.push(new InfluencerOutreachAgent());
        agents.push(new PRManagerAgent());
        agents.push(new EmailMarketerAgent());
        agents.push(new ContentDistributorAgent());
        break;

      case 'optimization':
        agents.push(new AnalyticsInterpreterAgent());
        agents.push(new ConversionOptimizerAgent());
        agents.push(new SEOSpecialistAgent());
        agents.push(new PaidAdsManagerAgent());
        agents.push(new FeedbackAnalyzerAgent());
        break;

      case 'supporting':
        agents.push(new DataCollectorAgent());
        agents.push(new QualityControllerAgent());
        agents.push(new ComplianceCheckerAgent());
        agents.push(new BudgetManagerAgent());
        agents.push(new RiskAssessorAgent());
        break;
    }

    return agents;
  }

  /**
   * Get count of implemented agents
   */
  static getImplementedAgentsCount(): { total: number; byCategory: Record<string, number> } {
    return {
      total: 26,
      byCategory: {
        'Market Intelligence': 5,
        'Creative & Branding': 5,
        'Launch Execution': 6,
        'Optimization': 5,
        'Supporting': 5,
      },
    };
  }

  /**
   * Get list of agent types that are implemented
   */
  static getImplementedTypes(): AgentType[] {
    return [
      // Market Intelligence
      AgentType.COMPETITOR_ANALYST,
      AgentType.TREND_DETECTOR,
      AgentType.CUSTOMER_RESEARCHER,
      AgentType.PRICING_STRATEGIST,
      AgentType.MARKET_SIZER,
      // Creative & Branding
      AgentType.BRAND_ARCHITECT,
      AgentType.COPYWRITER,
      AgentType.VISUAL_DESIGNER,
      AgentType.VIDEO_PRODUCER,
      AgentType.STORYTELLER,
      // Launch Execution
      AgentType.CAMPAIGN_MANAGER,
      AgentType.SOCIAL_MEDIA_STRATEGIST,
      AgentType.INFLUENCER_OUTREACH,
      AgentType.PR_MANAGER,
      AgentType.EMAIL_MARKETER,
      AgentType.CONTENT_DISTRIBUTOR,
      // Optimization
      AgentType.ANALYTICS_INTERPRETER,
      AgentType.CONVERSION_OPTIMIZER,
      AgentType.SEO_SPECIALIST,
      AgentType.PAID_ADS_MANAGER,
      AgentType.FEEDBACK_ANALYZER,
      // Supporting
      AgentType.DATA_COLLECTOR,
      AgentType.QUALITY_CONTROLLER,
      AgentType.COMPLIANCE_CHECKER,
      AgentType.BUDGET_MANAGER,
      AgentType.RISK_ASSESSOR,
    ];
  }

  /**
   * Check if an agent type is implemented
   */
  static isImplemented(type: AgentType): boolean {
    return this.getImplementedTypes().includes(type);
  }
}
