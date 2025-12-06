import { Request, Response } from 'express';
import { contentGenerationService } from '../services/content-generation.service';
import { ApiResponse } from '@marketing-automation/types';

export class ContentController {
  async generateContent(req: Request, res: Response) {
    try {
      const result = await contentGenerationService.generateContent(req.body);
      const response: ApiResponse<any> = {
        success: true,
        data: result
      };
      res.json(response);
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateSocialPost(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generateSocialMediaPost(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateBlogArticle(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generateBlogArticle(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateEmailCampaign(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generateEmailCampaign(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateVideoScript(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generateVideoScript(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateAdCopy(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generateAdCopy(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateProductDescription(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generateProductDescription(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generatePressRelease(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generatePressRelease(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generatePodcastScript(req: Request, res: Response) {
    try {
      const content = await contentGenerationService.generatePodcastScript(req.body);
      res.json({ success: true, data: content });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateContentCalendar(req: Request, res: Response) {
    try {
      const calendar = await contentGenerationService.generateContentCalendar(req.body);
      res.json({ success: true, data: calendar });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async generateCampaign(req: Request, res: Response) {
    try {
      const campaign = await contentGenerationService.generateCampaignContent(req.body);
      res.json({ success: true, data: campaign });
    } catch (error: any) {
      res.status(500).json({
        success: false,
        error: { code: 'GENERATION_FAILED', message: error.message }
      });
    }
  }

  async getContent(req: Request, res: Response) {
    res.json({ success: true, data: { id: req.params.id } });
  }

  async updateContent(req: Request, res: Response) {
    res.json({ success: true, data: { id: req.params.id } });
  }

  async deleteContent(req: Request, res: Response) {
    res.json({ success: true, data: { deleted: true } });
  }

  async listContent(req: Request, res: Response) {
    res.json({ success: true, data: [] });
  }
}
