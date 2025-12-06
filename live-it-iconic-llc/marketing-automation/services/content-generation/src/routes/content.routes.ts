import { Router } from 'express';
import { ContentController } from '../controllers/content.controller';

const router = Router();
const controller = new ContentController();

// General content generation
router.post('/generate', controller.generateContent);

// Specialized endpoints
router.post('/social-post', controller.generateSocialPost);
router.post('/blog-article', controller.generateBlogArticle);
router.post('/email-campaign', controller.generateEmailCampaign);
router.post('/video-script', controller.generateVideoScript);
router.post('/ad-copy', controller.generateAdCopy);
router.post('/product-description', controller.generateProductDescription);
router.post('/press-release', controller.generatePressRelease);
router.post('/podcast-script', controller.generatePodcastScript);

// Bulk operations
router.post('/content-calendar', controller.generateContentCalendar);
router.post('/campaign', controller.generateCampaign);

// Content management
router.get('/:id', controller.getContent);
router.put('/:id', controller.updateContent);
router.delete('/:id', controller.deleteContent);
router.get('/', controller.listContent);

export { router as contentRouter };
