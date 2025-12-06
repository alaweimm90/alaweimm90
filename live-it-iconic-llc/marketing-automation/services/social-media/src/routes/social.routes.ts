import { Router } from 'express';
import { socialMediaService } from '../services/social-media.service';

const router = Router();

router.post('/publish', async (req, res) => {
  try {
    const result = await socialMediaService.publishPost(req.body);
    res.json({ success: true, data: result });
  } catch (error: any) {
    res.status(500).json({ success: false, error: { message: error.message } });
  }
});

router.post('/schedule', async (req, res) => {
  try {
    const result = await socialMediaService.schedulePost(req.body);
    res.json({ success: true, data: result });
  } catch (error: any) {
    res.status(500).json({ success: false, error: { message: error.message } });
  }
});

router.delete('/post/:id', async (req, res) => {
  try {
    const result = await socialMediaService.deletePost(req.body);
    res.json({ success: true, data: result });
  } catch (error: any) {
    res.status(500).json({ success: false, error: { message: error.message } });
  }
});

router.get('/performance/:id', async (req, res) => {
  try {
    const result = await socialMediaService.getPostPerformance(req.body);
    res.json({ success: true, data: result });
  } catch (error: any) {
    res.status(500).json({ success: false, error: { message: error.message } });
  }
});

router.post('/engage/comment', async (req, res) => {
  try {
    await socialMediaService.replyToComment(req.body.platform, req.body.commentId, req.body.message);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ success: false, error: { message: error.message } });
  }
});

router.get('/analytics/:platform/:accountId', async (req, res) => {
  try {
    const result = await socialMediaService.getAccountAnalytics(
      req.params.platform as any,
      req.params.accountId,
      req.body.dateRange
    );
    res.json({ success: true, data: result });
  } catch (error: any) {
    res.status(500).json({ success: false, error: { message: error.message } });
  }
});

export { router as socialMediaRouter };
