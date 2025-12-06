import {
  EmailCampaign,
  EmailCampaignType,
  EmailPerformance,
  EmailSegment,
  ABTest
} from '@marketing-automation/types';
import sendgrid from '@sendgrid/mail';
import nodemailer from 'nodemailer';
import mjml from 'mjml';
import Handlebars from 'handlebars';

export class EmailService {
  private transporter: any;

  constructor() {
    sendgrid.setApiKey(process.env.SENDGRID_API_KEY || '');

    this.transporter = nodemailer.createTransporter({
      host: process.env.SMTP_HOST,
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
      }
    });
  }

  async sendCampaign(campaign: EmailCampaign): Promise<void> {
    // Compile template with Handlebars
    const template = Handlebars.compile(campaign.htmlContent);

    // Send to all segments
    for (const segment of campaign.segments) {
      await this.sendToSegment(campaign, segment, template);
    }
  }

  private async sendToSegment(
    campaign: EmailCampaign,
    segment: EmailSegment,
    template: HandlebarsTemplateDelegate
  ): Promise<void> {
    // Get recipients from segment (would query database)
    const recipients = await this.getSegmentRecipients(segment);

    for (const recipient of recipients) {
      // Personalize content
      const personalizedContent = template({
        firstName: recipient.firstName,
        lastName: recipient.lastName,
        email: recipient.email,
        ...recipient.customFields
      });

      // Send email
      await this.sendEmail({
        to: recipient.email,
        from: `${campaign.fromName} <${campaign.fromEmail}>`,
        replyTo: campaign.replyTo,
        subject: campaign.subject,
        html: personalizedContent,
        text: campaign.textContent
      });
    }
  }

  private async sendEmail(options: {
    to: string;
    from: string;
    replyTo: string;
    subject: string;
    html: string;
    text: string;
  }): Promise<void> {
    if (process.env.EMAIL_PROVIDER === 'sendgrid') {
      await sendgrid.send(options);
    } else {
      await this.transporter.sendMail(options);
    }
  }

  async sendTransactional(params: {
    to: string;
    templateId: string;
    dynamicData: Record<string, any>;
  }): Promise<void> {
    await sendgrid.send({
      to: params.to,
      from: process.env.DEFAULT_FROM_EMAIL || '',
      templateId: params.templateId,
      dynamicTemplateData: params.dynamicData
    });
  }

  async createDripCampaign(params: {
    name: string;
    emails: {
      delay: number; // days after subscription
      subject: string;
      content: string;
    }[];
  }): Promise<void> {
    // Create automated drip campaign
    // This would set up scheduled emails based on user actions
  }

  async trackEmailOpen(emailId: string, recipientId: string): Promise<void> {
    // Track email opens via pixel
  }

  async trackEmailClick(emailId: string, recipientId: string, url: string): Promise<void> {
    // Track link clicks
  }

  async getCampaignPerformance(campaignId: string): Promise<EmailPerformance> {
    // Fetch performance metrics from database
    return {
      sent: 1000,
      delivered: 980,
      opened: 350,
      clicked: 120,
      converted: 25,
      bounced: 20,
      unsubscribed: 5,
      spam: 2,
      openRate: 35.7,
      clickRate: 12.2,
      conversionRate: 2.5,
      bounceRate: 2.0,
      unsubscribeRate: 0.5,
      revenue: 2500
    };
  }

  async segmentAudience(criteria: any): Promise<EmailSegment> {
    // Create audience segment based on criteria
    return {
      id: 'seg_123',
      name: 'High Value Customers',
      criteria,
      recipientCount: 500
    };
  }

  async runABTest(test: ABTest): Promise<string> {
    // Run A/B test and return winning variant ID
    return 'variant_a';
  }

  async optimizeSendTime(recipientId: string): Promise<Date> {
    // Determine optimal send time for individual recipient
    return new Date();
  }

  async validateEmailList(emails: string[]): Promise<{ valid: string[]; invalid: string[] }> {
    // Validate email addresses
    const valid: string[] = [];
    const invalid: string[] = [];

    for (const email of emails) {
      if (this.isValidEmail(email)) {
        valid.push(email);
      } else {
        invalid.push(email);
      }
    }

    return { valid, invalid };
  }

  private isValidEmail(email: string): boolean {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  }

  async compileFromMJML(mjmlContent: string): Promise<string> {
    const result = mjml(mjmlContent);
    if (result.errors.length > 0) {
      throw new Error(`MJML compilation failed: ${result.errors[0].message}`);
    }
    return result.html;
  }

  private async getSegmentRecipients(segment: EmailSegment): Promise<any[]> {
    // Query database for segment recipients
    return [];
  }

  // ==================== SPECIALIZED CAMPAIGNS ====================

  async sendWelcomeEmail(params: {
    to: string;
    firstName: string;
    companyName: string;
  }): Promise<void> {
    const template = `
      <h1>Welcome to {{companyName}}, {{firstName}}!</h1>
      <p>We're excited to have you on board.</p>
      <a href="{{ctaUrl}}">Get Started</a>
    `;

    const compiled = Handlebars.compile(template);
    const html = compiled(params);

    await this.sendEmail({
      to: params.to,
      from: process.env.DEFAULT_FROM_EMAIL || '',
      replyTo: process.env.DEFAULT_FROM_EMAIL || '',
      subject: `Welcome to ${params.companyName}!`,
      html,
      text: `Welcome to ${params.companyName}, ${params.firstName}!`
    });
  }

  async sendAbandonedCartEmail(params: {
    to: string;
    firstName: string;
    cartItems: any[];
    cartTotal: number;
  }): Promise<void> {
    // Send abandoned cart recovery email
  }

  async sendNewsLetter(params: {
    subject: string;
    articles: { title: string; excerpt: string; url: string }[];
    segmentId: string;
  }): Promise<void> {
    // Send newsletter to segment
  }

  async sendPromotionalEmail(params: {
    subject: string;
    offer: string;
    discountCode: string;
    expiryDate: Date;
    segmentId: string;
  }): Promise<void> {
    // Send promotional email
  }
}

export const emailService = new EmailService();
