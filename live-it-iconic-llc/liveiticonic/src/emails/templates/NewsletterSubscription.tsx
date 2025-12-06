import { Heading, Text, Button, Hr, Section, Link } from '@react-email/components';
import { EmailLayout } from '../components/EmailLayout';

interface NewsletterSubscriptionProps {
  name: string;
  confirmUrl: string;
  interests?: string[];
}

export function NewsletterSubscription({
  name,
  confirmUrl,
  interests = ['New Releases', 'Styling Tips', 'Exclusive Offers'],
}: NewsletterSubscriptionProps) {
  return (
    <EmailLayout preview="Welcome to the Live It Iconic Newsletter">
      <Heading style={h1}>Welcome to Our Newsletter</Heading>
      <Text style={text}>Hi {name},</Text>
      <Text style={text}>
        Thank you for subscribing to the Live It Iconic newsletter! We're thrilled to have you join our community of automotive enthusiasts and lifestyle connoisseurs.
      </Text>

      <Section style={welcomeSection}>
        <Text style={welcomeTitle}>What You'll Get</Text>
        <Text style={benefitItem}>
          <strong>New Release Updates:</strong> Be the first to know about new collections and products
        </Text>
        <Text style={benefitItem}>
          <strong>Styling Inspiration:</strong> Tips and guides on how to style your Live It Iconic pieces
        </Text>
        <Text style={benefitItem}>
          <strong>Exclusive Offers:</strong> Special discounts and promotions for newsletter subscribers
        </Text>
        <Text style={benefitItem}>
          <strong>Community Stories:</strong> Features from our community of Iconic enthusiasts
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={confirmSection}>
        <Heading style={h3}>Confirm Your Subscription</Heading>
        <Text style={text}>
          To ensure you really want to receive our newsletter, please confirm your subscription by clicking the button below:
        </Text>

        <Section style={ctaSection}>
          <Button style={button} href={confirmUrl}>
            Confirm Subscription
          </Button>
        </Section>

        <Text style={smallText}>
          If you didn't subscribe to our newsletter or prefer not to receive emails from us,{' '}
          <Link href={confirmUrl.replace('/confirm', '/unsubscribe')} style={link}>
            click here to unsubscribe
          </Link>
          .
        </Text>
      </Section>

      <Hr style={hr} />

      {interests && interests.length > 0 && (
        <>
          <Section style={interestsSection}>
            <Heading style={h3}>Your Interests</Heading>
            <Text style={interestText}>
              You've expressed interest in the following topics:
            </Text>
            <Text style={interestList}>
              {interests.join(' • ')}
            </Text>
            <Text style={smallText}>
              Want to change your preferences?{' '}
              <Link href="https://liveiconic.com/newsletter-preferences" style={link}>
                Manage your preferences
              </Link>
            </Text>
          </Section>

          <Hr style={hr} />
        </>
      )}

      <Section style={aboutSection}>
        <Heading style={h3}>About Live It Iconic</Heading>
        <Text style={text}>
          Live It Iconic represents a lifestyle rooted in precision, discipline, and sophistication. Our collections are designed for those who appreciate automotive engineering and luxury lifestyle. Each piece is crafted with attention to detail, embodying the essence of motion and refinement.
        </Text>
        <Text style={text}>
          Explore our latest collection:{' '}
          <Link href="https://liveiconic.com/shop" style={link}>
            View Shop
          </Link>
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={connectSection}>
        <Heading style={h3}>Connect With Us</Heading>
        <Text style={text}>
          Follow us on social media to stay updated and see styling inspiration:
        </Text>
        <Text style={socialLinks}>
          <Link href="https://instagram.com/liveiconic" style={link}>
            Instagram
          </Link>
          {' • '}
          <Link href="https://twitter.com/liveiconic" style={link}>
            Twitter
          </Link>
          {' • '}
          <Link href="https://tiktok.com/@liveiconic" style={link}>
            TikTok
          </Link>
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={frequencySection}>
        <Heading style={h3}>Email Frequency</Heading>
        <Text style={text}>
          We send newsletters approximately once per week, but we also send special announcements for new releases and exclusive offers. If this is too much, you can{' '}
          <Link href="https://liveiconic.com/newsletter-preferences" style={link}>
            adjust your frequency preferences
          </Link>
          .
        </Text>
      </Section>

      <Section style={supportSection}>
        <Text style={text}>
          Questions? We'd love to hear from you. Contact us at{' '}
          <Link href="mailto:hello@liveiconic.com" style={link}>
            hello@liveiconic.com
          </Link>
          .
        </Text>
      </Section>
    </EmailLayout>
  );
}

const h1 = {
  color: '#0B0B0C',
  fontSize: '28px',
  fontWeight: 'bold',
  margin: '0 0 16px',
};

const h3 = {
  color: '#0B0B0C',
  fontSize: '18px',
  fontWeight: 'bold',
  margin: '16px 0 12px',
};

const text = {
  color: '#0B0B0C',
  fontSize: '16px',
  lineHeight: '24px',
  margin: '0 0 16px',
};

const smallText = {
  color: '#5C6270',
  fontSize: '13px',
  lineHeight: '20px',
  margin: '0 0 16px',
};

const hr = {
  borderColor: '#E6E9EF',
  margin: '24px 0',
  borderWidth: '1px',
  borderStyle: 'solid',
};

const welcomeSection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const welcomeTitle = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: 'bold',
  margin: '0 0 12px',
};

const benefitItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 8px',
  paddingLeft: '16px',
};

const confirmSection = {
  backgroundColor: '#f0f8f0',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const ctaSection = {
  textAlign: 'center' as const,
  margin: '20px 0',
};

const button = {
  backgroundColor: '#C1A060',
  color: '#0B0B0C',
  padding: '14px 32px',
  borderRadius: '6px',
  textDecoration: 'none',
  display: 'inline-block',
  fontWeight: '600',
  fontSize: '16px',
  cursor: 'pointer',
  border: 'none',
};

const interestsSection = {
  margin: '24px 0',
};

const interestText = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 8px',
};

const interestList = {
  color: '#C1A060',
  fontSize: '15px',
  fontWeight: '600',
  margin: '0 0 12px',
  padding: '12px',
  backgroundColor: '#f8f8f8',
  borderRadius: '4px',
};

const aboutSection = {
  backgroundColor: '#FFF8E1',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const connectSection = {
  margin: '24px 0',
};

const socialLinks = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '12px 0',
  textAlign: 'center' as const,
};

const frequencySection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const supportSection = {
  backgroundColor: '#f0f0f0',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
  textAlign: 'center' as const,
};

const link = {
  color: '#C1A060',
  textDecoration: 'underline',
};
