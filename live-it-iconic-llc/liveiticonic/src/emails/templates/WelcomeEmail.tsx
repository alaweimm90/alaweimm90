import { Heading, Text, Button, Hr, Section } from '@react-email/components';
import { EmailLayout } from '../components/EmailLayout';

interface WelcomeEmailProps {
  name: string;
}

export function WelcomeEmail({ name }: WelcomeEmailProps) {
  return (
    <EmailLayout preview="Welcome to Live It Iconic">
      <Heading style={h1}>Welcome to Live It Iconic</Heading>
      <Text style={text}>Hi {name},</Text>
      <Text style={text}>
        Thank you for joining us. We're excited to have you as part of the Iconic community.
      </Text>
      <Text style={text}>
        Our collection features precision-cut apparel inspired by automotive discipline —
        designed for motion, refined for nights out.
      </Text>

      <Section style={ctaSection}>
        <Button style={button} href="https://liveiconic.com/shop">
          Shop the Collection
        </Button>
      </Section>

      <Hr style={hr} />

      <Section style={offerSection}>
        <Heading style={h3}>Exclusive Offer</Heading>
        <Text style={text}>
          Get <strong>10% off your first order</strong> with code:
        </Text>
        <Text style={couponCode}>WELCOME10</Text>
        <Text style={smallText}>
          Offer valid for 14 days from your signup date
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={benefitsSection}>
        <Heading style={h3}>Why Live It Iconic?</Heading>
        <Text style={benefitItem}>
          <strong>Premium Quality:</strong> Precision-cut garments crafted with attention to detail
        </Text>
        <Text style={benefitItem}>
          <strong>Lifestyle Brand:</strong> Inspired by automotive discipline and precision
        </Text>
        <Text style={benefitItem}>
          <strong>Fast Shipping:</strong> Get your order within 3-5 business days
        </Text>
        <Text style={benefitItem}>
          <strong>30-Day Returns:</strong> Shop with confidence with our hassle-free returns
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

const button = {
  backgroundColor: '#C1A060',
  color: '#0B0B0C',
  padding: '14px 32px',
  borderRadius: '6px',
  textDecoration: 'none',
  display: 'inline-block',
  fontWeight: 600,
  fontSize: '16px',
  cursor: 'pointer',
  border: 'none',
};

const hr = {
  borderColor: '#E6E9EF',
  margin: '24px 0',
  borderWidth: '1px',
  borderStyle: 'solid',
};

const ctaSection = {
  textAlign: 'center' as const,
  margin: '24px 0',
};

const offerSection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const couponCode = {
  color: '#C1A060',
  fontSize: '20px',
  fontWeight: 'bold',
  margin: '12px 0',
  textAlign: 'center' as const,
  display: 'block' as const,
  letterSpacing: '2px',
};

const benefitsSection = {
  margin: '24px 0',
};

const benefitItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 12px',
  paddingLeft: '16px',
};
