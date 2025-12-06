import { Section, Text, Link, Hr } from '@react-email/components';

export function EmailFooter() {
  const currentYear = new Date().getFullYear();

  return (
    <Section style={footer}>
      <Hr style={hr} />
      <Section style={socialLinks}>
        <Text style={socialText}>
          <Link href="https://instagram.com/liveiconic" style={link}>
            Instagram
          </Link>
          {' • '}
          <Link href="https://twitter.com/liveiconic" style={link}>
            Twitter
          </Link>
          {' • '}
          <Link href="https://liveiconic.com" style={link}>
            Website
          </Link>
        </Text>
      </Section>
      <Section style={contactSection}>
        <Text style={contactText}>
          Have questions?{' '}
          <Link href="mailto:hello@liveiconic.com" style={link}>
            hello@liveiconic.com
          </Link>
        </Text>
      </Section>
      <Section style={unsubscribeSection}>
        <Text style={unsubscribeText}>
          You received this email because you're subscribed to Live It Iconic.{' '}
          <Link href="https://liveiconic.com/unsubscribe" style={link}>
            Unsubscribe
          </Link>
        </Text>
      </Section>
      <Text style={copyright}>
        © {currentYear} Live It Iconic. All rights reserved.
      </Text>
    </Section>
  );
}

const footer = {
  padding: '32px',
  backgroundColor: '#f8f8f8',
  borderRadius: '0 0 8px 8px',
};

const hr = {
  borderColor: '#E6E9EF',
  margin: '0 0 24px',
};

const socialLinks = {
  margin: '24px 0',
  textAlign: 'center' as const,
};

const socialText = {
  fontSize: '14px',
  color: '#5C6270',
  margin: 0,
};

const contactSection = {
  margin: '16px 0',
  textAlign: 'center' as const,
};

const contactText = {
  fontSize: '13px',
  color: '#5C6270',
  margin: 0,
};

const unsubscribeSection = {
  margin: '16px 0',
  textAlign: 'center' as const,
};

const unsubscribeText = {
  fontSize: '12px',
  color: '#999999',
  margin: 0,
};

const copyright = {
  fontSize: '12px',
  color: '#999999',
  margin: '24px 0 0 0',
  textAlign: 'center' as const,
};

const link = {
  color: '#C1A060',
  textDecoration: 'underline',
};
