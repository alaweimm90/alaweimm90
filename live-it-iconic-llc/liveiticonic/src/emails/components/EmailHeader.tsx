import { Section, Text, Link } from '@react-email/components';

export function EmailHeader() {
  return (
    <Section style={header}>
      <Text style={logo}>
        <Link href="https://liveiconic.com" style={logoLink}>
          LIVE IT ICONIC
        </Link>
      </Text>
      <Text style={tagline}>
        Precision-cut apparel inspired by automotive discipline
      </Text>
    </Section>
  );
}

const header = {
  padding: '32px',
  backgroundColor: '#C1A060',
  borderRadius: '8px 8px 0 0',
  textAlign: 'center' as const,
};

const logo = {
  fontSize: '28px',
  fontWeight: 'bold',
  color: '#0B0B0C',
  margin: '0 0 8px',
  letterSpacing: '2px',
};

const logoLink = {
  color: '#0B0B0C',
  textDecoration: 'none',
};

const tagline = {
  fontSize: '13px',
  color: '#0B0B0C',
  margin: 0,
  fontStyle: 'italic',
};
