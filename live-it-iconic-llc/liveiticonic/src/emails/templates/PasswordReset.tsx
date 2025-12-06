import { Heading, Text, Button, Hr, Section, Link } from '@react-email/components';
import { EmailLayout } from '../components/EmailLayout';

interface PasswordResetProps {
  name: string;
  resetUrl: string;
  expirationTime?: string;
}

export function PasswordReset({
  name,
  resetUrl,
  expirationTime = '24 hours',
}: PasswordResetProps) {
  return (
    <EmailLayout preview="Reset your password">
      <Heading style={h1}>Reset Your Password</Heading>
      <Text style={text}>Hi {name},</Text>
      <Text style={text}>
        We received a request to reset your password. Click the button below to create a new password.
      </Text>

      <Section style={ctaSection}>
        <Button style={button} href={resetUrl}>
          Reset Password
        </Button>
      </Section>

      <Text style={smallText}>
        Or copy this link in your browser: <Link href={resetUrl} style={link}>{resetUrl}</Link>
      </Text>

      <Hr style={hr} />

      <Section style={securitySection}>
        <Heading style={h3}>Security Information</Heading>
        <Text style={securityItem}>
          <strong>Link expiration:</strong> This reset link will expire in {expirationTime} for your security.
        </Text>
        <Text style={securityItem}>
          <strong>Safe to click:</strong> This link only works for you and is completely safe.
        </Text>
        <Text style={securityItem}>
          <strong>Choose a strong password:</strong> Use a combination of uppercase, lowercase, numbers, and symbols.
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={warningSection}>
        <Heading style={h3}>Didn't Request This?</Heading>
        <Text style={text}>
          If you didn't request a password reset, you can safely ignore this email. Your account is secure.
        </Text>
        <Text style={text}>
          If you believe someone else is trying to access your account,{' '}
          <Link href="mailto:hello@liveiconic.com" style={link}>
            contact us immediately
          </Link>
          .
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={tipsSection}>
        <Heading style={h3}>Password Best Practices</Heading>
        <Text style={tipItem}>
          <strong>Use a unique password:</strong> Don't use the same password for multiple accounts
        </Text>
        <Text style={tipItem}>
          <strong>Avoid personal information:</strong> Don't use birthdates, addresses, or other personal details
        </Text>
        <Text style={tipItem}>
          <strong>Enable two-factor authentication:</strong> Add an extra layer of security to your account
        </Text>
        <Text style={tipItem}>
          <strong>Use a password manager:</strong> Tools like 1Password or LastPass help you create and remember secure passwords
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={supportSection}>
        <Text style={text}>
          Questions? Visit our{' '}
          <Link href="https://liveiconic.com/help/account" style={link}>
            Account Help Center
          </Link>
          {' '}or contact us at{' '}
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
  wordBreak: 'break-all' as const,
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

const securitySection = {
  backgroundColor: '#f0f8f0',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const securityItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 12px',
  paddingLeft: '16px',
};

const warningSection = {
  backgroundColor: '#FFF8E1',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const tipsSection = {
  margin: '24px 0',
};

const tipItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 12px',
  paddingLeft: '16px',
};

const supportSection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const link = {
  color: '#C1A060',
  textDecoration: 'underline',
};
