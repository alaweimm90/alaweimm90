import {
  Html,
  Head,
  Body,
  Container,
  Section,
  Preview,
} from '@react-email/components';
import { EmailHeader } from './EmailHeader';
import { EmailFooter } from './EmailFooter';

interface EmailLayoutProps {
  children: React.ReactNode;
  preview: string;
}

export function EmailLayout({ children, preview }: EmailLayoutProps) {
  return (
    <Html>
      <Head />
      <Preview>{preview}</Preview>
      <Body style={main}>
        <Container style={container}>
          <EmailHeader />
          <Section style={content}>
            {children}
          </Section>
          <EmailFooter />
        </Container>
      </Body>
    </Html>
  );
}

const main = {
  backgroundColor: '#0B0B0C',
  fontFamily:
    '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif',
  lineHeight: '1.4',
  margin: 0,
  padding: 0,
};

const container = {
  margin: '0 auto',
  padding: '20px 0 48px',
  maxWidth: '600px',
};

const content = {
  padding: '32px',
  backgroundColor: '#ffffff',
  borderRadius: '0 0 8px 8px',
};
