import { Heading, Text, Button, Hr, Section, Link } from '@react-email/components';
import { EmailLayout } from '../components/EmailLayout';

interface ShippingNotificationProps {
  name: string;
  orderId: string;
  trackingNumber: string;
  carrier: 'UPS' | 'FedEx' | 'DHL' | 'USPS';
  estimatedDelivery: string;
  shippingAddress?: string;
}

export function ShippingNotification({
  name,
  orderId,
  trackingNumber,
  carrier,
  estimatedDelivery,
  shippingAddress,
}: ShippingNotificationProps) {
  const trackingUrls: Record<string, string> = {
    UPS: `https://tracking.ups.com/track?tracknum=${trackingNumber}`,
    FedEx: `https://tracking.fedex.com/en-us/tracking?tracknumbers=${trackingNumber}`,
    DHL: `https://www.dhl.com/en/en/express/tracking.html?AWB=${trackingNumber}`,
    USPS: `https://tools.usps.com/go/TrackConfirmAction?tLabels=${trackingNumber}`,
  };

  return (
    <EmailLayout preview="Your order has shipped!">
      <Heading style={h1}>Your Order Has Shipped</Heading>
      <Text style={text}>Hi {name},</Text>
      <Text style={text}>
        Great news! Your order #{orderId} is on its way to you.
      </Text>

      <Section style={trackingBox}>
        <Heading style={trackingTitle}>Tracking Information</Heading>

        <Section style={trackingInfoRow}>
          <Text style={trackingLabel}>Order Number</Text>
          <Text style={trackingValue}>{orderId}</Text>
        </Section>

        <Section style={trackingInfoRow}>
          <Text style={trackingLabel}>Carrier</Text>
          <Text style={trackingValue}>{carrier}</Text>
        </Section>

        <Section style={trackingInfoRow}>
          <Text style={trackingLabel}>Tracking Number</Text>
          <Text style={trackingNumber}>{trackingNumber}</Text>
        </Section>

        <Section style={trackingInfoRow}>
          <Text style={trackingLabel}>Estimated Delivery</Text>
          <Text style={deliveryDate}>{estimatedDelivery}</Text>
        </Section>

        <Section style={buttonSection}>
          <Button style={button} href={trackingUrls[carrier]}>
            Track Your Package
          </Button>
        </Section>
      </Section>

      <Hr style={hr} />

      <Section style={deliveryTipsSection}>
        <Heading style={h3}>Delivery Tips</Heading>
        <Text style={tipItem}>
          <strong>Monitor tracking:</strong> Check the tracking number above for real-time updates
        </Text>
        <Text style={tipItem}>
          <strong>Signature required:</strong> Some orders may require a signature at delivery
        </Text>
        <Text style={tipItem}>
          <strong>Package protection:</strong> Keep your tracking number for reference and insurance
        </Text>
        <Text style={tipItem}>
          <strong>Delivery window:</strong> Carriers typically deliver between 8am and 8pm
        </Text>
      </Section>

      <Hr style={hr} />

      {shippingAddress && (
        <>
          <Heading style={h3}>Shipping To</Heading>
          <Text style={addressText}>{shippingAddress}</Text>
          <Hr style={hr} />
        </>
      )}

      <Section style={supportSection}>
        <Heading style={h3}>Need Help?</Heading>
        <Text style={text}>
          If you have any questions about your delivery or need to contact us:
        </Text>
        <Text style={contactItem}>
          Email:{' '}
          <Link href="mailto:hello@liveiconic.com" style={link}>
            hello@liveiconic.com
          </Link>
        </Text>
        <Text style={contactItem}>
          Phone: <Link href="tel:+1-800-ICONIC-1" style={link}>+1-800-ICONIC-1</Link>
        </Text>
        <Text style={contactItem}>
          Help Center:{' '}
          <Link href="https://liveiconic.com/help/shipping" style={link}>
            Shipping & Delivery FAQs
          </Link>
        </Text>
      </Section>

      <Section style={excitementSection}>
        <Text style={excitementText}>
          We can't wait for you to experience your Live It Iconic pieces.
          Follow us on{' '}
          <Link href="https://instagram.com/liveiconic" style={link}>
            Instagram
          </Link>
          {' '}to see styling inspiration!
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

const hr = {
  borderColor: '#E6E9EF',
  margin: '24px 0',
  borderWidth: '1px',
  borderStyle: 'solid',
};

const trackingBox = {
  backgroundColor: '#f8f8f8',
  padding: '24px',
  borderRadius: '8px',
  margin: '24px 0',
  border: '2px solid #C1A060',
};

const trackingTitle = {
  color: '#0B0B0C',
  fontSize: '20px',
  fontWeight: 'bold',
  margin: '0 0 20px',
};

const trackingInfoRow = {
  margin: '0 0 16px',
};

const trackingLabel = {
  color: '#5C6270',
  fontSize: '13px',
  fontWeight: '600',
  textTransform: 'uppercase' as const,
  letterSpacing: '0.5px',
  margin: '0 0 4px',
};

const trackingValue = {
  color: '#0B0B0C',
  fontSize: '16px',
  margin: '0 0 8px',
};

const trackingNumber = {
  color: '#C1A060',
  fontSize: '20px',
  fontWeight: 'bold',
  margin: '0 0 8px',
  letterSpacing: '1px',
  fontFamily: 'monospace',
};

const deliveryDate = {
  color: '#4CAF50',
  fontSize: '16px',
  fontWeight: 'bold',
  margin: 0,
};

const buttonSection = {
  margin: '20px 0 0 0',
  textAlign: 'center' as const,
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

const deliveryTipsSection = {
  margin: '24px 0',
};

const tipItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 12px',
  paddingLeft: '16px',
};

const addressText = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '24px',
  margin: '0 0 16px',
  whiteSpace: 'pre-wrap' as const,
};

const supportSection = {
  backgroundColor: '#f0f8f0',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
};

const contactItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 8px',
};

const excitementSection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
  textAlign: 'center' as const,
};

const excitementText = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: 0,
};

const link = {
  color: '#C1A060',
  textDecoration: 'underline',
};
