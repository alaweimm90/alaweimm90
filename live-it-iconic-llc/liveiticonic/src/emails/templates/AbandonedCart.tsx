import { Heading, Text, Button, Hr, Row, Column, Section, Link } from '@react-email/components';
import { EmailLayout } from '../components/EmailLayout';

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
  image?: string;
  color?: string;
  size?: string;
}

interface AbandonedCartProps {
  name: string;
  items: CartItem[];
  cartUrl: string;
  cartTotal: number;
  couponCode?: string;
  couponDiscount?: number;
}

export function AbandonedCart({
  name,
  items,
  cartUrl,
  cartTotal,
  couponCode = 'CART10',
  couponDiscount = 10,
}: AbandonedCartProps) {
  const savings = (cartTotal * couponDiscount) / 100;

  return (
    <EmailLayout preview="You left something behind">
      <Heading style={h1}>You Left Something Behind</Heading>
      <Text style={text}>Hi {name},</Text>
      <Text style={text}>
        We noticed you left some items in your cart. They're still waiting for you, and we'd love to help you complete your purchase!
      </Text>

      <Hr style={hr} />

      <Heading style={h3}>Items in Your Cart</Heading>
      {items.map((item) => (
        <Section key={item.id} style={itemRow}>
          <Row>
            {item.image && (
              <Column style={itemImageCol}>
                <img
                  src={item.image}
                  alt={item.name}
                  style={itemImage}
                  width="80"
                  height="80"
                />
              </Column>
            )}
            <Column style={itemDetailsCol}>
              <Text style={itemName}>{item.name}</Text>
              {item.color && (
                <Text style={itemSpec}>Color: {item.color}</Text>
              )}
              {item.size && (
                <Text style={itemSpec}>Size: {item.size}</Text>
              )}
              <Text style={itemSpec}>Quantity: {item.quantity}</Text>
            </Column>
            <Column align="right" style={itemPriceCol}>
              <Text style={itemPrice}>
                ${(item.price * item.quantity).toFixed(2)}
              </Text>
            </Column>
          </Row>
        </Section>
      ))}

      <Hr style={hr} />

      <Section style={totalSection}>
        <Row style={totalRow}>
          <Column align="right">
            <Text style={subtotalLabel}>Subtotal:</Text>
          </Column>
          <Column align="right" style={totalValue}>
            <Text style={subtotalValue}>${cartTotal.toFixed(2)}</Text>
          </Column>
        </Row>
      </Section>

      <Section style={offerSection}>
        <Heading style={offerTitle}>Limited Time Offer</Heading>
        <Text style={offerText}>
          Complete your purchase today and get <strong>{couponDiscount}% off</strong>!
        </Text>
        <Text style={couponText}>Use code: <span style={couponCode}>{couponCode}</span></Text>
        <Text style={savingsText}>
          You'll save approximately ${savings.toFixed(2)} on this order
        </Text>
      </Section>

      <Section style={ctaSection}>
        <Button style={button} href={cartUrl}>
          Complete Your Purchase
        </Button>
        <Text style={codeNote}>
          Code {couponCode} will be automatically applied at checkout
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={reasonSection}>
        <Heading style={h3}>Why Shop Live It Iconic?</Heading>
        <Text style={reasonItem}>
          <strong>Premium Quality:</strong> Precision-cut apparel crafted with automotive-inspired design
        </Text>
        <Text style={reasonItem}>
          <strong>Fast & Free Shipping:</strong> Orders over $100 ship free. Standard shipping is 3-5 business days
        </Text>
        <Text style={reasonItem}>
          <strong>30-Day Returns:</strong> Not satisfied? Return items within 30 days for a full refund
        </Text>
        <Text style={reasonItem}>
          <strong>Secure Checkout:</strong> Your payment information is protected with industry-standard encryption
        </Text>
      </Section>

      <Hr style={hr} />

      <Section style={urgencySection}>
        <Text style={urgencyText}>
          Don't wait! This offer is valid for 48 hours only. Your items are reserved, but stock is limited.
        </Text>
      </Section>

      <Section style={supportSection}>
        <Text style={text}>
          Having trouble with checkout? Contact us at{' '}
          <Link href="mailto:hello@liveiconic.com" style={link}>
            hello@liveiconic.com
          </Link>
          {' '}— we're here to help!
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

const itemRow = {
  margin: '0 0 20px',
  paddingBottom: '20px',
  borderBottom: '1px solid #f0f0f0',
};

const itemImageCol = {
  width: '100px',
  paddingRight: '16px',
  verticalAlign: 'top' as const,
};

const itemImage = {
  width: '80px',
  height: '80px',
  borderRadius: '4px',
  objectFit: 'cover' as const,
};

const itemDetailsCol = {
  verticalAlign: 'top' as const,
};

const itemName = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: '600',
  margin: '0 0 4px',
};

const itemSpec = {
  color: '#5C6270',
  fontSize: '14px',
  margin: '0 0 2px',
};

const itemPriceCol = {
  paddingLeft: '16px',
  verticalAlign: 'top' as const,
  minWidth: '80px',
};

const itemPrice = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: '600',
  margin: 0,
};

const totalSection = {
  margin: '16px 0',
};

const totalRow = {
  display: 'flex' as const,
  justifyContent: 'flex-end' as const,
};

const subtotalLabel = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: '600',
  paddingRight: '16px',
  margin: 0,
};

const totalValue = {
  minWidth: '80px',
  textAlign: 'right' as const,
};

const subtotalValue = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: '600',
  margin: 0,
};

const offerSection = {
  backgroundColor: '#FFF8E1',
  border: '2px solid #C1A060',
  padding: '24px',
  borderRadius: '8px',
  margin: '24px 0',
  textAlign: 'center' as const,
};

const offerTitle = {
  color: '#0B0B0C',
  fontSize: '20px',
  fontWeight: 'bold',
  margin: '0 0 12px',
};

const offerText = {
  color: '#0B0B0C',
  fontSize: '16px',
  lineHeight: '24px',
  margin: '0 0 16px',
};

const couponCode = {
  backgroundColor: '#0B0B0C',
  color: '#C1A060',
  padding: '8px 16px',
  borderRadius: '4px',
  fontWeight: 'bold',
  fontSize: '16px',
  letterSpacing: '1px',
  fontFamily: 'monospace',
  display: 'inline-block' as const,
};

const couponText = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: '600',
  margin: '0 0 12px',
};

const savingsText = {
  color: '#4CAF50',
  fontSize: '14px',
  fontWeight: '600',
  margin: 0,
};

const ctaSection = {
  textAlign: 'center' as const,
  margin: '24px 0',
};

const button = {
  backgroundColor: '#C1A060',
  color: '#0B0B0C',
  padding: '14px 40px',
  borderRadius: '6px',
  textDecoration: 'none',
  display: 'inline-block',
  fontWeight: '600',
  fontSize: '16px',
  cursor: 'pointer',
  border: 'none',
};

const codeNote = {
  color: '#5C6270',
  fontSize: '13px',
  margin: '12px 0 0 0',
};

const reasonSection = {
  margin: '24px 0',
};

const reasonItem = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '22px',
  margin: '0 0 12px',
  paddingLeft: '16px',
};

const urgencySection = {
  backgroundColor: '#FFE8E8',
  padding: '16px',
  borderRadius: '6px',
  margin: '24px 0',
  textAlign: 'center' as const,
};

const urgencyText = {
  color: '#0B0B0C',
  fontSize: '15px',
  fontWeight: '600',
  lineHeight: '22px',
  margin: 0,
};

const supportSection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
  textAlign: 'center' as const,
};

const link = {
  color: '#C1A060',
  textDecoration: 'underline',
};
