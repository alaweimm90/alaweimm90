import { Heading, Text, Hr, Row, Column, Section, Link } from '@react-email/components';
import { EmailLayout } from '../components/EmailLayout';

interface OrderItem {
  name: string;
  quantity: number;
  price: number;
  image?: string;
  sku?: string;
}

interface OrderConfirmationProps {
  orderId: string;
  name: string;
  items: OrderItem[];
  subtotal: number;
  shipping: number;
  tax?: number;
  total: number;
  shippingAddress: string;
  estimatedDelivery?: string;
}

export function OrderConfirmation({
  orderId,
  name,
  items,
  subtotal,
  shipping,
  tax,
  total,
  shippingAddress,
  estimatedDelivery,
}: OrderConfirmationProps) {
  return (
    <EmailLayout preview={`Order #${orderId} confirmed`}>
      <Heading style={h1}>Order Confirmed</Heading>
      <Text style={text}>Hi {name},</Text>
      <Text style={text}>
        Thank you for your order! We're preparing your items for shipment. You'll receive a shipping notification with tracking information once your order ships.
      </Text>

      <Section style={orderNumberSection}>
        <Text style={orderNumber}>Order #{orderId}</Text>
        <Text style={orderDate}>
          Placed on {new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </Text>
      </Section>

      <Hr style={hr} />

      <Heading style={h3}>Order Items</Heading>
      {items.map((item, i) => (
        <Section key={i} style={itemRow}>
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
              {item.sku && (
                <Text style={itemSku}>SKU: {item.sku}</Text>
              )}
              <Text style={itemQty}>Quantity: {item.quantity}</Text>
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

      <Section style={totalsSection}>
        <Row style={totalRow}>
          <Column align="right" style={totalLabel}>
            <Text style={labelText}>Subtotal:</Text>
            <Text style={labelText}>Shipping:</Text>
            {tax && tax > 0 && (
              <Text style={labelText}>Tax:</Text>
            )}
            <Text style={boldLabel}>Total:</Text>
          </Column>
          <Column align="right" style={totalValue}>
            <Text style={valueText}>${subtotal.toFixed(2)}</Text>
            <Text style={valueText}>${shipping.toFixed(2)}</Text>
            {tax && tax > 0 && (
              <Text style={valueText}>${tax.toFixed(2)}</Text>
            )}
            <Text style={boldValue}>${total.toFixed(2)}</Text>
          </Column>
        </Row>
      </Section>

      <Hr style={hr} />

      <Heading style={h3}>Shipping Address</Heading>
      <Text style={addressText}>{shippingAddress}</Text>

      {estimatedDelivery && (
        <>
          <Hr style={hr} />
          <Section style={deliverySection}>
            <Text style={deliveryLabel}>Estimated Delivery:</Text>
            <Text style={deliveryDate}>{estimatedDelivery}</Text>
          </Section>
        </>
      )}

      <Hr style={hr} />

      <Section style={nextStepsSection}>
        <Heading style={h3}>What's Next?</Heading>
        <Text style={stepItem}>
          <strong>1. Order Confirmation:</strong> You'll receive a confirmation email shortly.
        </Text>
        <Text style={stepItem}>
          <strong>2. Processing:</strong> We'll prepare and pack your order within 24 hours.
        </Text>
        <Text style={stepItem}>
          <strong>3. Shipping:</strong> You'll get a shipping notification with tracking information.
        </Text>
        <Text style={stepItem}>
          <strong>4. Delivery:</strong> Receive your order and enjoy your Live It Iconic collection!
        </Text>
      </Section>

      <Section style={supportSection}>
        <Text style={text}>
          Questions about your order? Contact us at{' '}
          <Link href="mailto:hello@liveiconic.com" style={link}>
            hello@liveiconic.com
          </Link>
          {' '}or{' '}
          <Link href="https://liveiconic.com/help" style={link}>
            visit our help center
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

const hr = {
  borderColor: '#E6E9EF',
  margin: '24px 0',
  borderWidth: '1px',
  borderStyle: 'solid',
};

const orderNumberSection = {
  backgroundColor: '#f8f8f8',
  padding: '20px',
  borderRadius: '6px',
  margin: '24px 0',
  textAlign: 'center' as const,
};

const orderNumber = {
  color: '#C1A060',
  fontSize: '24px',
  fontWeight: 'bold',
  margin: '0 0 8px',
  letterSpacing: '1px',
};

const orderDate = {
  color: '#5C6270',
  fontSize: '14px',
  margin: 0,
};

const itemRow = {
  margin: '16px 0',
};

const itemImageCol = {
  width: '100px',
  paddingRight: '16px',
};

const itemImage = {
  width: '80px',
  height: '80px',
  borderRadius: '4px',
  objectFit: 'cover',
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

const itemSku = {
  color: '#5C6270',
  fontSize: '13px',
  margin: '0 0 4px',
};

const itemQty = {
  color: '#5C6270',
  fontSize: '14px',
  margin: 0,
};

const itemPriceCol = {
  paddingLeft: '16px',
  verticalAlign: 'top' as const,
};

const itemPrice = {
  color: '#0B0B0C',
  fontSize: '16px',
  fontWeight: '600',
  margin: 0,
};

const totalsSection = {
  margin: '24px 0',
};

const totalRow = {
  display: 'flex' as const,
  justifyContent: 'flex-end' as const,
};

const totalLabel = {
  width: '120px',
  paddingRight: '16px',
  textAlign: 'right' as const,
};

const labelText = {
  color: '#5C6270',
  fontSize: '14px',
  margin: '4px 0',
};

const boldLabel = {
  color: '#0B0B0C',
  fontSize: '14px',
  fontWeight: 'bold',
  margin: '8px 0 0 0',
  borderTop: '1px solid #E6E9EF',
  paddingTop: '8px',
};

const totalValue = {
  width: '100px',
  textAlign: 'right' as const,
};

const valueText = {
  color: '#0B0B0C',
  fontSize: '14px',
  margin: '4px 0',
};

const boldValue = {
  color: '#C1A060',
  fontSize: '16px',
  fontWeight: 'bold',
  margin: '8px 0 0 0',
  borderTop: '1px solid #E6E9EF',
  paddingTop: '8px',
};

const addressText = {
  color: '#0B0B0C',
  fontSize: '15px',
  lineHeight: '24px',
  margin: '0 0 16px',
  whiteSpace: 'pre-wrap' as const,
};

const deliverySection = {
  backgroundColor: '#f0f8f0',
  padding: '16px',
  borderRadius: '6px',
  margin: '16px 0',
};

const deliveryLabel = {
  color: '#0B0B0C',
  fontSize: '14px',
  fontWeight: '600',
  margin: '0 0 4px',
};

const deliveryDate = {
  color: '#4CAF50',
  fontSize: '16px',
  fontWeight: 'bold',
  margin: 0,
};

const nextStepsSection = {
  margin: '24px 0',
};

const stepItem = {
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
