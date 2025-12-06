import { Shield, Lock, CreditCard, Truck } from 'lucide-react';

const SecurityBadges = () => {
  const badges = [
    {
      icon: Shield,
      title: 'SSL Secured',
      description: '256-bit encryption',
    },
    {
      icon: Lock,
      title: 'Secure Payments',
      description: 'PCI DSS compliant',
    },
    {
      icon: CreditCard,
      title: 'Accepted Cards',
      description: 'Visa, Mastercard, Amex',
    },
    {
      icon: Truck,
      title: 'Fast Shipping',
      description: 'Free over $100',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {badges.map((badge, index) => (
        <div
          key={index}
          className="flex flex-col items-center text-center p-4 bg-lii-ink/50 border border-lii-gold/10 rounded-lg hover:bg-lii-ink/70 transition-colors duration-300"
        >
          <badge.icon className="w-8 h-8 text-lii-gold mb-2" />
          <h4 className="text-lii-cloud font-ui font-medium text-sm mb-1">{badge.title}</h4>
          <p className="text-lii-ash font-ui text-xs">{badge.description}</p>
        </div>
      ))}
    </div>
  );
};

export default SecurityBadges;
