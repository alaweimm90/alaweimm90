import { ReactNode } from 'react';

interface IconCardProps {
  name: string;
  concept: string;
  meaning: string;
  children: ReactNode;
  isBrandmark?: boolean;
}

const IconCard: React.FC<IconCardProps> = ({
  name,
  concept,
  meaning,
  children,
  isBrandmark = false,
}) => {
  return (
    <div className="group relative p-8 bg-lii-ink/20 border border-lii-charcoal/30 rounded-lg hover:border-lii-gold/50 transition-all duration-500 hover:shadow-lg hover:shadow-lii-gold/10">
      <div className="flex flex-col items-center text-center space-y-6">
        <div className={isBrandmark ? 'w-full flex items-center justify-center min-h-[100px]' : 'w-16 h-16 flex items-center justify-center'}>
          {children}
        </div>

        <div className="space-y-3">
          <h3 className="text-xl font-medium text-lii-ash font-headline">{name}</h3>
          <p className="text-sm text-lii-gold font-ui font-medium">{concept}</p>
          <p className="text-sm text-lii-silver font-ui leading-relaxed">{meaning}</p>
        </div>
      </div>

      <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-lii-gold/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
    </div>
  );
};

export default IconCard;
