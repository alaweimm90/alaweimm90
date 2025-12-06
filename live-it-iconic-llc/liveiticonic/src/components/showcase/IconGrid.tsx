import { ComponentType } from 'react';
import IconCard from './IconCard';
import { ConceptItem } from './iconShowcaseData';

interface IconGridProps {
  title: string;
  description: string;
  items: ConceptItem[];
  isBrandmark?: boolean;
  gridCols?: string;
}

const IconGrid: React.FC<IconGridProps> = ({
  title,
  description,
  items,
  isBrandmark = false,
  gridCols = 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
}) => {
  return (
    <div>
      <div className="text-center mb-16">
        <h2 className="text-4xl font-light font-headline text-lii-ash mb-4">{title}</h2>
        <p className="text-lg text-lii-silver font-ui max-w-2xl mx-auto">{description}</p>
      </div>

      <div className={`grid ${gridCols} ${isBrandmark ? 'gap-12' : 'gap-8'} max-w-7xl mx-auto`}>
        {items.map((item, index) => {
          const Component = item.component as ComponentType<{ className?: string }>;
          return (
            <IconCard
              key={index}
              name={item.name}
              concept={item.concept}
              meaning={item.meaning}
              isBrandmark={isBrandmark}
            >
              <Component className={isBrandmark ? 'w-full h-auto' : 'w-full h-full'} />
            </IconCard>
          );
        })}
      </div>
    </div>
  );
};

export default IconGrid;
