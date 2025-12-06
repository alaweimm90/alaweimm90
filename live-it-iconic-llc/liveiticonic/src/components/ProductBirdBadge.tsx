import { ElegantFlamingo, MajesticPelican, CaribbeanFrigateBird, TropicalTanager } from './logo';

interface ProductBirdBadgeProps {
  birdType?: 'flamingo' | 'pelican' | 'frigate' | 'tanager';
  size?: number;
  className?: string;
  showLabel?: boolean;
}

const birdComponents = {
  flamingo: { component: ElegantFlamingo, label: 'Elegant Flamingo' },
  pelican: { component: MajesticPelican, label: 'Majestic Pelican' },
  frigate: { component: CaribbeanFrigateBird, label: 'Caribbean Frigate' },
  tanager: { component: TropicalTanager, label: 'Tropical Tanager' },
};

/**
 * ProductBirdBadge component displays Caribbean bird logo badges on products
 *
 * Renders one of four bird logo variants (flamingo, pelican, frigate, tanager) with
 * optional label text. Includes hover effects with scale, rotation, and drop shadow animations.
 * Integrates Live It Iconic bird branding into product presentations.
 *
 * @component
 * @param {ProductBirdBadgeProps} props - Component props
 * @param {'flamingo' | 'pelican' | 'frigate' | 'tanager'} [props.birdType] - Bird logo to display
 * @param {number} [props.size] - SVG size in pixels (default: 32)
 * @param {string} [props.className] - Additional CSS classes
 * @param {boolean} [props.showLabel] - Display bird name label below icon
 *
 * @example
 * <ProductBirdBadge birdType="flamingo" size={32} showLabel={true} />
 *
 * @remarks
 * - Hover effects: scale(110%), rotate(6deg), gold glow shadow
 * - Bird colors default to brand gold
 * - Smooth transitions (500ms)
 * - Flex layout with gap between icon and label
 */
export const ProductBirdBadge: React.FC<ProductBirdBadgeProps> = ({
  birdType = 'flamingo',
  size = 32,
  className = '',
  showLabel = false,
}) => {
  const { component: Bird, label } = birdComponents[birdType];

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="transition-all duration-500 hover:scale-110 hover:rotate-6 hover:drop-shadow-[0_0_16px_rgba(212,175,55,0.6)]">
        <Bird size={size} color="hsl(var(--lii-gold))" />
      </div>
      {showLabel && (
        <span className="text-xs font-ui text-lii-gold tracking-wider uppercase">{label}</span>
      )}
    </div>
  );
};

export default ProductBirdBadge;
