import IconGrid from './showcase/IconGrid';
import { iconConcepts, brandmarkConcepts } from './showcase/iconShowcaseData';

const IconShowcase = () => {
  return (
    <section className="py-24 bg-gradient-to-b from-lii-ash/5 to-lii-charcoal/5">
      <div className="container mx-auto px-4">
        <IconGrid
          title="Icon Designs"
          description="Each icon embodies different aspects of the Live It Iconic philosophy"
          items={iconConcepts}
          gridCols="grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        />

        <div className="mb-32" />

        <IconGrid
          title="Brand Name Styles"
          description="Diverse typographic treatments for the complete brand name"
          items={brandmarkConcepts}
          isBrandmark
          gridCols="grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
        />
      </div>
    </section>
  );
};

export default IconShowcase;
