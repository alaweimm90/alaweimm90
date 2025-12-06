import { Perspective } from '@/constants/podcastPerspectives';

interface PerspectiveCardProps {
  perspective: Perspective;
}

export const PerspectiveCard = ({ perspective }: PerspectiveCardProps) => (
  <div className="relative p-8 rounded-xl border border-lii-gold/10 bg-gradient-to-br from-lii-charcoal/20 to-lii-ink/40 backdrop-blur-sm hover:border-lii-gold/40 hover:shadow-[0_0_30px_rgba(193,160,96,0.2)] transition-all duration-500 group">
    <div className="absolute top-0 right-0 w-16 h-16 border-r border-t border-lii-gold/20 rounded-tr-xl group-hover:border-lii-gold/50 transition-colors duration-500"></div>
    <div className="absolute top-0 right-0 w-2 h-2 bg-lii-gold/40 rounded-full group-hover:scale-150 group-hover:bg-lii-gold/60 transition-all duration-500"></div>

    <div className="relative z-10">
      <div className="mb-6 flex items-center justify-center">{perspective.icon}</div>
      <h3 className="text-xl font-display font-light text-lii-gold mb-2">
        {perspective.title}
      </h3>
      <div className="w-12 h-px bg-lii-gold/40 mb-4"></div>
      <h4 className="text-base font-ui font-semibold text-lii-cloud mb-3">
        {perspective.name}
      </h4>
      <p className="text-sm font-ui font-light text-lii-ash leading-relaxed">
        {perspective.description}
      </p>
    </div>
  </div>
);
