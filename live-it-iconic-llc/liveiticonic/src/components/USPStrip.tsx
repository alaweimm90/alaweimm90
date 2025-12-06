import { Award, Zap, Headphones } from 'lucide-react';

const features = [
  {
    icon: Award,
    title: 'Performance-Grade Fabrics',
    description: 'Materials tested at the limit, engineered for every environment',
  },
  {
    icon: Zap,
    title: 'Precision Engineering',
    description: 'Cut for movement, built for endurance, crafted for confidence',
  },
  {
    icon: Headphones,
    title: 'Track-Speed Support',
    description: 'Immediate response when performance matters most',
  },
];

const USPStrip = () => {
  return (
    <section className="py-16 bg-card">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16 animate-fade-in-up">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div
              className="w-16 h-0.5 bg-accent animate-scale-in"
              style={{ animationDelay: '0.2s' }}
            />
            <span className="text-accent font-semibold text-sm uppercase tracking-wider">
              Performance Promise
            </span>
            <div
              className="w-16 h-0.5 bg-accent animate-scale-in"
              style={{ animationDelay: '0.4s' }}
            />
          </div>
          <h2 className="text-3xl md:text-4xl font-headline font-bold mb-6 text-gradient">
            Built different, performs better
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Every thread, every stitch, every detail engineered to exceed expectations when it
            matters most
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
          {features.map((feature, index) => (
            <div
              key={index}
              className="text-center group hover-lift animate-fade-in-up"
              style={{ animationDelay: `${index * 0.15}s` }}
            >
              <div className="inline-flex items-center justify-center w-20 h-20 bg-accent/10 group-hover:bg-accent/20 rounded-full mb-6 transition-all duration-300 group-hover:shadow-lg">
                <feature.icon className="w-10 h-10 text-accent group-hover:scale-110 transition-transform duration-300" />
              </div>
              <h3 className="text-xl font-headline font-semibold mb-3 group-hover:text-accent transition-colors duration-300">
                {feature.title}
              </h3>
              <p className="text-muted-foreground group-hover:text-foreground transition-colors duration-300 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default USPStrip;
