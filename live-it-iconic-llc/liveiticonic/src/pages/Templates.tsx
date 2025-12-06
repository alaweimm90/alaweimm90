import Footer from '@/components/Footer';
import IconShowcase from '@/components/IconShowcase';
import Navigation from '@/components/Navigation';

const Templates = () => {
  return (
    <div className="min-h-screen bg-lii-bg text-foreground font-ui overflow-x-hidden relative">
      <Navigation />

      <main>
        {/* Hero Section */}
        <section className="relative min-h-[60vh] w-full overflow-hidden pt-20">
          <div className="absolute inset-0 bg-gradient-to-br from-lii-bg via-lii-ink to-lii-bg">
            {/* Subtle animated background */}
            <div className="absolute inset-0 opacity-10">
              <div
                className="absolute top-1/4 left-1/4 w-96 h-96 bg-lii-gold rounded-full blur-3xl animate-pulse"
                style={{ animationDuration: '8s' }}
              ></div>
              <div
                className="absolute bottom-1/3 right-1/4 w-80 h-80 bg-lii-gold/50 rounded-full blur-3xl animate-pulse"
                style={{ animationDuration: '12s', animationDelay: '4s' }}
              ></div>
            </div>
          </div>

          <div className="relative z-10 h-full flex items-center justify-center px-4 sm:px-6 pt-20">
            <div className="container mx-auto">
              <div className="max-w-4xl mx-auto text-center">
                <h1 className="text-4xl sm:text-5xl md:text-6xl font-display tracking-tight mb-6 text-lii-cloud">
                  Design Templates
                </h1>
                <p className="text-xl sm:text-2xl font-ui text-lii-ash mb-8 max-w-2xl mx-auto leading-relaxed">
                  Icon designs and brand name styles for Live It Iconic
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Icon Showcase Component */}
        <IconShowcase />
      </main>

      <Footer />
    </div>
  );
};

export default Templates;
