import lifestyleImage1 from '@/assets/lifestyle-athlete-gullwing-car.jpg';
import lifestyleImage2 from '@/assets/lifestyle-cobblestone-supercar.jpg';
import BirdLogoShowcase from '@/components/BirdLogoShowcase';
import Footer from '@/components/Footer';
import Navigation from '@/components/Navigation';
import SEO from '@/components/SEO';

const Lifestyle = () => {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name: 'Lifestyle | Live It Iconic',
    description:
      'Explore the Live It Iconic lifestyle - where motorsport precision meets everyday sophistication. Discover stories of performance, urban elegance, and bold living.',
    url: 'https://liveiticonic.com/lifestyle',
    mainEntity: {
      '@type': 'Article',
      headline: 'Live It Iconic Lifestyle Stories',
      description:
        'Featured stories showcasing the blend of motorsport discipline and modern sophistication in athletic apparel design',
      articleSection: 'Lifestyle',
      about: [
        {
          '@type': 'Thing',
          name: 'Performance Precision',
          description:
            'Clean lines meet durable fabrics engineered for motion and refined for every setting',
        },
        {
          '@type': 'Thing',
          name: 'Urban Sophistication',
          description:
            'Timeless materials and innovative construction for modern athletes navigating diverse environments',
        },
      ],
    },
  };

  return (
    <>
      <SEO
        title="Lifestyle | Live It Iconic"
        description="Explore the Live It Iconic lifestyle - where motorsport precision meets everyday sophistication. Discover stories of performance, urban elegance, and bold living."
        canonical="/lifestyle"
        structuredData={structuredData}
      />
      <div className="min-h-screen bg-lii-bg text-foreground font-ui overflow-x-hidden relative">
        <Navigation />
        <main>
          {/* Hero Section */}
          <section className="relative h-screen w-full overflow-hidden">
            <div
              className="absolute inset-0 bg-cover bg-center bg-no-repeat"
              style={{
                backgroundImage: `url(${lifestyleImage1})`,
                filter:
                  'grayscale(0.2) contrast(1.25) brightness(0.85) saturate(0.85) sepia(0.12) hue-rotate(15deg)',
              }}
              role="img"
              aria-label="Athlete in performance wear standing beside luxury gull-wing supercar on scenic mountain road"
            >
              <div
                className="absolute inset-0 bg-gradient-to-br from-lii-black/90 via-lii-black/70 to-lii-black/85"
                aria-hidden="true"
              ></div>
            </div>

            <div className="relative z-10 h-full flex items-center justify-center px-4 sm:px-6 pt-20">
              <div className="container mx-auto">
                <div className="max-w-4xl mx-auto text-center">
                  <h1 className="text-4xl sm:text-5xl md:text-6xl font-display tracking-tight mb-6 hero-text-reveal text-lii-cloud">
                    Lifestyle
                  </h1>
                  <p className="text-xl sm:text-2xl font-ui text-lii-ash mb-8 hero-subtitle-reveal max-w-2xl mx-auto leading-relaxed">
                    Bold days, lived iconic.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Featured Stories */}
          <section className="py-20 sm:py-32 bg-gradient-to-b from-lii-black to-lii-ink">
            <div className="container mx-auto px-6">
              <div className="max-w-6xl mx-auto">
                <h2 className="text-3xl sm:text-4xl font-display mb-16 text-center text-lii-gold">
                  Featured Stories
                </h2>

                <div className="space-y-20">
                  {/* Story 1 */}
                  <article className="grid lg:grid-cols-2 gap-12 items-center">
                    <div className="relative group">
                      <div className="absolute inset-0 bg-gradient-to-br from-lii-gold/20 to-transparent rounded-2xl blur-xl transition-all duration-700 group-hover:blur-2xl"></div>
                      <img
                        src={lifestyleImage1}
                        alt="Athlete in performance wear standing beside black gull-wing coupe on mountain road at dusk with valley views"
                        width="800"
                        height="600"
                        className="relative w-full h-96 object-cover rounded-card"
                        loading="lazy"
                        style={{
                          filter:
                            'grayscale(0.2) contrast(1.25) brightness(0.88) saturate(0.85) sepia(0.12) hue-rotate(15deg)',
                        }}
                      />
                    </div>
                    <div>
                      <h3 className="text-2xl sm:text-3xl font-ui font-semibold mb-6 text-lii-gold">
                        Performance Precision
                      </h3>
                      <p className="text-lii-ash font-ui leading-relaxed mb-6 text-lg">
                        Clean lines meet durable fabrics. Precision fits engineered for motion,
                        refined for every setting.
                      </p>
                      <p className="text-lii-ash font-ui leading-relaxed mb-8">
                        From the discipline of high-performance machines to the silhouettes of
                        modern apparel—functional beauty, purposeful design.
                      </p>
                    </div>
                  </article>

                  {/* Story 2 */}
                  <article className="grid lg:grid-cols-2 gap-12 items-center">
                    <div className="lg:order-2 relative group">
                      <div className="absolute inset-0 bg-gradient-to-br from-lii-champagne/20 to-transparent rounded-2xl blur-xl transition-all duration-700 group-hover:blur-2xl"></div>
                      <img
                        src={lifestyleImage2}
                        alt="Red supercar on European cobblestone street with historic architecture and evening ambient lighting"
                        width="800"
                        height="600"
                        className="relative w-full h-96 object-cover rounded-card"
                        loading="lazy"
                        style={{
                          filter:
                            'grayscale(0.2) contrast(1.25) brightness(0.88) saturate(0.85) sepia(0.12) hue-rotate(15deg)',
                        }}
                      />
                    </div>
                    <div className="lg:order-1">
                      <h3 className="text-2xl sm:text-3xl font-ui font-semibold mb-6 text-lii-gold">
                        Urban Sophistication
                      </h3>
                      <p className="text-lii-ash font-ui leading-relaxed mb-6 text-lg">
                        Timeless materials meet innovative construction, creating pieces that
                        transition from performance activities to social settings.
                      </p>
                      <p className="text-lii-ash font-ui leading-relaxed mb-8">
                        Designed for the modern athlete who navigates diverse environments with
                        confidence.
                      </p>
                    </div>
                  </article>
                </div>
              </div>
            </div>
          </section>

          {/* Bird Logo Carousel */}
          <section className="py-20 sm:py-32 bg-lii-ink">
            <div className="container mx-auto px-6">
              <h2 className="text-3xl sm:text-4xl font-display mb-12 text-center text-lii-gold">
                Our Caribbean Heritage
              </h2>
              <BirdLogoShowcase variant="carousel" />
            </div>
          </section>
        </main>
        <Footer />
      </div>
    </>
  );
};

export default Lifestyle;
