import heroImage from '@/assets/about-hero-athlete-gullwing.jpg';
import BirdLogoShowcase from '@/components/BirdLogoShowcase';
import Footer from '@/components/Footer';
import Navigation from '@/components/Navigation';
import SEO from '@/components/SEO';
import SecurityBadges from '@/components/SecurityBadges';
import Testimonials from '@/components/Testimonials';

const About = () => {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'AboutPage',
    name: 'About Live It Iconic',
    description:
      "Learn about Live It Iconic's philosophy of blending motorsport discipline with everyday refinement. Discover our design principles and commitment to precision, performance, and luxury.",
    url: 'https://liveiticonic.com/about',
    mainEntity: {
      '@type': 'Organization',
      name: 'Live It Iconic',
      description:
        'Premium lifestyle merchandise for luxury automotive enthusiasts and high-achievers.',
    },
  };

  return (
    <>
      <SEO
        title="About Live It Iconic - Automotive-Inspired Premium Apparel"
        description="Discover our design philosophy blending motorsport discipline with luxury refinement. Learn how we cut, finish, and test every piece for precision, performance, and sophisticated style."
        canonical="/about"
        ogImage="/og-about.jpg"
        ogImageAlt="Live It Iconic - About Our Brand"
        keywords="luxury apparel brand, automotive fashion, motorsport design, premium clothing philosophy, Live It Iconic story"
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
                backgroundImage: `url(${heroImage})`,
                filter:
                  'grayscale(0.2) contrast(1.25) brightness(0.85) saturate(0.85) sepia(0.12) hue-rotate(15deg)',
              }}
              role="img"
              aria-label="Athlete showcasing Live It Iconic apparel with luxury gullwing supercar in urban setting"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-lii-black/95 via-lii-black/80 to-lii-black/90"></div>
            </div>

            <div className="relative z-10 h-full flex items-center justify-center px-4 sm:px-6 pt-20">
              <div className="container mx-auto">
                <div className="max-w-4xl mx-auto text-center">
                  <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-light tracking-tight mb-6 hero-text-reveal text-lii-cloud">
                    About Live It Iconic
                  </h1>
                  <p className="text-lg sm:text-xl font-ui text-lii-ash mb-4 hero-subtitle-reveal">
                    Confidence, in uniform
                  </p>
                  <p className="text-base sm:text-lg font-ui text-lii-ash hero-cta-reveal max-w-2xl mx-auto leading-relaxed">
                    Statement pieces for bold days.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Brand Story */}
          <section className="py-20 sm:py-32 bg-gradient-to-b from-lii-black to-lii-ink">
            <div className="container mx-auto px-6">
              <div className="max-w-4xl mx-auto">
                <div className="grid lg:grid-cols-2 gap-16 items-center">
                  <div>
                    <h2 className="text-3xl sm:text-4xl font-display font-light mb-8 text-lii-ash">
                      Our Philosophy
                    </h2>
                    <div className="space-y-6 text-lii-ash font-ui leading-relaxed">
                      <p className="text-lg">
                        We cut apparel with the discipline of performance machines—clean lines,
                        durable fabrics, precise fits. Designed for motion; refined for nights out.
                      </p>
                      <p>
                        Our design philosophy blends motorsport discipline with everyday refinement.
                        Learn how we cut, finish, and test our pieces.
                      </p>
                      <p>
                        Every element—from the geometric logo to the curated color palette—reflects
                        a commitment to sophisticated simplicity and purposeful design.
                      </p>
                    </div>
                  </div>

                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-lii-gold/20 to-lii-champagne/10 rounded-2xl blur-xl"></div>
                    <div className="relative bg-lii-charcoal/30 p-8 rounded-2xl border border-lii-gold/20">
                      <h3 className="text-xl font-display text-lii-gold mb-6 tracking-[0.2em]">
                        DESIGN PRINCIPLES
                      </h3>
                      <ul className="space-y-4 text-foreground/60 font-ui font-light">
                        <li className="flex items-start gap-3">
                          <div className="w-1.5 h-1.5 bg-lii-gold rounded-full mt-2 flex-shrink-0"></div>
                          <span>Precision in every detail</span>
                        </li>
                        <li className="flex items-start gap-3">
                          <div className="w-1.5 h-1.5 bg-lii-champagne rounded-full mt-2 flex-shrink-0"></div>
                          <span>Performance-driven aesthetics</span>
                        </li>
                        <li className="flex items-start gap-3">
                          <div className="w-1.5 h-1.5 bg-lii-bronze rounded-full mt-2 flex-shrink-0"></div>
                          <span>Timeless luxury appeal</span>
                        </li>
                        <li className="flex items-start gap-3">
                          <div className="w-1.5 h-1.5 bg-lii-gold rounded-full mt-2 flex-shrink-0"></div>
                          <span>Sustainable innovation focus</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Values */}
          <section className="py-20 sm:py-32 bg-lii-ink">
            <div className="container mx-auto px-6">
              <div className="max-w-6xl mx-auto">
                <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-light text-center mb-16 text-lii-ash">
                  Core Values
                </h2>

                <div className="grid md:grid-cols-3 gap-8">
                  {[
                    {
                      title: 'PRECISION',
                      description:
                        'Every detail crafted with meticulous attention to form and function, ensuring excellence in design execution.',
                      accent: 'lii-gold',
                    },
                    {
                      title: 'PERFORMANCE',
                      description:
                        'Design that enhances movement and supports an active lifestyle without compromising on aesthetic appeal.',
                      accent: 'lii-champagne',
                    },
                    {
                      title: 'LUXURY',
                      description:
                        'Premium materials and sophisticated styling that elevate everyday athletic wear to new heights.',
                      accent: 'lii-bronze',
                    },
                  ].map((value, index) => (
                    <div key={index} className="group text-center">
                      <div className="relative mb-6">
                        <div
                          className={`w-16 h-16 mx-auto border-2 border-${value.accent} rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-500`}
                        >
                          <div className={`w-3 h-3 bg-${value.accent} rounded-full`}></div>
                        </div>
                      </div>
                      <h3 className="text-xl font-display font-light text-lii-ash mb-4 tracking-[0.3em]">
                        {value.title}
                      </h3>
                      <p className="text-foreground/60 font-ui font-light leading-relaxed">
                        {value.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Heritage & Craftsmanship */}
          <section className="py-20 sm:py-32 bg-gradient-to-b from-lii-ink to-lii-black">
            <div className="container mx-auto px-6">
              <div className="max-w-6xl mx-auto">
                <div className="grid lg:grid-cols-2 gap-16 items-center">
                  <div>
                    <h2 className="text-3xl sm:text-4xl font-display font-light mb-8 text-lii-ash">
                      Heritage & Craftsmanship
                    </h2>
                    <div className="space-y-6 text-lii-ash font-ui leading-relaxed">
                      <p className="text-lg">
                        Born from a passion for automotive excellence and refined craftsmanship,
                        Live It Iconic represents the perfect synthesis of high-performance quality
                        and sophisticated design.
                      </p>
                      <p>
                        Our pieces are crafted with the same precision that defines
                        championship-winning machines. Every stitch, every seam, every detail is
                        executed with meticulous accuracy and premium quality standards.
                      </p>
                      <p>
                        We source premium fabrics from heritage mills and work with master craftsmen
                        who understand that true luxury lies in the invisible details—the perfect
                        tension of a seam, the subtle drape of premium fabric, the thoughtful
                        placement of every element.
                      </p>
                    </div>
                  </div>

                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-lii-champagne/20 to-lii-gold/10 rounded-2xl blur-xl"></div>
                    <div className="relative bg-lii-charcoal/30 p-8 rounded-2xl border border-lii-champagne/20">
                      <h3 className="text-xl font-display text-lii-champagne mb-6 tracking-[0.2em]">
                        CRAFTSMANSHIP STANDARDS
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-lii-champagne rounded-full"></div>
                          <span className="text-foreground/70 font-ui text-sm">
                            Triple-stitched reinforced seams
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-lii-champagne rounded-full"></div>
                          <span className="text-foreground/70 font-ui text-sm">
                            Premium YKK zippers and hardware
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-lii-champagne rounded-full"></div>
                          <span className="text-foreground/70 font-ui text-sm">
                            Italian and Japanese fabric sourcing
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-lii-champagne rounded-full"></div>
                          <span className="text-foreground/70 font-ui text-sm">
                            Hand-finished details and inspections
                          </span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-lii-champagne rounded-full"></div>
                          <span className="text-foreground/70 font-ui text-sm">
                            Sustainable dyeing and finishing
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Vision */}
          <section className="py-20 sm:py-32 bg-lii-black">
            <div className="container mx-auto px-6">
              <div className="max-w-4xl mx-auto text-center">
                <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-light mb-12 text-lii-ash">
                  Our Vision
                </h2>

                <div className="space-y-8 text-lii-ash font-ui leading-relaxed">
                  <p className="text-xl sm:text-2xl font-light">
                    To redefine athletic luxury by creating pieces that transcend the boundaries
                    between performance and elegance.
                  </p>

                  <div className="grid md:grid-cols-2 gap-8 mt-16">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-lii-gold/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-lii-gold/20">
                        <div className="w-3 h-3 bg-lii-gold rounded-full"></div>
                      </div>
                      <h3 className="text-lg font-display text-lii-gold mb-3">For the Bold</h3>
                      <p className="text-foreground/70 font-ui text-sm leading-relaxed">
                        Statement pieces that command attention and respect, designed for those who
                        lead with confidence and purpose.
                      </p>
                    </div>

                    <div className="text-center">
                      <div className="w-16 h-16 bg-lii-champagne/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-lii-champagne/20">
                        <div className="w-3 h-3 bg-lii-champagne rounded-full"></div>
                      </div>
                      <h3 className="text-lg font-display text-lii-champagne mb-3">
                        For the Driven
                      </h3>
                      <p className="text-foreground/70 font-ui text-sm leading-relaxed">
                        Performance-driven design that supports an active lifestyle without
                        compromise, engineered for those who push boundaries.
                      </p>
                    </div>
                  </div>

                  <div className="mt-16 pt-8 border-t border-lii-gold/10">
                    <p className="text-lg font-ui font-light italic text-lii-cloud/80">
                      "Live It Iconic is not just apparel—it's a mindset. A commitment to excellence
                      in every moment, every movement, every decision."
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Bird Logo Showcase */}
          <section className="py-20 sm:py-32 bg-gradient-to-b from-lii-black to-lii-ink">
            <div className="container mx-auto px-6">
              <BirdLogoShowcase variant="full" />
            </div>
          </section>

          {/* Trust Signals */}
          <section className="py-16 bg-lii-ink">
            <div className="container mx-auto px-6">
              <div className="max-w-6xl mx-auto">
                <SecurityBadges />
              </div>
            </div>
          </section>

          {/* Testimonials */}
          <Testimonials limit={3} />
        </main>
        <Footer />
      </div>
    </>
  );
};

export default About;
