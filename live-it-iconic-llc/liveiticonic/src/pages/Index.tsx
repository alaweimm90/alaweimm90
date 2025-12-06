import BirdLogoShowcase from '@/components/BirdLogoShowcase';
import BrandManifesto from '@/components/BrandManifesto';
import ContentCreatorHub from '@/components/ContentCreatorHub';
import EmailCapture from '@/components/EmailCapture';
import Footer from '@/components/Footer';
import Hero from '@/components/Hero';
import LifestyleTeaser from '@/components/LifestyleTeaser';
import MinimalShowcase from '@/components/MinimalShowcase';
import Navigation from '@/components/Navigation';
import PodcastShowcase from '@/components/PodcastShowcase';
import ProductShowcase from '@/components/ProductShowcase';
import ScrollProgress from '@/components/ScrollProgress';
import SEO from '@/components/SEO';
import SkipLinks from '@/components/SkipLinks';
import SupercarGallery from '@/components/SupercarGallery';
import TrustBar from '@/components/TrustBar';
import TwitchLiveIntegration from '@/components/TwitchLiveIntegration';
import YouTubeIntegration from '@/components/YouTubeIntegration';

const Index = () => {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'Organization',
    '@id': 'https://liveiticonic.com/#organization',
    name: 'Live It Iconic',
    url: 'https://liveiticonic.com',
    logo: 'https://liveiticonic.com/logo.png',
    image: 'https://liveiticonic.com/og-home.jpg',
    description:
      'Precision-cut apparel inspired by automotive discipline. Statement pieces for bold days—luxury apparel with clean lines, durable materials, and precise fits.',
    sameAs: [
      'https://instagram.com/liveiticonic',
      'https://twitter.com/liveiticonic',
      'https://linkedin.com/company/liveiticonic',
    ],
    contactPoint: {
      '@type': 'ContactPoint',
      contactType: 'Customer Service',
      email: 'hello@liveiticonic.com',
    },
  };

  return (
    <>
      <SEO
        title="Premium Apparel Inspired by Automotive Excellence"
        description="Precision-cut luxury apparel with clean lines, durable materials, and motorsport-inspired design. Built for motion; refined for nights out. Shop Live It Iconic."
        canonical="/"
        ogImage="/og-home.jpg"
        ogImageAlt="Live It Iconic - Premium Apparel Collection"
        keywords="premium apparel, automotive inspired clothing, luxury t-shirts, motorsport fashion, lifestyle brand"
        ogType="website"
        structuredData={structuredData}
      />
      <SkipLinks />
      <div className="min-h-screen bg-lii-bg text-foreground font-ui overflow-x-hidden relative">
        <ScrollProgress />
        <Navigation />
        <main id="main-content">
          <Hero />
          <TrustBar />
          <ProductShowcase />
          <div id="collection">
            <MinimalShowcase />
          </div>
          <BrandManifesto />
          <YouTubeIntegration />
          <ContentCreatorHub />
          <section className="py-20 bg-gradient-to-b from-lii-ink to-lii-black">
            <div className="container mx-auto px-6">
              <BirdLogoShowcase variant="carousel" />
            </div>
          </section>
          <SupercarGallery />
          <LifestyleTeaser />
          <PodcastShowcase />
          <div id="connect">
            <EmailCapture />
          </div>
        </main>
        <TwitchLiveIntegration />
        <Footer />
      </div>
    </>
  );
};

export default Index;
