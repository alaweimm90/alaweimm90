import Footer from '@/components/Footer';
import Navigation from '@/components/Navigation';
import SEO from '@/components/SEO';
import { useState } from 'react';

export default function Contact() {
  const [submitted, setSubmitted] = useState(false);

  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'ContactPage',
    name: 'Contact Live It Iconic',
    description:
      "Get in touch with Live It Iconic. Send us a message about our luxury automotive lifestyle apparel and we'll reply shortly.",
    url: 'https://liveiticonic.com/contact',
    mainEntity: {
      '@type': 'Organization',
      name: 'Live It Iconic',
      contactPoint: {
        '@type': 'ContactPoint',
        contactType: 'customer service',
        availableLanguage: 'English',
      },
    },
  };

  return (
    <>
      <SEO
        title="Contact Live It Iconic - Get in Touch"
        description="Connect with our team about premium automotive-inspired apparel, orders, or partnership inquiries. We respond to all messages promptly."
        canonical="/contact"
        ogImage="/og-contact.jpg"
        ogImageAlt="Live It Iconic Contact"
        keywords="contact Live It Iconic, customer support, premium apparel inquiry, brand partnership"
        structuredData={structuredData}
      />
      <div className="min-h-screen bg-lii-bg text-foreground font-ui">
        <Navigation />
        <main className="container mx-auto px-6 pt-32 pb-24">
          <header className="text-center mb-10">
            <h1 className="text-4xl font-display tracking-tight text-lii-cloud">Contact</h1>
            <p className="text-lii-ash mt-2">Send a message and we’ll reply shortly.</p>
          </header>

          {!submitted ? (
            <form
              onSubmit={e => {
                e.preventDefault();
                // TODO: hook up to an email/service endpoint
                setSubmitted(true);
              }}
              className="max-w-xl mx-auto grid gap-4"
            >
              <label className="grid gap-2">
                <span className="text-sm text-lii-ash">Name</span>
                <input
                  required
                  className="w-full px-4 py-3 rounded-lg bg-lii-ink/60 border border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash focus:outline-none focus:ring-2 focus:ring-lii-gold/40"
                  placeholder="Your name"
                />
              </label>
              <label className="grid gap-2">
                <span className="text-sm text-lii-ash">Message</span>
                <textarea
                  required
                  rows={6}
                  className="w-full px-4 py-3 rounded-lg bg-lii-ink/60 border border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash focus:outline-none focus:ring-2 focus:ring-lii-gold/40"
                  placeholder="How can we help?"
                />
              </label>
              <button
                type="submit"
                className="h-12 rounded-lg bg-lii-gold text-lii-black font-ui font-medium tracking-wide hover:opacity-90 transition"
              >
                Send
              </button>
            </form>
          ) : (
            <p className="text-center text-lii-cloud">Thanks — your message has been noted.</p>
          )}
        </main>
        <Footer />
      </div>
    </>
  );
}
