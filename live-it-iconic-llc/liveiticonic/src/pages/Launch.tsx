import Footer from '@/components/Footer';
import Navigation from '@/components/Navigation';

export default function Launch() {
  return (
    <div className="min-h-screen bg-lii-bg text-foreground font-ui">
      <Navigation />
      <main className="container mx-auto px-6 pt-32 pb-24">
        <header className="text-center mb-10">
          <h1 className="text-4xl font-display tracking-tight text-lii-cloud">Launch</h1>
          <p className="text-lii-ash mt-2">
            Milestones and timelines for the initial drops and channel rollouts.
          </p>
        </header>

        <section className="max-w-3xl mx-auto grid gap-6">
          <div className="rounded-2xl p-6 bg-lii-ink/60 border border-lii-gold/20">
            <h2 className="text-lg font-display text-lii-gold mb-2 tracking-widest">SEED</h2>
            <ul className="list-disc pl-5 text-lii-cloud/80 space-y-1">
              <li>Teasers and email capture</li>
              <li>AI-generated product mockups (internal validation)</li>
              <li>Channel handles secured</li>
            </ul>
          </div>
          <div className="rounded-2xl p-6 bg-lii-ink/60 border border-lii-gold/20">
            <h2 className="text-lg font-display text-lii-gold mb-2 tracking-widest">DROP 1</h2>
            <ul className="list-disc pl-5 text-lii-cloud/80 space-y-1">
              <li>Core tee, hoodie, cap</li>
              <li>Landing experience and lookbook export</li>
              <li>Fulfillment workflow validation</li>
            </ul>
          </div>
          <div className="rounded-2xl p-6 bg-lii-ink/60 border border-lii-gold/20">
            <h2 className="text-lg font-display text-lii-gold mb-2 tracking-widest">COMMUNITY</h2>
            <ul className="list-disc pl-5 text-lii-cloud/80 space-y-1">
              <li>Creator collaborations</li>
              <li>Meetups and showcases</li>
              <li>Companion app alpha</li>
            </ul>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
