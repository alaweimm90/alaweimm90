import Footer from '@/components/Footer';
import Navigation from '@/components/Navigation';
import { LiveItIconicWordmark } from '@/components/logo/LiveItIconicWordmark';
import { IconicDiamondMark } from '@/components/logo/IconicDiamondMark';
import { AutomotiveCrestLogo } from '@/components/logo/AutomotiveCrestLogo';
import { MinimalistRoadmark } from '@/components/logo/MinimalistRoadmark';

export default function BrandAssets() {
  return (
    <div className="min-h-screen bg-lii-bg text-foreground font-ui">
      <Navigation />
      <main className="container mx-auto px-6 pt-32 pb-24">
        <header className="mb-6">
          <h1 className="text-4xl font-display tracking-tight text-lii-cloud">Brand Assets</h1>
          <p className="text-lii-ash mt-2">
            Download logos, view icons, review typography, and inspect color tokens.
          </p>
        </header>

        <section className="mt-6">
          <h2 className="text-xl font-display text-lii-gold mb-6 tracking-widest">PRIMARY LOGOS</h2>
          
          {/* Horizontal Wordmark */}
          <div className="mb-12 p-8 bg-lii-ink/30 rounded-xl border border-lii-charcoal/30">
            <div className="flex justify-center mb-4">
              <LiveItIconicWordmark size={400} variant="horizontal" />
            </div>
            <div className="text-center text-sm text-lii-ash mt-4">
              Horizontal Wordmark — Primary logo for headers and marketing
            </div>
          </div>

          {/* Logo variations grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-8 bg-lii-ink/30 rounded-xl border border-lii-charcoal/30 hover:border-lii-gold/50 transition-colors">
              <div className="flex justify-center mb-4">
                <LiveItIconicWordmark size={180} variant="stacked" />
              </div>
              <div className="text-sm text-lii-ash">Stacked Wordmark</div>
              <div className="text-xs text-lii-ash/60 mt-1">For vertical layouts</div>
            </div>

            <div className="text-center p-8 bg-lii-ink/30 rounded-xl border border-lii-charcoal/30 hover:border-lii-gold/50 transition-colors">
              <div className="flex justify-center items-center h-40 mb-4">
                <IconicDiamondMark size={120} />
              </div>
              <div className="text-sm text-lii-ash">Diamond Mark</div>
              <div className="text-xs text-lii-ash/60 mt-1">Icon-only for favicon & apps</div>
            </div>

            <div className="text-center p-8 bg-lii-ink/30 rounded-xl border border-lii-charcoal/30 hover:border-lii-gold/50 transition-colors">
              <div className="flex justify-center items-center h-40 mb-4">
                <AutomotiveCrestLogo size={120} />
              </div>
              <div className="text-sm text-lii-ash">Heritage Crest</div>
              <div className="text-xs text-lii-ash/60 mt-1">Premium badge variant</div>
            </div>

            <div className="text-center p-8 bg-lii-ink/30 rounded-xl border border-lii-charcoal/30 hover:border-lii-gold/50 transition-colors">
              <div className="flex justify-center items-center h-40 mb-4">
                <MinimalistRoadmark size={120} />
              </div>
              <div className="text-sm text-lii-ash">Road Mark</div>
              <div className="text-xs text-lii-ash/60 mt-1">Motion & journey symbol</div>
            </div>
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-display text-lii-gold mb-3 tracking-widest">COLORS</h2>
          <div className="flex gap-3 mt-3">
            <div className="w-32 h-12 rounded-lg bg-lii-gold" />
            <div className="w-32 h-12 rounded-lg bg-lii-champagne" />
            <div className="w-32 h-12 rounded-lg bg-lii-ink" />
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-display text-lii-gold mb-3 tracking-widest">TYPOGRAPHY</h2>
          <p className="text-lii-cloud/80">Display: var(--font-display) | UI: var(--font-ui)</p>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-display text-lii-gold mb-3 tracking-widest">DOWNLOADS</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-lii-cloud/80">
            <div className="p-4 bg-lii-ink/20 rounded-lg border border-lii-charcoal/30">
              <div className="font-semibold text-lii-gold mb-2">Logo Package (ZIP)</div>
              <ul className="text-sm space-y-1">
                <li>• All logo variations (SVG, PNG, PDF)</li>
                <li>• Light & dark backgrounds</li>
                <li>• Multiple sizes (favicon to billboard)</li>
              </ul>
            </div>
            <div className="p-4 bg-lii-ink/20 rounded-lg border border-lii-charcoal/30">
              <div className="font-semibold text-lii-gold mb-2">Brand Guidelines (PDF)</div>
              <ul className="text-sm space-y-1">
                <li>• Logo usage & clearspace rules</li>
                <li>• Color specifications (RGB, CMYK, HEX)</li>
                <li>• Typography guidelines</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-xl font-display text-lii-gold mb-3 tracking-widest">USAGE NOTES</h2>
          <ol className="list-decimal pl-5 text-lii-cloud/80 space-y-1">
            <li>Keep clearspace equal to the cap height of the letter "I" in the wordmark.</li>
            <li>Reserve gold for accents and CTAs; avoid small gold text on dark backgrounds.</li>
            <li>Use the horizontal wordmark in headers; mark-only for avatars and favicons.</li>
          </ol>
        </section>
      </main>
      <Footer />
    </div>
  );
}
