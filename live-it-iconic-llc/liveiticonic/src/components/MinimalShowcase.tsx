const MinimalShowcase = () => {
  return (
    <section className="relative pt-0 md:pt-0 pb-16 md:pb-20 bg-lii-bg flex items-center overflow-hidden">
      {/* Subtle animated orbs */}
      <div className="absolute inset-0 opacity-15">
        <div
          className="absolute top-1/2 left-1/4 w-80 h-80 bg-lii-gold rounded-full blur-[100px]"
          style={{
            animation: 'luxuryPulse 12s ease-in-out infinite',
          }}
        ></div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Glass container for header */}
          <div className="backdrop-blur-xl bg-lii-ink/30 border border-lii-gold/20 rounded-2xl p-10 md:p-14 shadow-[0_8px_32px_rgba(193,160,96,0.1)]">
            <h2 className="text-4xl sm:text-5xl md:text-6xl font-display tracking-tight mb-6 leading-tight relative">
              <span className="bg-gradient-to-r from-lii-cloud via-lii-gold to-lii-cloud bg-clip-text text-transparent">
                Understanding
              </span>
              <span className="block italic text-lii-gold mt-2 text-3xl sm:text-4xl md:text-5xl">
                your lifestyle
              </span>
            </h2>

            <div className="w-20 h-px bg-gradient-to-r from-transparent via-lii-gold to-transparent mx-auto my-6"></div>

            <div className="space-y-4 max-w-2xl mx-auto">
              <p className="text-xl sm:text-2xl font-ui text-lii-ash leading-relaxed">
                Embracing what moves you
              </p>
              <p className="text-lg sm:text-xl font-ui text-lii-gold/70 italic leading-relaxed">
                Living with intention
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
export default MinimalShowcase;
