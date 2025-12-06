import Footer from '@/components/Footer';
import Navigation from '@/components/Navigation';

const Policies = () => {
  return (
    <div className="min-h-screen bg-lii-black text-foreground font-ui overflow-x-hidden">
      <Navigation />
      <main>
        <div className="pt-32 pb-20">
          <div className="container mx-auto px-6">
            <div className="max-w-4xl mx-auto">
              <h1 className="text-4xl sm:text-5xl font-display font-light text-center mb-16 text-luxury">
                Policies & Information
              </h1>

              {/* Prototype Notice */}
              <div className="bg-lii-charcoal/20 border border-lii-gold/20 rounded-2xl p-8 mb-16">
                <h2 className="text-xl font-display text-lii-gold mb-4 tracking-[0.2em]">
                  PROTOTYPE NOTICE
                </h2>
                <p className="text-foreground/70 font-ui font-light leading-relaxed">
                  This is a design prototype. All products, features, and policies shown are for
                  demonstration purposes. No actual products are currently available for purchase.
                </p>
              </div>

              <div className="space-y-16">
                {/* Privacy Policy */}
                <section>
                  <h2 className="text-2xl font-display font-light text-lii-gold mb-8 tracking-[0.15em]">
                    PRIVACY POLICY
                  </h2>
                  <div className="space-y-6 text-foreground/70 font-ui font-light leading-relaxed">
                    <h3 className="text-lg text-foreground/90 font-medium">
                      Information Collection
                    </h3>
                    <p>
                      This prototype website may collect basic analytics data to improve the design
                      concept demonstration. No personal information is stored or processed for
                      commercial purposes.
                    </p>

                    <h3 className="text-lg text-foreground/90 font-medium">Design Data</h3>
                    <p>
                      All product information, sizing guides, and specifications are presented for
                      design demonstration purposes only.
                    </p>

                    <h3 className="text-lg text-foreground/90 font-medium">Contact Information</h3>
                    <p>
                      Any contact forms or newsletter signups are non-functional in this prototype
                      version and are included for user experience demonstration only.
                    </p>
                  </div>
                </section>

                {/* Terms of Service */}
                <section>
                  <h2 className="text-2xl font-display font-light text-lii-champagne mb-8 tracking-[0.15em]">
                    TERMS OF SERVICE
                  </h2>
                  <div className="space-y-6 text-foreground/70 font-ui font-light leading-relaxed">
                    <h3 className="text-lg text-foreground/90 font-medium">Prototype Usage</h3>
                    <p>
                      This website is a design prototype created for demonstration purposes. Users
                      may browse and interact with the interface to experience the proposed design
                      concept.
                    </p>

                    <h3 className="text-lg text-foreground/90 font-medium">
                      No Commercial Transactions
                    </h3>
                    <p>
                      No actual products are available for purchase. All shopping cart
                      functionality, payment processing, and order systems are non-functional design
                      demonstrations.
                    </p>

                    <h3 className="text-lg text-foreground/90 font-medium">
                      Intellectual Property
                    </h3>
                    <p>
                      The LIVE IT ICONIC brand name, logo, and designs are created for this
                      prototype demonstration. All imagery is used for design purposes.
                    </p>
                  </div>
                </section>

                {/* Shipping & Returns (Conceptual) */}
                <section>
                  <h2 className="text-2xl font-display font-light text-lii-bronze mb-8 tracking-[0.15em]">
                    SHIPPING & RETURNS
                  </h2>
                  <div className="space-y-6 text-foreground/70 font-ui font-light leading-relaxed">
                    <p className="italic text-foreground/60">
                      The following represents the proposed shipping and return policies for LIVE IT
                      ICONIC:
                    </p>

                    <h3 className="text-lg text-foreground/90 font-medium">Proposed Shipping</h3>
                    <ul className="space-y-2 pl-6">
                      <li>• Premium packaging for all orders</li>
                      <li>• Carbon-neutral shipping options</li>
                      <li>• Express delivery for performance essentials</li>
                      <li>• International shipping to select markets</li>
                    </ul>

                    <h3 className="text-lg text-foreground/90 font-medium">Proposed Returns</h3>
                    <ul className="space-y-2 pl-6">
                      <li>• 30-day return window for unworn items</li>
                      <li>• Complimentary return shipping</li>
                      <li>• Quality guarantee on all products</li>
                      <li>• Size exchange program</li>
                    </ul>
                  </div>
                </section>

                {/* Size Guide (Conceptual) */}
                <section>
                  <h2 className="text-2xl font-display font-light text-lii-gold mb-8 tracking-[0.15em]">
                    SIZE GUIDE
                  </h2>
                  <div className="space-y-6 text-foreground/70 font-ui font-light leading-relaxed">
                    <p className="italic text-foreground/60">
                      Proposed sizing system for LIVE IT ICONIC athletic wear:
                    </p>

                    <div className="bg-lii-charcoal/10 rounded-xl p-6 overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-lii-gold/20">
                            <th className="text-left py-3 text-lii-gold">Size</th>
                            <th className="text-left py-3 text-lii-gold">Chest (in)</th>
                            <th className="text-left py-3 text-lii-gold">Waist (in)</th>
                            <th className="text-left py-3 text-lii-gold">Hips (in)</th>
                          </tr>
                        </thead>
                        <tbody className="text-foreground/60">
                          <tr className="border-b border-lii-gold/10">
                            <td className="py-3">XS</td>
                            <td className="py-3">32-34</td>
                            <td className="py-3">26-28</td>
                            <td className="py-3">32-34</td>
                          </tr>
                          <tr className="border-b border-lii-gold/10">
                            <td className="py-3">S</td>
                            <td className="py-3">34-36</td>
                            <td className="py-3">28-30</td>
                            <td className="py-3">34-36</td>
                          </tr>
                          <tr className="border-b border-lii-gold/10">
                            <td className="py-3">M</td>
                            <td className="py-3">36-38</td>
                            <td className="py-3">30-32</td>
                            <td className="py-3">36-38</td>
                          </tr>
                          <tr className="border-b border-lii-gold/10">
                            <td className="py-3">L</td>
                            <td className="py-3">38-40</td>
                            <td className="py-3">32-34</td>
                            <td className="py-3">38-40</td>
                          </tr>
                          <tr>
                            <td className="py-3">XL</td>
                            <td className="py-3">40-42</td>
                            <td className="py-3">34-36</td>
                            <td className="py-3">40-42</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    <p className="text-sm text-foreground/50">
                      * Sizing chart for design demonstration purposes only
                    </p>
                  </div>
                </section>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Policies;
