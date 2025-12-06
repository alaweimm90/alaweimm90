'use client'

import { motion } from 'framer-motion'
import { ArrowRight, Brain, Search, FileText, Sparkles, Users, Zap, CheckCircle } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Navigation } from '@/components/navigation'
import { Footer } from '@/components/footer'
import { PricingSection } from '@/components/pricing-section'
import { FAQSection } from '@/components/faq-section'
import { TestimonialsSection } from '@/components/testimonials-section'

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: 'Advanced AI algorithms analyze thousands of papers to extract key insights and patterns.',
  },
  {
    icon: Search,
    title: 'Literature Review',
    description: 'Comprehensive literature reviews organized by themes with citation network visualization.',
  },
  {
    icon: FileText,
    title: 'Paper Writing Assistant',
    description: 'Real-time writing suggestions, structure guidance, and citation management.',
  },
  {
    icon: Sparkles,
    title: 'Hypothesis Generator',
    description: 'Generate novel research hypotheses based on existing literature gaps.',
  },
  {
    icon: Users,
    title: 'Collaboration Tools',
    description: 'Work with your team in real-time with shared projects and permissions.',
  },
  {
    icon: Zap,
    title: 'Fast & Efficient',
    description: 'Save weeks of research time with instant comprehensive analysis.',
  },
]

const stats = [
  { value: '10M+', label: 'Papers Analyzed' },
  { value: '50K+', label: 'Active Researchers' },
  { value: '95%', label: 'Time Saved' },
  { value: '4.9/5', label: 'User Rating' },
]

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <Navigation />

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-32 pb-24 px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="container mx-auto text-center"
        >
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary rounded-full mb-6"
            >
              <Sparkles className="w-4 h-4" />
              <span className="text-sm font-medium">Revolutionizing Academic Research</span>
            </motion.div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6 gradient-text">
              Your AI Research Assistant
            </h1>

            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Transform months of research into hours. Ghost Researcher uses advanced AI to analyze literature,
              generate hypotheses, and assist with paper writing.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/signup">
                <Button size="lg" className="gap-2">
                  Start Free Trial
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/demo">
                <Button size="lg" variant="outline">
                  Watch Demo
                </Button>
              </Link>
            </div>
          </div>
        </motion.div>

        {/* Background decoration */}
        <div className="absolute top-0 left-0 w-full h-full -z-10 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-6 border-y">
        <div className="container mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-primary">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground mt-1">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-6">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Everything You Need for Research Excellence
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Powerful features designed to accelerate your research workflow and enhance productivity.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="p-6 hover-scale">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                    <feature.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 px-6 bg-secondary/30">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              How Ghost Researcher Works
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Three simple steps to transform your research process
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Input Your Research Topic',
                description: 'Enter your research question or upload relevant papers to get started.',
              },
              {
                step: '02',
                title: 'AI Analyzes Literature',
                description: 'Our AI analyzes thousands of papers, extracting key insights and patterns.',
              },
              {
                step: '03',
                title: 'Get Comprehensive Results',
                description: 'Receive organized literature reviews, hypotheses, and writing assistance.',
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="relative"
              >
                <div className="text-6xl font-bold text-primary/20 mb-4">
                  {item.step}
                </div>
                <h3 className="text-2xl font-semibold mb-3">{item.title}</h3>
                <p className="text-muted-foreground">{item.description}</p>
                {index < 2 && (
                  <ArrowRight className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 w-8 h-8 text-primary/40" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="container mx-auto"
        >
          <Card className="p-12 text-center bg-gradient-to-r from-primary/10 to-purple-500/10 border-none">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Ready to Transform Your Research?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of researchers who are already saving time and producing better research with Ghost Researcher.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/signup">
                <Button size="lg" className="gap-2">
                  Start Your Free Trial
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/contact">
                <Button size="lg" variant="outline">
                  Contact Sales
                </Button>
              </Link>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              No credit card required • 14-day free trial • Cancel anytime
            </p>
          </Card>
        </motion.div>
      </section>

      <TestimonialsSection />
      <PricingSection />
      <FAQSection />
      <Footer />
    </div>
  )
}