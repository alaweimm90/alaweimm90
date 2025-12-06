'use client'

import { motion } from 'framer-motion'
import { Zap, Shuffle, Lightbulb, Rocket, Brain, Layers, ArrowRight, Sparkles } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Navigation } from '@/components/navigation'
import { Footer } from '@/components/footer'
import { useState, useEffect } from 'react'

const features = [
  {
    icon: Shuffle,
    title: 'Domain Collision',
    description: 'Randomly combine concepts from different fields to spark innovative ideas.',
  },
  {
    icon: Brain,
    title: 'AI-Powered Generation',
    description: 'Advanced AI analyzes domain patterns to create viable business concepts.',
  },
  {
    icon: Rocket,
    title: 'Business Plan Generator',
    description: 'Transform ideas into complete business plans with market analysis.',
  },
  {
    icon: Layers,
    title: 'Pitch Deck Builder',
    description: 'Auto-generate professional pitch decks for investors.',
  },
  {
    icon: Lightbulb,
    title: 'Idea Refinement',
    description: 'AI suggestions to improve and evolve your generated concepts.',
  },
  {
    icon: Sparkles,
    title: 'Community Voting',
    description: 'Share ideas and get feedback from the innovation community.',
  },
]

const exampleIdeas = [
  {
    domain1: 'Quantum Computing',
    domain2: 'Agriculture',
    idea: 'Quantum-optimized crop yield prediction',
    viability: 85,
  },
  {
    domain1: 'Blockchain',
    domain2: 'Mental Health',
    idea: 'Decentralized therapy session verification',
    viability: 72,
  },
  {
    domain1: 'AR/VR',
    domain2: 'Fine Dining',
    idea: 'Virtual reality taste experiences',
    viability: 68,
  },
  {
    domain1: 'AI',
    domain2: 'Fashion',
    idea: 'Personalized style prediction algorithm',
    viability: 91,
  },
]

const stats = [
  { value: '1M+', label: 'Ideas Generated' },
  { value: '50K+', label: 'Active Users' },
  { value: '500+', label: 'Funded Startups' },
  { value: '95%', label: 'User Satisfaction' },
]

export default function HomePage() {
  const [currentIdea, setCurrentIdea] = useState(0)
  const [isGenerating, setIsGenerating] = useState(false)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIdea((prev) => (prev + 1) % exampleIdeas.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-purple-950/5 to-background">
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
              className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500/10 to-blue-500/10 text-purple-600 rounded-full mb-6"
            >
              <Zap className="w-4 h-4" />
              <span className="text-sm font-medium">Where Innovation Meets Chaos</span>
            </motion.div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="gradient-text bg-gradient-to-r from-purple-600 via-blue-600 to-cyan-600">
                Chaos Engine
              </span>
            </h1>

            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Generate breakthrough ideas by colliding domains. Our AI creates unexpected
              connections that lead to revolutionary innovations.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <Link href="/generate">
                <Button
                  size="lg"
                  className="gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                  onClick={() => setIsGenerating(true)}
                >
                  {isGenerating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Shuffle className="w-4 h-4" />
                      </motion.div>
                      Generating...
                    </>
                  ) : (
                    <>
                      Generate Ideas
                      <Zap className="w-4 h-4" />
                    </>
                  )}
                </Button>
              </Link>
              <Link href="/explore">
                <Button size="lg" variant="outline">
                  Explore Gallery
                </Button>
              </Link>
            </div>

            {/* Live Idea Generator Preview */}
            <motion.div
              key={currentIdea}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-2xl mx-auto"
            >
              <Card className="p-6 bg-gradient-to-r from-purple-500/5 to-blue-500/5 border-purple-500/20">
                <div className="flex items-center justify-center gap-3 mb-3">
                  <span className="px-3 py-1 bg-purple-500/10 rounded-full text-sm font-medium">
                    {exampleIdeas[currentIdea].domain1}
                  </span>
                  <Zap className="w-4 h-4 text-yellow-500" />
                  <span className="px-3 py-1 bg-blue-500/10 rounded-full text-sm font-medium">
                    {exampleIdeas[currentIdea].domain2}
                  </span>
                </div>
                <h3 className="text-lg font-semibold mb-2">
                  {exampleIdeas[currentIdea].idea}
                </h3>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Viability Score</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 h-2 bg-secondary rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${exampleIdeas[currentIdea].viability}%` }}
                        transition={{ duration: 1, delay: 0.5 }}
                        className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                      />
                    </div>
                    <span className="text-sm font-medium">
                      {exampleIdeas[currentIdea].viability}%
                    </span>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>
        </motion.div>

        {/* Animated Background */}
        <div className="absolute top-0 left-0 w-full h-full -z-10 overflow-hidden">
          <motion.div
            animate={{
              scale: [1, 1.5, 1],
              rotate: [0, 90, 0],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute top-1/3 left-1/3 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1.5, 1, 1.5],
              rotate: [0, -90, 0],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
          />
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
                <div className="text-3xl md:text-4xl font-bold gradient-text bg-gradient-to-r from-purple-600 to-blue-600">
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

      {/* How It Works */}
      <section className="py-24 px-6">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              How Chaos Engine Works
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Three steps to your next breakthrough innovation
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              {
                step: '01',
                title: 'Select Domains',
                description: 'Choose industries or let our AI randomly select domains to collide.',
                icon: Layers,
              },
              {
                step: '02',
                title: 'Generate Ideas',
                description: 'AI analyzes patterns and creates innovative concepts at intersections.',
                icon: Zap,
              },
              {
                step: '03',
                title: 'Develop & Launch',
                description: 'Get business plans, pitch decks, and community feedback.',
                icon: Rocket,
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
                <Card className="p-8 h-full hover:shadow-xl transition-shadow">
                  <div className="w-14 h-14 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl flex items-center justify-center mb-4">
                    <item.icon className="w-7 h-7 text-purple-600" />
                  </div>
                  <div className="text-5xl font-bold text-purple-500/20 mb-3">
                    {item.step}
                  </div>
                  <h3 className="text-xl font-semibold mb-3">{item.title}</h3>
                  <p className="text-muted-foreground">{item.description}</p>
                </Card>
                {index < 2 && (
                  <ArrowRight className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 w-8 h-8 text-purple-500/30" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24 px-6 bg-gradient-to-b from-secondary/30 to-background">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Powerful Innovation Tools
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Everything you need to turn chaos into breakthrough innovations
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="p-6 h-full hover:shadow-xl hover:scale-105 transition-all group">
                  <div className="w-12 h-12 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <feature.icon className="w-6 h-6 text-purple-600" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Success Stories */}
      <section className="py-24 px-6">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Ideas That Became Reality
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Real startups born from Chaos Engine collisions
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                name: 'QuantumFarm',
                domains: 'Quantum + Agriculture',
                funding: '$12M Series A',
                description: 'Quantum computing for crop optimization raised significant funding.',
              },
              {
                name: 'NeuroFashion',
                domains: 'Neuroscience + Fashion',
                funding: '$5M Seed',
                description: 'Brain-computer interfaces for personalized style recommendations.',
              },
              {
                name: 'CryptoHealth',
                domains: 'Blockchain + Healthcare',
                funding: '$8M Series A',
                description: 'Decentralized patient records with smart contract insurance.',
              },
            ].map((story, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="p-6 h-full bg-gradient-to-br from-purple-500/5 to-blue-500/5">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold">{story.name}</h3>
                    <span className="text-sm font-medium text-green-600 bg-green-100 dark:bg-green-900/30 px-2 py-1 rounded">
                      {story.funding}
                    </span>
                  </div>
                  <p className="text-sm text-purple-600 font-medium mb-2">
                    {story.domains}
                  </p>
                  <p className="text-muted-foreground">
                    {story.description}
                  </p>
                </Card>
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
          <Card className="p-12 text-center bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-cyan-500/10 border-none">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Ready to Create Your Breakthrough?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of innovators using Chaos Engine to generate their next big idea.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/generate">
                <Button size="lg" className="gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700">
                  Start Generating
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/showcase">
                <Button size="lg" variant="outline">
                  View Showcase
                </Button>
              </Link>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              Free to start • No credit card required • Generate unlimited ideas
            </p>
          </Card>
        </motion.div>
      </section>

      <Footer />
    </div>
  )
}