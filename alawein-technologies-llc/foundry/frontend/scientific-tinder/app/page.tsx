'use client'

import { motion } from 'framer-motion'
import { Heart, Users, Sparkles, Zap, ArrowRight, Globe, Award, TrendingUp } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Navigation } from '@/components/navigation'
import { Footer } from '@/components/footer'

const features = [
  {
    icon: Heart,
    title: 'Smart Matching',
    description: 'AI-powered algorithm matches you with researchers based on interests, expertise, and collaboration style.',
  },
  {
    icon: Users,
    title: 'Global Network',
    description: 'Connect with researchers from universities and institutions worldwide across all disciplines.',
  },
  {
    icon: Sparkles,
    title: 'Swipe Interface',
    description: 'Intuitive swipe interface makes finding collaborators fun and efficient.',
  },
  {
    icon: Zap,
    title: 'Instant Messaging',
    description: 'Real-time chat with matched researchers to discuss ideas and potential collaborations.',
  },
  {
    icon: Globe,
    title: 'Cross-Disciplinary',
    description: 'Discover unexpected connections between fields for groundbreaking research.',
  },
  {
    icon: Award,
    title: 'Verified Profiles',
    description: 'All researchers are verified through institutional affiliations and publication records.',
  },
]

const stats = [
  { value: '100K+', label: 'Active Researchers' },
  { value: '500+', label: 'Universities' },
  { value: '15K+', label: 'Collaborations' },
  { value: '92%', label: 'Match Success' },
]

const testimonials = [
  {
    quote: "Scientific Tinder helped me find the perfect collaborator for my quantum biology research. We've published 3 papers together!",
    author: 'Dr. Maria Santos',
    role: 'Physics Professor, MIT',
    avatar: 'MS',
  },
  {
    quote: "The cross-disciplinary matches are incredible. I never thought a computer scientist would revolutionize my neuroscience research.",
    author: 'Prof. John Chen',
    role: 'Neuroscientist, Stanford',
    avatar: 'JC',
  },
  {
    quote: "Finally, a platform that understands research collaboration. The matching algorithm is remarkably accurate.",
    author: 'Dr. Aisha Patel',
    role: 'Biologist, Oxford',
    avatar: 'AP',
  },
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
              className="inline-flex items-center gap-2 px-4 py-2 bg-pink-500/10 text-pink-600 rounded-full mb-6"
            >
              <Heart className="w-4 h-4" />
              <span className="text-sm font-medium">Find Your Research Match</span>
            </motion.div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="gradient-text bg-gradient-to-r from-pink-500 to-purple-600">
                Scientific Tinder
              </span>
            </h1>

            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Swipe right on your next research collaboration. Connect with brilliant minds
              from around the world and make scientific breakthroughs together.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/signup">
                <Button size="lg" className="gap-2 bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700">
                  Start Matching
                  <Heart className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/explore">
                <Button size="lg" variant="outline">
                  Browse Researchers
                </Button>
              </Link>
            </div>
          </div>
        </motion.div>

        {/* Animated Background */}
        <div className="absolute top-0 left-0 w-full h-full -z-10 overflow-hidden">
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              rotate: [0, 180, 360],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1.2, 1, 1.2],
              rotate: [360, 180, 0],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
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
                <div className="text-3xl md:text-4xl font-bold gradient-text bg-gradient-to-r from-pink-500 to-purple-600">
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
              How Scientific Tinder Works
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Find your perfect research partner in three simple steps
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              {
                step: '1',
                title: 'Create Your Profile',
                description: 'Add your research interests, publications, and what you\'re looking for in a collaborator.',
                icon: '👤',
              },
              {
                step: '2',
                title: 'Swipe & Match',
                description: 'Browse researcher profiles and swipe right on those you\'d like to collaborate with.',
                icon: '💝',
              },
              {
                step: '3',
                title: 'Start Collaborating',
                description: 'When you match, start chatting and plan your next groundbreaking research project.',
                icon: '🚀',
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
              >
                <Card className="p-8 text-center hover:shadow-lg transition-shadow">
                  <div className="text-5xl mb-4">{item.icon}</div>
                  <div className="text-6xl font-bold text-pink-500/20 mb-4">
                    {item.step}
                  </div>
                  <h3 className="text-xl font-semibold mb-3">{item.title}</h3>
                  <p className="text-muted-foreground">{item.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24 px-6 bg-secondary/30">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Features for Modern Researchers
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Everything you need to find and collaborate with the perfect research partner
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="p-6 hover:shadow-lg hover:scale-105 transition-all">
                  <div className="w-12 h-12 bg-gradient-to-r from-pink-500/10 to-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                    <feature.icon className="w-6 h-6 text-pink-500" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24 px-6">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Success Stories
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Researchers worldwide are finding their perfect collaboration matches
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="p-6 h-full">
                  <p className="text-lg mb-6 italic">"{testimonial.quote}"</p>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center text-white font-bold">
                      {testimonial.avatar}
                    </div>
                    <div>
                      <p className="font-semibold">{testimonial.author}</p>
                      <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                    </div>
                  </div>
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
          <Card className="p-12 text-center bg-gradient-to-r from-pink-500/10 to-purple-500/10 border-none">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Ready to Find Your Research Match?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of researchers already collaborating on groundbreaking projects.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/signup">
                <Button size="lg" className="gap-2 bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700">
                  Create Free Profile
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/demo">
                <Button size="lg" variant="outline">
                  See Live Demo
                </Button>
              </Link>
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              Free to join • Verified researchers only • Start matching today
            </p>
          </Card>
        </motion.div>
      </section>

      <Footer />
    </div>
  )
}