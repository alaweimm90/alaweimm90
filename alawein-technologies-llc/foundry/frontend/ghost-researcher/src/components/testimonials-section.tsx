'use client'

import { motion } from 'framer-motion'
import { Star, Quote } from 'lucide-react'
import { Card } from '@/components/ui/card'
import Image from 'next/image'

const testimonials = [
  {
    name: 'Dr. Sarah Chen',
    role: 'Professor of Biology',
    institution: 'Stanford University',
    avatar: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150',
    content: 'Ghost Researcher has completely transformed how I conduct literature reviews. What used to take weeks now takes hours. The AI-powered insights have helped me identify research gaps I would have never found on my own.',
    rating: 5,
  },
  {
    name: 'Michael Rodriguez',
    role: 'PhD Candidate',
    institution: 'MIT',
    avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150',
    content: "As a graduate student, Ghost Researcher is invaluable. The hypothesis generation feature has sparked ideas for three new research projects. It's like having a brilliant research assistant available 24/7.",
    rating: 5,
  },
  {
    name: 'Prof. Emily Watson',
    role: 'Research Director',
    institution: 'Oxford University',
    avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150',
    content: 'Our entire research team now uses Ghost Researcher. The collaboration features are excellent, and the time savings are remarkable. We published 40% more papers last year thanks to this tool.',
    rating: 5,
  },
  {
    name: 'Dr. James Liu',
    role: 'Principal Investigator',
    institution: 'Harvard Medical School',
    avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150',
    content: 'The citation network visualization alone is worth the subscription. Ghost Researcher helps us see connections between research areas that we would have missed otherwise. Essential tool for modern research.',
    rating: 5,
  },
  {
    name: 'Dr. Priya Patel',
    role: 'Assistant Professor',
    institution: 'Caltech',
    avatar: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150',
    content: "Ghost Researcher's writing assistant has improved the quality of our papers significantly. The suggestions are intelligent and context-aware. It's like having an experienced editor on demand.",
    rating: 5,
  },
  {
    name: 'Robert Thompson',
    role: 'Research Scientist',
    institution: 'NASA',
    avatar: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150',
    content: 'In aerospace research, staying current with literature is crucial but time-consuming. Ghost Researcher helps us stay on top of developments across multiple disciplines. Absolutely game-changing.',
    rating: 5,
  },
]

export function TestimonialsSection() {
  return (
    <section className="py-24 px-6">
      <div className="container mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Trusted by Leading Researchers
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Join thousands of researchers who are advancing their fields with Ghost Researcher
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="p-6 h-full flex flex-col">
                <div className="flex items-center mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star
                      key={i}
                      className="w-5 h-5 fill-yellow-400 text-yellow-400"
                    />
                  ))}
                </div>

                <Quote className="w-8 h-8 text-primary/20 mb-2" />

                <p className="text-muted-foreground flex-grow mb-6">
                  "{testimonial.content}"
                </p>

                <div className="flex items-center gap-4">
                  <div className="relative w-12 h-12 rounded-full overflow-hidden bg-secondary">
                    <Image
                      src={testimonial.avatar}
                      alt={testimonial.name}
                      fill
                      className="object-cover"
                    />
                  </div>
                  <div>
                    <p className="font-semibold">{testimonial.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {testimonial.role}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {testimonial.institution}
                    </p>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary rounded-full">
            <Star className="w-5 h-5 fill-current" />
            <span className="font-semibold">4.9/5 rating from 10,000+ researchers</span>
          </div>
        </motion.div>
      </div>
    </section>
  )
}