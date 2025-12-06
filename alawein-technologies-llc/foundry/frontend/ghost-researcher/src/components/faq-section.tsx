'use client'

import { motion } from 'framer-motion'
import * as Accordion from '@radix-ui/react-accordion'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

const faqs = [
  {
    question: 'How does Ghost Researcher analyze research papers?',
    answer: 'Ghost Researcher uses advanced natural language processing and machine learning algorithms to analyze research papers. It extracts key information, identifies patterns, and synthesizes insights from thousands of papers in seconds.',
  },
  {
    question: 'Can I upload my own papers for analysis?',
    answer: 'Yes! You can upload PDFs, Word documents, or paste text directly into Ghost Researcher. Our AI will analyze your papers alongside our database of millions of academic papers to provide comprehensive insights.',
  },
  {
    question: 'How accurate is the hypothesis generation feature?',
    answer: 'Our hypothesis generation is based on identifying gaps in existing research and potential connections between different fields. While the AI generates novel and plausible hypotheses, they should always be validated through proper research methods.',
  },
  {
    question: 'Is my research data secure?',
    answer: 'Absolutely. We use enterprise-grade encryption for all data transmission and storage. Your research remains private and is never shared with third parties. Enterprise customers can opt for on-premise deployment for additional security.',
  },
  {
    question: 'Can I collaborate with my research team?',
    answer: 'Yes! Professional and Enterprise plans include collaboration features. You can share projects, work on papers together in real-time, and manage permissions for team members.',
  },
  {
    question: 'What citation formats are supported?',
    answer: 'Ghost Researcher supports all major citation formats including APA, MLA, Chicago, Harvard, IEEE, and more. You can easily export citations in your preferred format or integrate with citation managers like Zotero and Mendeley.',
  },
  {
    question: 'How does the paper writing assistant work?',
    answer: "Our AI writing assistant provides real-time suggestions for structure, content, and style. It helps with everything from outlining to polishing final drafts, while ensuring your writing remains original and maintains your voice.",
  },
  {
    question: 'Is there an API available for integration?',
    answer: 'Yes, Professional and Enterprise plans include API access. You can integrate Ghost Researcher into your existing research workflows, institutional systems, or custom applications.',
  },
]

export function FAQSection() {
  return (
    <section className="py-24 px-6 bg-secondary/30">
      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Frequently Asked Questions
          </h2>
          <p className="text-xl text-muted-foreground">
            Everything you need to know about Ghost Researcher
          </p>
        </motion.div>

        <Accordion.Root type="single" collapsible className="space-y-4">
          {faqs.map((faq, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.05 }}
            >
              <Accordion.Item
                value={`item-${index}`}
                className="border rounded-lg px-6 overflow-hidden"
              >
                <Accordion.Trigger className="flex w-full items-center justify-between py-4 text-left hover:text-primary transition-colors group">
                  <span className="text-lg font-medium">{faq.question}</span>
                  <ChevronDown className="h-5 w-5 shrink-0 transition-transform duration-200 group-data-[state=open]:rotate-180" />
                </Accordion.Trigger>
                <Accordion.Content className="overflow-hidden transition-all data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
                  <div className="pb-4 pt-0 text-muted-foreground">
                    {faq.answer}
                  </div>
                </Accordion.Content>
              </Accordion.Item>
            </motion.div>
          ))}
        </Accordion.Root>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <p className="text-muted-foreground">
            Still have questions?{' '}
            <a href="/contact" className="text-primary hover:underline">
              Contact our support team
            </a>
          </p>
        </motion.div>
      </div>
    </section>
  )
}