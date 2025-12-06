'use client'

import { motion } from 'framer-motion'
import { Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import Link from 'next/link'

const plans = [
  {
    name: 'Starter',
    price: '$29',
    period: '/month',
    description: 'Perfect for individual researchers and students',
    features: [
      'Up to 100 paper analyses per month',
      'Basic literature review',
      'Citation management',
      'PDF export',
      'Email support',
      'Basic hypothesis generation',
    ],
    limitations: [
      'No collaboration features',
      'Limited API access',
      'No priority support',
    ],
    cta: 'Start Free Trial',
    highlighted: false,
  },
  {
    name: 'Professional',
    price: '$99',
    period: '/month',
    description: 'For research teams and institutions',
    features: [
      'Unlimited paper analyses',
      'Advanced literature review',
      'Full citation network visualization',
      'Collaboration tools (up to 10 members)',
      'Priority support',
      'Advanced hypothesis generation',
      'Paper writing assistant',
      'API access',
      'Custom integrations',
    ],
    limitations: [],
    cta: 'Start Free Trial',
    highlighted: true,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'For large organizations with specific needs',
    features: [
      'Everything in Professional',
      'Unlimited team members',
      'Custom AI model training',
      'On-premise deployment option',
      'Dedicated account manager',
      '24/7 phone support',
      'Custom workflows',
      'Advanced security features',
      'SLA guarantee',
    ],
    limitations: [],
    cta: 'Contact Sales',
    highlighted: false,
  },
]

export function PricingSection() {
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
            Choose Your Research Plan
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Flexible pricing options designed to scale with your research needs
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Card
                className={cn(
                  'relative p-8 h-full flex flex-col',
                  plan.highlighted && 'border-primary shadow-xl scale-105'
                )}
              >
                {plan.highlighted && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                    <span className="bg-primary text-primary-foreground text-sm font-semibold px-4 py-1 rounded-full">
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="mb-6">
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <div className="flex items-baseline mb-2">
                    <span className="text-4xl font-bold">{plan.price}</span>
                    <span className="text-muted-foreground ml-1">
                      {plan.period}
                    </span>
                  </div>
                  <p className="text-muted-foreground">{plan.description}</p>
                </div>

                <div className="space-y-4 mb-8 flex-grow">
                  {plan.features.map((feature, idx) => (
                    <div key={idx} className="flex items-start gap-3">
                      <Check className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
                      <span className="text-sm">{feature}</span>
                    </div>
                  ))}
                  {plan.limitations.map((limitation, idx) => (
                    <div key={idx} className="flex items-start gap-3">
                      <X className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                      <span className="text-sm text-muted-foreground">
                        {limitation}
                      </span>
                    </div>
                  ))}
                </div>

                <Link href={plan.name === 'Enterprise' ? '/contact' : '/signup'}>
                  <Button
                    className="w-full"
                    variant={plan.highlighted ? 'default' : 'outline'}
                  >
                    {plan.cta}
                  </Button>
                </Link>
              </Card>
            </motion.div>
          ))}
        </div>

        <div className="mt-12 text-center">
          <p className="text-muted-foreground">
            All plans include a 14-day free trial. No credit card required.
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Need help choosing?{' '}
            <Link href="/contact" className="text-primary hover:underline">
              Contact our team
            </Link>
          </p>
        </div>
      </div>
    </section>
  )
}