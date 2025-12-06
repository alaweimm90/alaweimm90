import React from 'react';
import { Card } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';

const Pricing = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900">
            Pricing
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Choose the perfect plan for your fitness journey
          </p>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[
        { title: 'Core Program', price: '$89/month', features: ['Essential training', 'Nutrition basics'] },
        { title: 'Adaptive Engine', price: '$149/month', features: ['Interactive coaching', 'Progress tracking'] },
        { title: 'Performance Suite', price: '$229/month', features: ['Advanced analytics', 'Biohacking tools'] }
      ].map((plan, index) => (
        <Card key={index} className="p-6">
          <h3 className="text-xl font-semibold mb-2">{plan.title}</h3>
          <p className="text-2xl font-bold text-orange-600 mb-4">{plan.price}</p>
          <ul className="space-y-2">
            {plan.features.map((feature, i) => (
              <li key={i} className="flex items-center">
                <span className="text-green-500 mr-2">✓</span>
                {feature}
              </li>
            ))}
          </ul>
          <Button className="w-full mt-4" onClick={() => console.log('Select plan:', plan.title)}>
            Get Started
          </Button>
        </Card>
      ))}
        </div>
        
        <div className="mt-8 text-center">
          <Button 
            onClick={() => window.history.back()}
            variant="outline"
          >
            Go Back
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Pricing;
