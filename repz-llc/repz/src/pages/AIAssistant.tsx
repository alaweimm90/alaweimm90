import React from 'react';
import { Card } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';

const AIAssistant = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900">
            A I Assistant
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Your intelligent fitness companion powered by advanced AI
          </p>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[
        { title: 'Workout Planning', description: 'AI-generated personalized workouts' },
        { title: 'Nutrition Guidance', description: 'Smart meal planning and tracking' },
        { title: 'Progress Analysis', description: 'Intelligent progress insights' }
      ].map((feature, index) => (
        <Card key={index} className="p-6 text-center">
          <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
          <p className="text-gray-600 mb-4">{feature.description}</p>
          <Button onClick={() => console.log('Access:', feature.title)}>
            Try Now
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

export default AIAssistant;
