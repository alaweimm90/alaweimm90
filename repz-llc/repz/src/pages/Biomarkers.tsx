import React from 'react';
import { Card } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';

const Biomarkers = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900">
            Biomarkers
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Track and analyze your health biomarkers for optimal performance
          </p>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[
        { metric: 'Heart Rate Variability', value: '45ms', status: 'Good' },
        { metric: 'Sleep Quality Score', value: '82/100', status: 'Excellent' },
        { metric: 'Recovery Index', value: '7.5/10', status: 'Good' }
      ].map((biomarker, index) => (
        <Card key={index} className="p-6">
          <h3 className="text-lg font-semibold mb-2">{biomarker.metric}</h3>
          <p className="text-3xl font-bold text-blue-600 mb-2">{biomarker.value}</p>
          <p className={`text-sm font-medium ${
            biomarker.status === 'Excellent' ? 'text-green-600' : 
            biomarker.status === 'Good' ? 'text-blue-600' : 'text-yellow-600'
          }`}>
            {biomarker.status}
          </p>
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

export default Biomarkers;
