import React from 'react';
import { Card } from '@/components/ui/card';
import { CalendlyBooking } from './CalendlyBooking';
import { Calendar, MapPin, Home, Building2, Video } from 'lucide-react';

export const BookingSection: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-4">Book Your Training Session</h2>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Choose your preferred training location and schedule a session with our expert coaches
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Free Consultation */}
        <Card className="p-6 bg-gradient-to-br from-gray-900 to-black border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <Calendar className="h-8 w-8 text-orange-500" />
            <h3 className="text-xl font-semibold">Free Consultation</h3>
          </div>
          <p className="text-gray-400 mb-6">
            Discover how REPZ can transform your fitness journey with a personalized consultation
          </p>
          <CalendlyBooking
            eventType="consultation"
            buttonText="Book Free Consultation"
            embed={true}
            className="w-full"
          />
        </Card>

        {/* Gym Training */}
        <Card className="p-6 bg-gradient-to-br from-gray-900 to-black border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <MapPin className="h-8 w-8 text-blue-500" />
            <h3 className="text-xl font-semibold">Train at Your Gym</h3>
          </div>
          <p className="text-gray-400 mb-6">
            Meet your coach at your preferred gym location for personalized training
          </p>
          <CalendlyBooking
            eventType="gymTraining"
            buttonText="Book Gym Session"
            embed={true}
            className="w-full"
          />
        </Card>

        {/* Home Training */}
        <Card className="p-6 bg-gradient-to-br from-gray-900 to-black border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <Home className="h-8 w-8 text-green-500" />
            <h3 className="text-xl font-semibold">Train at Home</h3>
          </div>
          <p className="text-gray-400 mb-6">
            Private coaching in the comfort of your own home with your equipment
          </p>
          <CalendlyBooking
            eventType="homeTraining"
            buttonText="Book Home Session"
            embed={true}
            className="w-full"
          />
        </Card>

        {/* City Sports Club */}
        <Card className="p-6 bg-gradient-to-br from-gray-900 to-black border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <Building2 className="h-8 w-8 text-purple-500" />
            <h3 className="text-xl font-semibold">City Sports Club</h3>
          </div>
          <p className="text-gray-400 mb-6">
            Premium facility access with parking validation in the Bay Area
          </p>
          <CalendlyBooking
            eventType="cityTraining"
            buttonText="Book at City Sports"
            embed={true}
            className="w-full"
          />
        </Card>

        {/* Virtual Check-in */}
        <Card className="p-6 bg-gradient-to-br from-gray-900 to-black border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <Video className="h-8 w-8 text-pink-500" />
            <h3 className="text-xl font-semibold">Virtual Check-In</h3>
          </div>
          <p className="text-gray-400 mb-6">
            Weekly progress reviews and coaching adjustments via video call
          </p>
          <CalendlyBooking
            eventType="virtualCheckIn"
            buttonText="Book Virtual Session"
            embed={true}
            className="w-full"
          />
        </Card>
      </div>
    </div>
  );
};