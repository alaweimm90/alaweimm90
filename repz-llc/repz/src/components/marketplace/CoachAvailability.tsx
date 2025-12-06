import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Calendar, Clock, MapPin, Video, Phone, MessageSquare } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/ui/atoms/Label';

interface TimeSlot {
  id: string;
  date: string;
  time: string;
  duration: number;
  type: 'consultation' | 'training' | 'checkin';
  isAvailable: boolean;
  price: number;
}

interface BookingRequest {
  timeSlotId: string;
  sessionType: string;
  message: string;
  contactMethod: string;
}

const mockTimeSlots: TimeSlot[] = [
  {
    id: '1',
    date: '2024-01-20',
    time: '09:00',
    duration: 60,
    type: 'consultation',
    isAvailable: true,
    price: 75
  },
  {
    id: '2',
    date: '2024-01-20',
    time: '14:00',
    duration: 30,
    type: 'checkin',
    isAvailable: true,
    price: 40
  },
  {
    id: '3',
    date: '2024-01-21',
    time: '10:00',
    duration: 90,
    type: 'training',
    isAvailable: true,
    price: 120
  },
  {
    id: '4',
    date: '2024-01-21',
    time: '15:30',
    duration: 60,
    type: 'consultation',
    isAvailable: false,
    price: 75
  },
  {
    id: '5',
    date: '2024-01-22',
    time: '08:00',
    duration: 60,
    type: 'training',
    isAvailable: true,
    price: 100
  }
];

interface CoachAvailabilityProps {
  coachId: string;
  coachName: string;
}

export const CoachAvailability: React.FC<CoachAvailabilityProps> = ({
  coachId,
  coachName
}) => {
  const [selectedTimeSlot, setSelectedTimeSlot] = useState<TimeSlot | null>(null);
  const [bookingRequest, setBookingRequest] = useState<BookingRequest>({
    timeSlotId: '',
    sessionType: '',
    message: '',
    contactMethod: 'video'
  });
  const [showBookingDialog, setShowBookingDialog] = useState(false);

  const getSessionTypeIcon = (type: string) => {
    switch (type) {
      case 'consultation':
        return <MessageSquare className="w-4 h-4" />;
      case 'training':
        return <Video className="w-4 h-4" />;
      case 'checkin':
        return <Phone className="w-4 h-4" />;
      default:
        return <Calendar className="w-4 h-4" />;
    }
  };

  const getSessionTypeColor = (type: string) => {
    switch (type) {
      case 'consultation':
        return 'bg-blue-100 text-blue-800';
      case 'training':
        return 'bg-green-100 text-green-800';
      case 'checkin':
        return 'bg-repz-orange/10 text-repz-orange';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const handleBookSlot = (timeSlot: TimeSlot) => {
    setSelectedTimeSlot(timeSlot);
    setBookingRequest({
      ...bookingRequest,
      timeSlotId: timeSlot.id,
      sessionType: timeSlot.type
    });
    setShowBookingDialog(true);
  };

  const handleSubmitBooking = () => {
    console.log('Booking request:', bookingRequest);
    // Here you would typically send the booking request to your backend
    setShowBookingDialog(false);
    setSelectedTimeSlot(null);
    setBookingRequest({
      timeSlotId: '',
      sessionType: '',
      message: '',
      contactMethod: 'video'
    });
  };

  const groupSlotsByDate = (slots: TimeSlot[]) => {
    return slots.reduce((groups, slot) => {
      const date = slot.date;
      if (!groups[date]) {
        groups[date] = [];
      }
      groups[date].push(slot);
      return groups;
    }, {} as Record<string, TimeSlot[]>);
  };

  const groupedSlots = groupSlotsByDate(mockTimeSlots);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-xl font-semibold mb-2">
          Book a Session with {coachName}
        </h3>
        <p className="text-muted-foreground">
          Choose from available time slots below. All times are in your local timezone.
        </p>
      </div>

      {/* Session Types Legend */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Session Types</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 p-3 border rounded-lg">
              <MessageSquare className="w-5 h-5 text-blue-600" />
              <div>
                <h4 className="font-medium">Consultation</h4>
                <p className="text-sm text-muted-foreground">
                  Initial assessment and goal setting
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 border rounded-lg">
              <Video className="w-5 h-5 text-green-600" />
              <div>
                <h4 className="font-medium">Training Session</h4>
                <p className="text-sm text-muted-foreground">
                  Live workout guidance and form checking
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 border rounded-lg">
              <Phone className="w-5 h-5 text-repz-orange" />
              <div>
                <h4 className="font-medium">Check-in</h4>
                <p className="text-sm text-muted-foreground">
                  Progress review and plan adjustments
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Available Time Slots */}
      <div className="space-y-4">
        {Object.entries(groupedSlots).map(([date, slots]) => (
          <Card key={date}>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                {formatDate(date)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                {slots.map((slot) => (
                  <div
                    key={slot.id}
                    className={`p-4 border rounded-lg ${
                      slot.isAvailable
                        ? 'hover:border-primary cursor-pointer'
                        : 'opacity-50 cursor-not-allowed'
                    }`}
                    onClick={() => slot.isAvailable && handleBookSlot(slot)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getSessionTypeIcon(slot.type)}
                        <span className="font-medium">{slot.time}</span>
                      </div>
                      <Badge className={getSessionTypeColor(slot.type)}>
                        {slot.type}
                      </Badge>
                    </div>
                    
                    <div className="text-sm text-muted-foreground mb-2">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {slot.duration} minutes
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="font-medium">${slot.price}</span>
                      {!slot.isAvailable && (
                        <span className="text-xs text-red-600">Booked</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Booking Dialog */}
      <Dialog open={showBookingDialog} onOpenChange={setShowBookingDialog}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Book Session with {coachName}</DialogTitle>
            <DialogDescription>
              {selectedTimeSlot && (
                <>
                  {formatDate(selectedTimeSlot.date)} at {selectedTimeSlot.time} 
                  ({selectedTimeSlot.duration} minutes) - ${selectedTimeSlot.price}
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label>Session Type</Label>
              <Select value={bookingRequest.sessionType} onValueChange={(value) => 
                setBookingRequest({ ...bookingRequest, sessionType: value })
              }>
                <SelectTrigger>
                  <SelectValue placeholder="Select session type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="consultation">Consultation</SelectItem>
                  <SelectItem value="training">Training Session</SelectItem>
                  <SelectItem value="checkin">Check-in</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label>Preferred Contact Method</Label>
              <Select value={bookingRequest.contactMethod} onValueChange={(value) => 
                setBookingRequest({ ...bookingRequest, contactMethod: value })
              }>
                <SelectTrigger>
                  <SelectValue placeholder="Select contact method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="video">Video Call</SelectItem>
                  <SelectItem value="phone">Phone Call</SelectItem>
                  <SelectItem value="in-person">In-Person</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="booking-message">Message (Optional)</Label>
              <Textarea
                id="booking-message"
                placeholder="Let the coach know what you'd like to focus on during this session..."
                value={bookingRequest.message}
                onChange={(e) => setBookingRequest({ 
                  ...bookingRequest, 
                  message: e.target.value 
                })}
                rows={3}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowBookingDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSubmitBooking}>
              Request Booking - ${selectedTimeSlot?.price}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {Object.keys(groupedSlots).length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Calendar className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground mb-4">
              No available time slots at the moment.
            </p>
            <Button variant="outline" onClick={() => console.log("CoachAvailability button clicked")}>
              Join Waitlist
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};