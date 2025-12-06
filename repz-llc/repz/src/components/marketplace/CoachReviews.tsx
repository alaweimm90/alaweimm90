import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Avatar, AvatarFallback, AvatarImage } from '@/ui/atoms/Avatar';
import { Badge } from '@/ui/atoms/Badge';
import { Star, ThumbsUp, Flag } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { Label } from '@/ui/atoms/Label';

interface Review {
  id: string;
  clientName: string;
  clientAvatar: string;
  rating: number;
  title: string;
  content: string;
  date: string;
  verified: boolean;
  helpful: number;
  tags: string[];
}

interface ReviewFormData {
  rating: number;
  title: string;
  content: string;
}

const mockReviews: Review[] = [
  {
    id: '1',
    clientName: 'Jessica M.',
    clientAvatar: '/placeholder-avatar.jpg',
    rating: 5,
    title: 'Amazing transformation coach!',
    content: 'Sarah helped me lose 30 pounds in 6 months. Her nutrition guidance and workout plans were exactly what I needed. She was always responsive and motivated me through tough times.',
    date: '2024-01-15',
    verified: true,
    helpful: 12,
    tags: ['Weight Loss', 'Nutrition', 'Supportive']
  },
  {
    id: '2',
    clientName: 'Mark T.',
    clientAvatar: '/placeholder-avatar.jpg',
    rating: 5,
    title: 'Professional and knowledgeable',
    content: 'Working with Sarah for strength training has been incredible. She knows exactly how to push you while keeping safety first. My lifts have improved dramatically.',
    date: '2024-01-10',
    verified: true,
    helpful: 8,
    tags: ['Strength Training', 'Safety', 'Results']
  },
  {
    id: '3',
    clientName: 'Amanda K.',
    clientAvatar: '/placeholder-avatar.jpg',
    rating: 4,
    title: 'Great coach, minor communication delays',
    content: 'Sarah is very knowledgeable and her programs work. Sometimes responses took longer than expected, but the quality of guidance made up for it.',
    date: '2024-01-05',
    verified: true,
    helpful: 3,
    tags: ['Knowledgeable', 'Effective Programs']
  }
];

interface CoachReviewsProps {
  coachId: string;
  coachName: string;
  averageRating: number;
  totalReviews: number;
}

export const CoachReviews: React.FC<CoachReviewsProps> = ({
  coachId,
  coachName,
  averageRating,
  totalReviews
}) => {
  const [reviews, setReviews] = useState<Review[]>(mockReviews);
  const [showReviewForm, setShowReviewForm] = useState(false);
  const [newReview, setNewReview] = useState<ReviewFormData>({
    rating: 5,
    title: '',
    content: ''
  });

  const handleSubmitReview = () => {
    const review: Review = {
      id: Date.now().toString(),
      clientName: 'You',
      clientAvatar: '/placeholder-avatar.jpg',
      rating: newReview.rating,
      title: newReview.title,
      content: newReview.content,
      date: new Date().toISOString().split('T')[0],
      verified: true,
      helpful: 0,
      tags: []
    };

    setReviews([review, ...reviews]);
    setNewReview({ rating: 5, title: '', content: '' });
    setShowReviewForm(false);
  };

  const handleHelpful = (reviewId: string) => {
    setReviews(reviews.map(review => 
      review.id === reviewId 
        ? { ...review, helpful: review.helpful + 1 }
        : review
    ));
  };

  const renderStars = (rating: number, interactive = false, onRate?: (rating: number) => void) => {
    return (
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`w-4 h-4 ${
              star <= rating 
                ? 'fill-yellow-400 text-yellow-400' 
                : 'text-gray-300'
            } ${interactive ? 'cursor-pointer hover:text-yellow-400' : ''}`}
            onClick={() => interactive && onRate && onRate(star)}
          />
        ))}
      </div>
    );
  };

  const getRatingDistribution = () => {
    const distribution = { 5: 0, 4: 0, 3: 0, 2: 0, 1: 0 };
    reviews.forEach(review => {
      distribution[review.rating as keyof typeof distribution]++;
    });
    return distribution;
  };

  const ratingDistribution = getRatingDistribution();

  return (
    <div className="space-y-6">
      {/* Reviews Header */}
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-xl font-semibold mb-2">
            Reviews for {coachName}
          </h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {renderStars(averageRating)}
              <span className="text-2xl font-bold">{averageRating}</span>
              <span className="text-muted-foreground">
                ({totalReviews} reviews)
              </span>
            </div>
          </div>
        </div>
        
        <Dialog open={showReviewForm} onOpenChange={setShowReviewForm}>
          <DialogTrigger asChild>
            <Button onClick={() => console.log("CoachReviews button clicked")}>Write a Review</Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[500px]">
            <DialogHeader>
              <DialogTitle>Write a Review for {coachName}</DialogTitle>
              <DialogDescription>
                Share your experience to help other clients make informed decisions.
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
              <div>
                <Label>Rating</Label>
                {renderStars(newReview.rating, true, (rating) => 
                  setNewReview({ ...newReview, rating })
                )}
              </div>
              
              <div>
                <Label htmlFor="review-title">Review Title</Label>
                <input
                  id="review-title"
                  type="text"
                  placeholder="Summarize your experience..."
                  value={newReview.title}
                  onChange={(e) => setNewReview({ ...newReview, title: e.target.value })}
                  className="w-full px-3 py-2 border rounded-md"
                />
              </div>
              
              <div>
                <Label htmlFor="review-content">Your Review</Label>
                <Textarea
                  id="review-content"
                  placeholder="Tell others about your experience with this coach..."
                  value={newReview.content}
                  onChange={(e) => setNewReview({ ...newReview, content: e.target.value })}
                  rows={4}
                />
              </div>
            </div>
            
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowReviewForm(false)}>
                Cancel
              </Button>
              <Button 
                onClick={handleSubmitReview}
                disabled={!newReview.title || !newReview.content}
              >
                Submit Review
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Rating Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Rating Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[5, 4, 3, 2, 1].map((rating) => {
              const count = ratingDistribution[rating as keyof typeof ratingDistribution];
              const percentage = reviews.length > 0 ? (count / reviews.length) * 100 : 0;
              
              return (
                <div key={rating} className="flex items-center gap-3">
                  <div className="flex items-center gap-1 w-12">
                    <span className="text-sm">{rating}</span>
                    <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                  </div>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-400 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <span className="text-sm text-muted-foreground w-8">
                    {count}
                  </span>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Individual Reviews */}
      <div className="space-y-4">
        {reviews.map((review) => (
          <Card key={review.id}>
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <Avatar className="w-10 h-10">
                    <AvatarImage src={review.clientAvatar} alt={review.clientName} />
                    <AvatarFallback>{review.clientName[0]}</AvatarFallback>
                  </Avatar>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{review.clientName}</span>
                      {review.verified && (
                        <Badge variant="secondary" className="text-xs">
                          Verified
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {renderStars(review.rating)}
                      <span className="text-sm text-muted-foreground">
                        {new Date(review.date).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={() => console.log("CoachReviews button clicked")}>
                  <Flag className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-3">
              <CardTitle className="text-base">{review.title}</CardTitle>
              <CardDescription>{review.content}</CardDescription>
              
              {review.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {review.tags.map((tag) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}
              
              <div className="flex items-center justify-between pt-2">
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => handleHelpful(review.id)}
                  className="flex items-center gap-2"
                >
                  <ThumbsUp className="w-4 h-4" />
                  Helpful ({review.helpful})
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {reviews.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <p className="text-muted-foreground mb-4">
              No reviews yet. Be the first to review this coach!
            </p>
            <Button onClick={() => setShowReviewForm(true)}>
              Write the First Review
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};