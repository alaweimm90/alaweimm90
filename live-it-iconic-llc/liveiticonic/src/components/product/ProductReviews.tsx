import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Star, CheckCircle, Loader2 } from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { reviewService, type Review, type ReviewStats } from '@/services/reviewService';
import { useAuth } from '@/hooks/useAuth';

interface ProductReviewsProps {
  productId: string;
  customerId?: string;
  orderId?: string;
}

/**
 * ProductReviews component displays customer reviews with ratings and allows adding new reviews
 *
 * Fetches reviews from Supabase via reviewService and displays them with verified purchase badges.
 * Includes a form to submit new reviews with star rating (1-5) and comment text.
 * Displays average rating and distribution calculated from reviewService stats.
 *
 * @component
 * @param {ProductReviewsProps} props - Component props
 * @param {string} props.productId - Product ID to fetch reviews for
 * @param {string} [props.customerId] - Optional customer ID for creating reviews
 * @param {string} [props.orderId] - Optional order ID for verified purchase badge
 *
 * @example
 * <ProductReviews
 *   productId="prod_123"
 *   customerId="customer_456"
 *   orderId="order_789"
 * />
 *
 * @remarks
 * - Reviews fetched from Supabase via reviewService
 * - Requires authentication to submit reviews
 * - Verified purchase badge shown when orderId provided
 * - Reviews require admin approval before appearing (is_approved: true)
 * - Star rating range: 1-5
 */
export const ProductReviews: React.FC<ProductReviewsProps> = ({
  productId,
  customerId,
  orderId,
}) => {
  const { user } = useAuth();
  const [reviews, setReviews] = useState<Review[]>([]);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    rating: 5,
    text: '',
  });

  // Fetch reviews and stats on mount and when productId changes
  useEffect(() => {
    const fetchReviews = async () => {
      try {
        setLoading(true);
        setError(null);
        const [reviewsData, statsData] = await Promise.all([
          reviewService.getProductReviews(productId),
          reviewService.getReviewStats(productId),
        ]);
        setReviews(reviewsData);
        setStats(statsData);
      } catch (err) {
        console.error('Failed to fetch reviews:', err);
        setError('Failed to load reviews. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchReviews();
  }, [productId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!user || !customerId) {
      setError('You must be logged in to submit a review');
      return;
    }

    if (!formData.text.trim()) {
      setError('Please enter a review');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      await reviewService.createReview({
        productId,
        customerId,
        orderId,
        rating: formData.rating,
        text: formData.text.trim(),
      });

      // Reset form
      setFormData({ rating: 5, text: '' });
      setShowForm(false);

      // Refresh reviews
      const [reviewsData, statsData] = await Promise.all([
        reviewService.getProductReviews(productId),
        reviewService.getReviewStats(productId),
      ]);
      setReviews(reviewsData);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to submit review:', err);
      setError('Failed to submit review. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <Card className="bg-lii-ink border-lii-gold/10">
        <CardContent className="py-12">
          <div className="flex items-center justify-center gap-3 text-lii-ash">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="font-ui">Loading reviews...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-lii-ink border-lii-gold/10">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lii-cloud font-display">Customer Reviews</CardTitle>
            {stats && stats.totalReviews > 0 && (
              <div className="flex items-center gap-2 mt-2">
                <div className="flex items-center">
                  {[1, 2, 3, 4, 5].map(star => (
                    <Star
                      key={star}
                      className={`w-5 h-5 ${
                        star <= stats.averageRating ? 'text-lii-gold fill-lii-gold' : 'text-lii-ash'
                      }`}
                    />
                  ))}
                </div>
                <span className="text-lii-ash font-ui text-sm">
                  {stats.averageRating.toFixed(1)} ({stats.totalReviews} reviews)
                </span>
              </div>
            )}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowForm(!showForm)}
            disabled={!user || !customerId}
          >
            {showForm ? 'Cancel' : 'Write Review'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-400 font-ui text-sm">{error}</p>
          </div>
        )}

        {showForm && (
          <form onSubmit={handleSubmit} className="space-y-4 p-4 bg-lii-charcoal/20 rounded-lg">
            <div>
              <Label className="text-lii-ash mb-2 block">Rating</Label>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map(star => (
                  <button
                    key={star}
                    type="button"
                    onClick={() => setFormData({ ...formData, rating: star })}
                    className="focus:outline-none"
                    disabled={submitting}
                  >
                    <Star
                      className={`w-6 h-6 transition-colors ${
                        star <= formData.rating
                          ? 'text-lii-gold fill-lii-gold'
                          : 'text-lii-ash hover:text-lii-gold/50'
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>
            <div>
              <Label htmlFor="reviewText" className="text-lii-ash">
                Your Review
              </Label>
              <Textarea
                id="reviewText"
                value={formData.text}
                onChange={e => setFormData({ ...formData, text: e.target.value })}
                required
                disabled={submitting}
                className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud min-h-[100px]"
                placeholder="Share your experience with this product..."
              />
            </div>
            {orderId && (
              <div className="flex items-center gap-2 text-sm text-lii-ash">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Verified purchase - this review will be marked as verified</span>
              </div>
            )}
            <Button type="submit" variant="primary" disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Submitting...
                </>
              ) : (
                'Submit Review'
              )}
            </Button>
          </form>
        )}

        {reviews.length === 0 && !showForm ? (
          <p className="text-lii-ash font-ui text-center py-8">
            No reviews yet. Be the first to review this product!
          </p>
        ) : (
          <div className="space-y-4">
            {reviews.map(review => (
              <div key={review.id} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-wrap">
                    <p className="text-lii-cloud font-ui font-medium">
                      {review.customer_name || 'Anonymous'}
                    </p>
                    {review.is_verified_purchase && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-500/10 border border-green-500/20 rounded text-green-400 text-xs font-ui">
                        <CheckCircle className="h-3 w-3" />
                        Verified Purchase
                      </span>
                    )}
                    <div className="flex items-center">
                      {[1, 2, 3, 4, 5].map(star => (
                        <Star
                          key={star}
                          className={`w-4 h-4 ${
                            star <= review.rating ? 'text-lii-gold fill-lii-gold' : 'text-lii-ash'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                  <p className="text-lii-ash font-ui text-xs">
                    {new Date(review.created_at).toLocaleDateString()}
                  </p>
                </div>
                <p className="text-lii-ash font-ui text-sm">{review.text}</p>
                <Separator className="bg-lii-gold/10" />
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ProductReviews;
