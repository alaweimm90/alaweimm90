import { useState, useEffect } from 'react';
import { Play, Eye, Calendar } from 'lucide-react';

interface YouTubeVideo {
  id: string;
  title: string;
  thumbnail: string;
  views: string;
  uploadedAt: string;
  channelUrl: string;
}

export default function YouTubeIntegration() {
  const [videos, setVideos] = useState<YouTubeVideo[]>([]);
  const [loading, setLoading] = useState(true);
  const [subscriberCount, setSubscriberCount] = useState('');

  useEffect(() => {
    // Fetch latest videos from YouTube RSS feed
    // For now, using hardcoded examples - replace with real API calls
    const mockVideos = [
      {
        id: 'dQw4w9WgXcQ',
        title: 'Inside the $2M Garage: Meet the Car Collectors',
        thumbnail: 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
        views: '24K',
        uploadedAt: '2 days ago',
        channelUrl: 'https://youtube.com/@liveiticonic',
      },
      {
        id: 'jNQXAC9IVRw',
        title: 'How to Dress for the Lifestyle You Want',
        thumbnail: 'https://img.youtube.com/vi/jNQXAC9IVRw/maxresdefault.jpg',
        views: '15K',
        uploadedAt: '1 week ago',
        channelUrl: 'https://youtube.com/@liveiticonic',
      },
      {
        id: '9bZkp7q19f0',
        title: 'Lamborghini vs Ferrari Owner Interview',
        thumbnail: 'https://img.youtube.com/vi/9bZkp7q19f0/maxresdefault.jpg',
        views: '42K',
        uploadedAt: '2 weeks ago',
        channelUrl: 'https://youtube.com/@liveiticonic',
      },
    ];

    setVideos(mockVideos);
    setSubscriberCount('8.5K');
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <section className="py-16 bg-lii-black border-t border-lii-ink">
        <div className="container mx-auto px-6">
          <div className="animate-pulse">Loading videos...</div>
        </div>
      </section>
    );
  }

  return (
    <section className="py-16 bg-gradient-to-b from-lii-black to-lii-ink border-t border-lii-ink">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="mb-12 flex items-center justify-between">
          <div>
            <h2 className="text-4xl font-bold text-white mb-2">Watch Our Stories</h2>
            <p className="text-lii-text">Interviews, cars, lifestyle, and the people behind the brand</p>
          </div>
          <a
            href="https://youtube.com/@liveiticonic"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
          >
            <svg
              className="w-5 h-5"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
            </svg>
            Subscribe ({subscriberCount})
          </a>
        </div>

        {/* Video Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {videos.map((video) => (
            <a
              key={video.id}
              href={`https://youtube.com/watch?v=${video.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="group cursor-pointer"
            >
              <div className="relative mb-4 overflow-hidden rounded-lg aspect-video bg-lii-ink">
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                {/* Play button overlay */}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center">
                  <div className="w-16 h-16 rounded-full bg-red-600 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <Play className="w-7 h-7 text-white ml-1" />
                  </div>
                </div>
              </div>

              {/* Video Info */}
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-white group-hover:text-red-400 transition-colors line-clamp-2">
                  {video.title}
                </h3>

                {/* Metadata */}
                <div className="flex items-center gap-4 text-sm text-lii-text">
                  <div className="flex items-center gap-1">
                    <Eye className="w-4 h-4" />
                    {video.views}
                  </div>
                  <div className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    {video.uploadedAt}
                  </div>
                </div>
              </div>
            </a>
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-12 text-center">
          <p className="text-lii-text mb-4">New episodes every week</p>
          <a
            href="https://youtube.com/@liveiticonic"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-white hover:text-red-400 transition-colors"
          >
            View all videos
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </a>
        </div>
      </div>
    </section>
  );
}
