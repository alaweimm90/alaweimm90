import { useState, useEffect } from 'react';
import { Radio, Users, Eye } from 'lucide-react';

interface TwitchStream {
  isLive: boolean;
  username: string;
  title: string;
  thumbnail: string;
  viewerCount: number;
  gameCategory: string;
  channelUrl: string;
  uptime: string;
}

export default function TwitchLiveIntegration() {
  const [stream, setStream] = useState<TwitchStream | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In production, fetch from Twitch API
    // For now, using mock data
    const checkLiveStatus = async () => {
      // Replace with actual Twitch API call
      const mockStream: TwitchStream = {
        isLive: true, // Toggle based on schedule
        username: 'liveiticonic',
        title: 'LIVE: Supercar Unboxing & Interview with Owner',
        thumbnail: 'https://static-cdn.jtvnw.net/previews-ttv/live_user_liveiticonic-320x180.jpg',
        viewerCount: 342,
        gameCategory: 'Just Chatting',
        channelUrl: 'https://twitch.tv/liveiticonic',
        uptime: '2h 15m',
      };

      // Simulate API call delay
      setTimeout(() => {
        setStream(mockStream);
        setLoading(false);
      }, 500);
    };

    checkLiveStatus();

    // Poll every 30 seconds to check if live status changed
    const interval = setInterval(checkLiveStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return null;
  }

  if (!stream?.isLive) {
    return null; // Don't show section when not live
  }

  return (
    <section className="fixed bottom-6 right-6 w-80 bg-lii-black border border-lii-ink rounded-xl shadow-2xl z-40 overflow-hidden hover:border-purple-500 transition-colors">
      {/* Live Badge */}
      <a
        href={stream.channelUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="block hover:opacity-90 transition-opacity"
      >
        {/* Thumbnail */}
        <div className="relative overflow-hidden aspect-video bg-lii-ink">
          <img
            src={stream.thumbnail}
            alt={stream.title}
            className="w-full h-full object-cover"
          />

          {/* Live indicator */}
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full">
            <Radio className="w-3 h-3 text-white animate-pulse" />
            <span className="text-xs font-bold text-white uppercase">LIVE</span>
          </div>

          {/* Viewer count overlay */}
          <div className="absolute bottom-3 right-3 bg-black/70 px-2 py-1 rounded text-xs text-white flex items-center gap-1">
            <Users className="w-3 h-3" />
            {stream.viewerCount.toLocaleString()}
          </div>
        </div>

        {/* Stream Info */}
        <div className="p-4 space-y-3">
          {/* Title */}
          <h3 className="font-semibold text-white line-clamp-2 hover:text-purple-400 transition-colors">
            {stream.title}
          </h3>

          {/* Metadata */}
          <div className="space-y-2 text-xs text-lii-text">
            <div className="flex items-center justify-between">
              <span>{stream.gameCategory}</span>
              <span>{stream.uptime} online</span>
            </div>
          </div>

          {/* CTA Button */}
          <button className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 rounded-lg transition-colors flex items-center justify-center gap-2">
            <Play className="w-4 h-4" />
            Watch Live
          </button>
        </div>

        {/* Notification for users who close this */}
      </a>
    </section>
  );
}

function Play({ className }: { className: string }) {
  return (
    <svg
      className={className}
      fill="currentColor"
      viewBox="0 0 24 24"
    >
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}
