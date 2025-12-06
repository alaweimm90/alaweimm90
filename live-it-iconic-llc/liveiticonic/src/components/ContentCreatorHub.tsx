import { useState } from 'react';
import { Camera, Mic2, FileText, Calendar, ArrowRight } from 'lucide-react';

interface ContentPiece {
  id: string;
  type: 'interview' | 'story' | 'spotlight';
  title: string;
  description: string;
  subject: string;
  releaseDate: string;
  thumbnail: string;
  icon: React.ReactNode;
  status: 'upcoming' | 'publishing' | 'published';
}

const CONTENT_PIECES: ContentPiece[] = [
  {
    id: '1',
    type: 'interview',
    title: 'The Collector\'s Mindset',
    description: 'Meet Marcus, who owns 3 hypercars and explains why they all serve different purposes',
    subject: 'Marcus Hypercar Collector',
    releaseDate: '2024-11-15',
    thumbnail: 'https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=600&h=400&fit=crop',
    icon: <Mic2 className="w-5 h-5" />,
    status: 'upcoming',
  },
  {
    id: '2',
    type: 'story',
    title: 'From 9-5 to Full-Time Enthusiast',
    description: 'How Sarah turned her passion for cars into a thriving business',
    subject: 'Sarah\'s Journey',
    releaseDate: '2024-11-08',
    thumbnail: 'https://images.unsplash.com/photo-1488747807830-63789f68bb65?w=600&h=400&fit=crop',
    icon: <Camera className="w-5 h-5" />,
    status: 'publishing',
  },
  {
    id: '3',
    type: 'spotlight',
    title: 'The $5M Garage',
    description: 'Inside a private collection that rivals museum pieces',
    subject: 'Private Collection Tour',
    releaseDate: '2024-10-30',
    thumbnail: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=600&h=400&fit=crop',
    icon: <Camera className="w-5 h-5" />,
    status: 'published',
  },
  {
    id: '4',
    type: 'interview',
    title: 'Luxury Redefined',
    description: 'A conversation with a luxury brand designer on what elegance means today',
    subject: 'Designer Interview',
    releaseDate: '2024-11-22',
    thumbnail: 'https://images.unsplash.com/photo-1549887534-f3bda66df53d?w=600&h=400&fit=crop',
    icon: <Mic2 className="w-5 h-5" />,
    status: 'upcoming',
  },
];

export default function ContentCreatorHub() {
  const [filter, setFilter] = useState<'all' | 'upcoming' | 'publishing' | 'published'>('all');

  const filtered = filter === 'all'
    ? CONTENT_PIECES
    : CONTENT_PIECES.filter(piece => piece.status === filter);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'publishing':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'upcoming':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getStatusLabel = (status: string) => {
    return status.charAt(0).toUpperCase() + status.slice(1);
  };

  return (
    <section className="py-16 bg-lii-black border-t border-lii-ink">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <h2 className="text-4xl font-bold text-white mb-3">Content Calendar</h2>
          <p className="text-lii-text max-w-2xl">
            Stories we're documenting. Real people. Real conversations. Authentic moments that define the Iconic lifestyle.
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-3 mb-8 overflow-x-auto pb-2">
          {(['all', 'upcoming', 'publishing', 'published'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setFilter(status)}
              className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                filter === status
                  ? 'bg-lii-gold text-lii-black'
                  : 'bg-lii-ink text-lii-text hover:bg-lii-ink/80'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {filtered.map((piece) => (
            <div
              key={piece.id}
              className="group bg-lii-ink border border-transparent hover:border-lii-gold rounded-xl overflow-hidden transition-all duration-300"
            >
              {/* Image Container */}
              <div className="relative overflow-hidden aspect-video bg-lii-black">
                <img
                  src={piece.thumbnail}
                  alt={piece.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />

                {/* Type Badge */}
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/70 backdrop-blur px-3 py-2 rounded-lg">
                  <div className="text-lii-gold">{piece.icon}</div>
                  <span className="text-xs font-semibold text-white uppercase">
                    {piece.type}
                  </span>
                </div>

                {/* Status Badge */}
                <div className={`absolute top-4 right-4 px-3 py-1 rounded-full border text-xs font-semibold ${getStatusColor(piece.status)}`}>
                  {getStatusLabel(piece.status)}
                </div>
              </div>

              {/* Content Info */}
              <div className="p-6 space-y-4">
                {/* Title */}
                <h3 className="text-xl font-semibold text-white group-hover:text-lii-gold transition-colors">
                  {piece.title}
                </h3>

                {/* Description */}
                <p className="text-sm text-lii-text line-clamp-2">
                  {piece.description}
                </p>

                {/* Subject/Guest */}
                <div className="py-3 border-t border-lii-black/50">
                  <p className="text-xs text-lii-text/70 uppercase tracking-wider">Featuring</p>
                  <p className="text-white font-medium">{piece.subject}</p>
                </div>

                {/* Release Date & CTA */}
                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center gap-2 text-sm text-lii-text">
                    <Calendar className="w-4 h-4" />
                    {new Date(piece.releaseDate).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </div>

                  <button className="text-lii-gold hover:text-white transition-colors flex items-center gap-1 text-sm font-semibold">
                    View
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Empty State */}
        {filtered.length === 0 && (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 text-lii-text/30 mx-auto mb-4" />
            <p className="text-lii-text">No content in this category yet</p>
          </div>
        )}

        {/* CTA for Submissions */}
        <div className="bg-gradient-to-r from-lii-ink to-lii-black border border-lii-gold/20 rounded-xl p-8 text-center">
          <h3 className="text-2xl font-bold text-white mb-3">Know Someone Worth Featuring?</h3>
          <p className="text-lii-text mb-6 max-w-2xl mx-auto">
            We're always looking for authentic stories and interesting people to interview. If you know someone with a unique perspective on the lifestyle, let us know.
          </p>
          <button className="inline-flex items-center gap-2 bg-lii-gold hover:bg-lii-gold/90 text-lii-black font-semibold px-6 py-3 rounded-lg transition-colors">
            Suggest a Guest
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </section>
  );
}
