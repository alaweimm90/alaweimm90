import React from 'react';
import { ExternalLink, Download, Quote, Users, Calendar } from 'lucide-react';
import { Publication } from '../../types/academic.types';
import { format } from 'date-fns';

interface PublicationCardProps {
  publication: Publication;
  onViewDetails: (id: string) => void;
  onDownloadPDF: (url: string) => void;
  compact?: boolean;
  showAbstract?: boolean;
}

export const PublicationCard: React.FC<PublicationCardProps> = ({
  publication,
  onViewDetails,
  onDownloadPDF,
  compact = false,
  showAbstract = true
}) => {
  const getStatusColor = (status: PublicationStatus) => {
    switch (status) {
      case 'published':
        return 'bg-green-100 text-green-800';
      case 'in-review':
        return 'bg-yellow-100 text-yellow-800';
      case 'preprint':
        return 'bg-blue-100 text-blue-800';
      case 'draft':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getTypeIcon = (type: PublicationType) => {
    switch (type) {
      case 'journal-article':
        return <Quote className="w-4 h-4" />;
      case 'conference-paper':
        return <Users className="w-4 h-4" />;
      default:
        return <ExternalLink className="w-4 h-4" />;
    }
  };

  return (
    <article className={`publication-card ${compact ? 'p-4' : 'p-6'}`}>
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            {getTypeIcon(publication.type)}
            <span className={`text-xs font-medium px-2 py-1 rounded-full ${getStatusColor(publication.status)}`}>
              {publication.status.replace('-', ' ').toUpperCase()}
            </span>
            <span className="text-xs text-gray-500">
              {publication.type.replace('-', ' ').toUpperCase()}
            </span>
          </div>
          <h3 className={`font-heading text-gray-900 leading-tight ${
            compact ? 'text-base' : 'text-lg'
          }`}>
            {publication.title}
          </h3>
        </div>
        <div className="flex flex-col items-end ml-4">
          <span className="text-sm text-gray-500 whitespace-nowrap flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            {format(publication.date, 'yyyy')}
          </span>
          {!compact && (
            <span className="text-sm font-medium text-academic-green mt-1">
              {publication.citations} citations
            </span>
          )}
        </div>
      </div>

      {/* Abstract */}
      {showAbstract && publication.abstract && (
        <p className="text-gray-600 mb-4 line-clamp-3 text-sm leading-relaxed">
          {publication.abstract}
        </p>
      )}

      {/* Journal and Metadata */}
      <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
        <div>
          <span className="font-medium">{publication.journal}</span>
          {publication.volume && (
            <span className="ml-2">
              Vol {publication.volume}
              {publication.issue && `, Issue ${publication.issue}`}
              {publication.pages && `, pp. ${publication.pages}`}
            </span>
          )}
        </div>
        {publication.doi && (
          <a
            href={`https://doi.org/${publication.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-academic-blue hover:text-academic-blue/80 flex items-center gap-1"
          >
            DOI
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>

      {/* Authors */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex flex-wrap gap-2">
          {publication.authors.slice(0, compact ? 2 : 3).map((author, index) => (
            <span
              key={index}
              className={`text-xs px-2 py-1 rounded ${
                author.isCorresponding
                  ? 'bg-academic-blue text-white'
                  : 'bg-gray-100 text-gray-700'
              }`}
            >
              {author.name}
              {author.isCorresponding && '*'}
            </span>
          ))}
          {publication.authors.length > (compact ? 2 : 3) && (
            <span className="text-xs text-gray-500">
              +{publication.authors.length - (compact ? 2 : 3)} more
            </span>
          )}
        </div>
      </div>

      {/* Keywords */}
      {!compact && publication.keywords.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-4">
          {publication.keywords.slice(0, 4).map((keyword, index) => (
            <span
              key={index}
              className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded"
            >
              {keyword}
            </span>
          ))}
          {publication.keywords.length > 4 && (
            <span className="text-xs text-gray-500">
              +{publication.keywords.length - 4} more
            </span>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-100">
        <div className="flex gap-2">
          <button
            onClick={() => onViewDetails(publication.id)}
            className="text-academic-blue hover:text-academic-blue/80 text-sm font-medium focus-ring px-2 py-1 rounded"
          >
            View Details
          </button>
          {publication.pdfUrl && (
            <button
              onClick={() => onDownloadPDF(publication.pdfUrl)}
              className="text-academic-green hover:text-academic-green/80 text-sm font-medium focus-ring px-2 py-1 rounded flex items-center gap-1"
            >
              <Download className="w-3 h-3" />
              PDF
            </button>
          )}
        </div>

        {/* Citation Badge */}
        {!compact && (
          <div className="citation-badge">
            {publication.citations} citations
          </div>
        )}
      </div>
    </article>
  );
};

export default PublicationCard;
