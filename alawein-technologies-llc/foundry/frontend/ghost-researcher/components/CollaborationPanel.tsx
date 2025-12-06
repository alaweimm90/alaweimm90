/**
 * Collaboration Panel Component
 * Real-time collaboration UI for Ghost Researcher
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  useCollaborativeDocument,
  useRealtimeUsers,
  useComments,
  useVersionHistory
} from '../hooks/useCollaboration';
import { Button } from '../../components/Button';
import { Card } from '../../components/Card';
import { Badge } from '../../components/Badge';
import { Input } from '../../components/Input';
import {
  Users,
  MessageCircle,
  Clock,
  Edit3,
  Check,
  X,
  ChevronDown,
  ChevronUp,
  Send,
  RefreshCw,
  Download,
  User
} from 'lucide-react';

interface CollaborationPanelProps {
  documentId: string;
  className?: string;
}

/**
 * Live Users List Component
 */
const LiveUsersList: React.FC<{ documentId: string }> = ({ documentId }) => {
  const { users, totalUsers } = useRealtimeUsers(documentId);
  const [showAll, setShowAll] = useState(false);

  const displayUsers = showAll ? users : users.slice(0, 3);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Active Users ({totalUsers})
        </h3>
        {totalUsers > 3 && (
          <button
            onClick={() => setShowAll(!showAll)}
            className="text-xs text-blue-600 hover:text-blue-700"
          >
            {showAll ? 'Show Less' : 'Show All'}
          </button>
        )}
      </div>

      <div className="space-y-2">
        {displayUsers.map((user) => (
          <div
            key={user.id}
            className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            <div className="relative">
              {user.avatar ? (
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-8 h-8 rounded-full"
                />
              ) : (
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-semibold"
                  style={{ backgroundColor: user.color }}
                >
                  {user.name.charAt(0).toUpperCase()}
                </div>
              )}
              <div
                className={`absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full border-2 border-white ${
                  user.status === 'online'
                    ? 'bg-green-500'
                    : user.status === 'idle'
                    ? 'bg-yellow-500'
                    : 'bg-gray-400'
                }`}
              />
            </div>

            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                {user.name}
              </p>
              {user.cursor && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Line {Math.floor(user.cursor.y / 24)}
                </p>
              )}
            </div>

            {user.cursor?.selection && (
              <Badge variant="secondary" size="sm">
                <Edit3 className="w-3 h-3" />
              </Badge>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Comments Thread Component
 */
const CommentsThread: React.FC<{ documentId: string }> = ({ documentId }) => {
  const { unresolvedComments, resolveComment } = useComments(documentId);
  const [newComment, setNewComment] = useState('');
  const [selectedText, setSelectedText] = useState<{ start: number; end: number } | null>(null);
  const { addComment } = useCollaborativeDocument(documentId);

  const handleAddComment = async () => {
    if (!newComment.trim() || !selectedText) return;

    try {
      await addComment(newComment, selectedText);
      setNewComment('');
      setSelectedText(null);
    } catch (error) {
      console.error('Failed to add comment:', error);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Comments ({unresolvedComments.length})
        </h3>
      </div>

      {/* Add Comment Form */}
      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg space-y-2">
        <Input
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Add a comment..."
          className="text-sm"
        />
        <div className="flex justify-between items-center">
          <p className="text-xs text-gray-500">
            {selectedText
              ? `Selected text: chars ${selectedText.start}-${selectedText.end}`
              : 'Select text to comment'}
          </p>
          <Button
            onClick={handleAddComment}
            size="sm"
            disabled={!newComment.trim() || !selectedText}
          >
            <Send className="w-3 h-3 mr-1" />
            Comment
          </Button>
        </div>
      </div>

      {/* Comments List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {unresolvedComments.map((comment) => (
          <div
            key={comment.id}
            className="p-3 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 space-y-2"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-2">
                {comment.userAvatar ? (
                  <img
                    src={comment.userAvatar}
                    alt={comment.userName}
                    className="w-6 h-6 rounded-full"
                  />
                ) : (
                  <div className="w-6 h-6 rounded-full bg-gray-400 flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
                <div>
                  <p className="text-xs font-medium text-gray-900 dark:text-white">
                    {comment.userName}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(comment.createdAt).toLocaleString()}
                  </p>
                </div>
              </div>

              <button
                onClick={() => resolveComment(comment.id)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
              >
                <Check className="w-4 h-4 text-green-600" />
              </button>
            </div>

            <p className="text-sm text-gray-700 dark:text-gray-300">
              {comment.content}
            </p>

            {comment.position && (
              <p className="text-xs text-gray-500">
                Line {Math.floor(comment.position.start / 80)}
              </p>
            )}

            {/* Replies */}
            {comment.replies.length > 0 && (
              <div className="ml-4 pt-2 border-t border-gray-100 dark:border-gray-800 space-y-1">
                {comment.replies.map((reply) => (
                  <div key={reply.id} className="text-sm">
                    <span className="font-medium text-gray-700 dark:text-gray-300">
                      {reply.userName}:
                    </span>{' '}
                    <span className="text-gray-600 dark:text-gray-400">
                      {reply.content}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Version History Component
 */
const VersionHistory: React.FC<{ documentId: string }> = ({ documentId }) => {
  const { versions, currentVersion } = useVersionHistory(documentId);
  const { createVersion, restoreVersion } = useCollaborativeDocument(documentId);
  const [showCreateVersion, setShowCreateVersion] = useState(false);
  const [versionMessage, setVersionMessage] = useState('');

  const handleCreateVersion = async () => {
    try {
      await createVersion(versionMessage);
      setVersionMessage('');
      setShowCreateVersion(false);
    } catch (error) {
      console.error('Failed to create version:', error);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Version History ({versions.length})
        </h3>
        <Button
          onClick={() => setShowCreateVersion(!showCreateVersion)}
          size="sm"
          variant="secondary"
        >
          <Clock className="w-3 h-3 mr-1" />
          Save Version
        </Button>
      </div>

      {/* Create Version Form */}
      {showCreateVersion && (
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg space-y-2">
          <Input
            value={versionMessage}
            onChange={(e) => setVersionMessage(e.target.value)}
            placeholder="Version description (optional)"
            className="text-sm"
          />
          <div className="flex justify-end space-x-2">
            <Button
              onClick={() => setShowCreateVersion(false)}
              size="sm"
              variant="ghost"
            >
              Cancel
            </Button>
            <Button onClick={handleCreateVersion} size="sm">
              Save Version
            </Button>
          </div>
        </div>
      )}

      {/* Versions List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {versions.map((version) => (
          <div
            key={version.id}
            className={`p-3 rounded-lg border ${
              version.version === currentVersion
                ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700'
                : 'bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    Version {version.version}
                  </p>
                  {version.version === currentVersion && (
                    <Badge variant="primary" size="sm">
                      Current
                    </Badge>
                  )}
                </div>
                {version.message && (
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {version.message}
                  </p>
                )}
                <p className="text-xs text-gray-500">
                  {version.userName} • {new Date(version.createdAt).toLocaleString()}
                </p>
              </div>

              {version.version !== currentVersion && (
                <Button
                  onClick={() => restoreVersion(version.id)}
                  size="sm"
                  variant="ghost"
                >
                  <RefreshCw className="w-3 h-3 mr-1" />
                  Restore
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Main Collaboration Panel
 */
export const CollaborationPanel: React.FC<CollaborationPanelProps> = ({
  documentId,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'users' | 'comments' | 'versions'>('users');
  const { document, isConnected, isSyncing } = useCollaborativeDocument(documentId);

  if (!document) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="flex items-center justify-center text-gray-500">
          Loading collaboration features...
        </div>
      </Card>
    );
  }

  return (
    <Card className={`${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Collaboration
          </h2>
          <div className="flex items-center space-x-2">
            {isSyncing && (
              <Badge variant="secondary" size="sm">
                <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                Syncing
              </Badge>
            )}
            <Badge
              variant={isConnected ? 'success' : 'danger'}
              size="sm"
            >
              {isConnected ? 'Connected' : 'Offline'}
            </Badge>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mt-4">
          <button
            onClick={() => setActiveTab('users')}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
              activeTab === 'users'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            <Users className="w-4 h-4 inline mr-1" />
            Users
          </button>
          <button
            onClick={() => setActiveTab('comments')}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
              activeTab === 'comments'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            <MessageCircle className="w-4 h-4 inline mr-1" />
            Comments
          </button>
          <button
            onClick={() => setActiveTab('versions')}
            className={`flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
              activeTab === 'versions'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            <Clock className="w-4 h-4 inline mr-1" />
            Versions
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {activeTab === 'users' && <LiveUsersList documentId={documentId} />}
        {activeTab === 'comments' && <CommentsThread documentId={documentId} />}
        {activeTab === 'versions' && <VersionHistory documentId={documentId} />}
      </div>
    </Card>
  );
};

export default CollaborationPanel;