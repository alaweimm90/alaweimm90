/**
 * Collaboration Hooks for Ghost Researcher
 * Real-time document collaboration, comments, and version history
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { create } from 'zustand';
import { getSocket } from '../../lib/socket';
import apiClient from '../../lib/api/client';
import { API_ENDPOINTS } from '../../lib/api/endpoints';

// Types
export interface CollaborativeUser {
  id: string;
  name: string;
  avatar?: string;
  color: string;
  cursor?: {
    x: number;
    y: number;
    selection?: { start: number; end: number };
  };
  status: 'online' | 'idle' | 'offline';
  lastSeen: Date;
}

export interface DocumentComment {
  id: string;
  userId: string;
  userName: string;
  userAvatar?: string;
  content: string;
  position: { start: number; end: number };
  resolved: boolean;
  replies: DocumentComment[];
  createdAt: Date;
  updatedAt: Date;
}

export interface DocumentVersion {
  id: string;
  version: number;
  content: string;
  changes: {
    type: 'add' | 'remove' | 'modify';
    position: number;
    content: string;
  }[];
  userId: string;
  userName: string;
  message?: string;
  createdAt: Date;
}

export interface CollaborativeDocument {
  id: string;
  title: string;
  content: string;
  users: CollaborativeUser[];
  comments: DocumentComment[];
  versions: DocumentVersion[];
  currentVersion: number;
  isLocked: boolean;
  lockedBy?: string;
  createdAt: Date;
  updatedAt: Date;
}

// Collaboration Store
interface CollaborationStore {
  documents: Map<string, CollaborativeDocument>;
  activeDocument: string | null;
  users: CollaborativeUser[];
  pendingChanges: any[];

  // Actions
  setActiveDocument: (documentId: string | null) => void;
  updateDocument: (documentId: string, updates: Partial<CollaborativeDocument>) => void;
  addUser: (user: CollaborativeUser) => void;
  removeUser: (userId: string) => void;
  updateUserCursor: (userId: string, cursor: CollaborativeUser['cursor']) => void;
  addComment: (documentId: string, comment: DocumentComment) => void;
  resolveComment: (documentId: string, commentId: string) => void;
  addVersion: (documentId: string, version: DocumentVersion) => void;
}

const useCollaborationStore = create<CollaborationStore>((set, get) => ({
  documents: new Map(),
  activeDocument: null,
  users: [],
  pendingChanges: [],

  setActiveDocument: (documentId) => {
    set({ activeDocument: documentId });
  },

  updateDocument: (documentId, updates) => {
    set((state) => {
      const documents = new Map(state.documents);
      const doc = documents.get(documentId);
      if (doc) {
        documents.set(documentId, { ...doc, ...updates });
      }
      return { documents };
    });
  },

  addUser: (user) => {
    set((state) => ({
      users: [...state.users.filter((u) => u.id !== user.id), user],
    }));
  },

  removeUser: (userId) => {
    set((state) => ({
      users: state.users.filter((u) => u.id !== userId),
    }));
  },

  updateUserCursor: (userId, cursor) => {
    set((state) => ({
      users: state.users.map((u) =>
        u.id === userId ? { ...u, cursor } : u
      ),
    }));
  },

  addComment: (documentId, comment) => {
    set((state) => {
      const documents = new Map(state.documents);
      const doc = documents.get(documentId);
      if (doc) {
        documents.set(documentId, {
          ...doc,
          comments: [...doc.comments, comment],
        });
      }
      return { documents };
    });
  },

  resolveComment: (documentId, commentId) => {
    set((state) => {
      const documents = new Map(state.documents);
      const doc = documents.get(documentId);
      if (doc) {
        documents.set(documentId, {
          ...doc,
          comments: doc.comments.map((c) =>
            c.id === commentId ? { ...c, resolved: true } : c
          ),
        });
      }
      return { documents };
    });
  },

  addVersion: (documentId, version) => {
    set((state) => {
      const documents = new Map(state.documents);
      const doc = documents.get(documentId);
      if (doc) {
        documents.set(documentId, {
          ...doc,
          versions: [...doc.versions, version],
          currentVersion: version.version,
        });
      }
      return { documents };
    });
  },
}));

/**
 * Hook for collaborative document editing
 */
export function useCollaborativeDocument(documentId: string) {
  const socket = getSocket();
  const store = useCollaborationStore();
  const [isConnected, setIsConnected] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const syncTimeout = useRef<NodeJS.Timeout>();

  const document = store.documents.get(documentId);

  // Load document
  useEffect(() => {
    const loadDocument = async () => {
      try {
        const response = await apiClient.get(
          API_ENDPOINTS.ghostResearcher.collaboration.getProject(documentId)
        );
        store.updateDocument(documentId, response.data);
      } catch (err: any) {
        setError(err.message);
      }
    };

    loadDocument();
  }, [documentId]);

  // Connect to WebSocket room
  useEffect(() => {
    if (!socket) return;

    socket.joinRoom(documentId);
    setIsConnected(true);

    // Listen for document updates
    const handleUpdate = (data: any) => {
      store.updateDocument(documentId, data.updates);
    };

    // Listen for user joins/leaves
    const handleUserJoin = (user: CollaborativeUser) => {
      store.addUser(user);
    };

    const handleUserLeave = (userId: string) => {
      store.removeUser(userId);
    };

    // Listen for cursor movements
    const handleCursorMove = (data: { userId: string; cursor: any }) => {
      store.updateUserCursor(data.userId, data.cursor);
    };

    // Listen for new comments
    const handleNewComment = (comment: DocumentComment) => {
      store.addComment(documentId, comment);
    };

    // Listen for version updates
    const handleNewVersion = (version: DocumentVersion) => {
      store.addVersion(documentId, version);
    };

    socket.on('document:update', handleUpdate);
    socket.on('user:join', handleUserJoin);
    socket.on('user:leave', handleUserLeave);
    socket.on('cursor:move', handleCursorMove);
    socket.on('comment:new', handleNewComment);
    socket.on('version:new', handleNewVersion);

    return () => {
      socket.off('document:update', handleUpdate);
      socket.off('user:join', handleUserJoin);
      socket.off('user:leave', handleUserLeave);
      socket.off('cursor:move', handleCursorMove);
      socket.off('comment:new', handleNewComment);
      socket.off('version:new', handleNewVersion);
      socket.leaveRoom();
      setIsConnected(false);
    };
  }, [socket, documentId]);

  // Update document content
  const updateContent = useCallback(
    (content: string, immediate = false) => {
      if (!socket || !document) return;

      // Update local state immediately
      store.updateDocument(documentId, { content });

      // Clear existing sync timeout
      if (syncTimeout.current) {
        clearTimeout(syncTimeout.current);
      }

      const sync = () => {
        setIsSyncing(true);
        socket.send('document:update', {
          documentId,
          content,
        });

        // Also save to backend
        apiClient
          .patch(
            API_ENDPOINTS.ghostResearcher.collaboration.getProject(documentId),
            { content }
          )
          .finally(() => setIsSyncing(false));
      };

      if (immediate) {
        sync();
      } else {
        // Debounce updates
        syncTimeout.current = setTimeout(sync, 500);
      }
    },
    [socket, document, documentId]
  );

  // Update cursor position
  const updateCursor = useCallback(
    (cursor: CollaborativeUser['cursor']) => {
      if (!socket) return;

      socket.send('cursor:move', {
        documentId,
        cursor,
      });
    },
    [socket, documentId]
  );

  // Add comment
  const addComment = useCallback(
    async (content: string, position: { start: number; end: number }) => {
      try {
        const response = await apiClient.post(
          API_ENDPOINTS.ghostResearcher.collaboration.comments(documentId),
          {
            content,
            position,
          }
        );

        const comment = response.data;
        store.addComment(documentId, comment);

        if (socket) {
          socket.send('comment:new', {
            documentId,
            comment,
          });
        }

        return comment;
      } catch (err: any) {
        setError(err.message);
        throw err;
      }
    },
    [socket, documentId]
  );

  // Create version/snapshot
  const createVersion = useCallback(
    async (message?: string) => {
      try {
        const response = await apiClient.post(
          API_ENDPOINTS.ghostResearcher.notes.versions(documentId),
          {
            content: document?.content,
            message,
          }
        );

        const version = response.data;
        store.addVersion(documentId, version);

        if (socket) {
          socket.send('version:new', {
            documentId,
            version,
          });
        }

        return version;
      } catch (err: any) {
        setError(err.message);
        throw err;
      }
    },
    [socket, document, documentId]
  );

  // Restore version
  const restoreVersion = useCallback(
    async (versionId: string) => {
      try {
        const version = document?.versions.find((v) => v.id === versionId);
        if (!version) throw new Error('Version not found');

        updateContent(version.content, true);
        await createVersion(`Restored from version ${version.version}`);
      } catch (err: any) {
        setError(err.message);
        throw err;
      }
    },
    [document, updateContent, createVersion]
  );

  return {
    document,
    users: store.users,
    isConnected,
    isSyncing,
    error,
    updateContent,
    updateCursor,
    addComment,
    createVersion,
    restoreVersion,
  };
}

/**
 * Hook for real-time users in a document
 */
export function useRealtimeUsers(documentId: string) {
  const store = useCollaborationStore();
  const users = store.users.filter((u) => u.status === 'online');

  return {
    users,
    totalUsers: users.length,
    getUserById: (userId: string) => users.find((u) => u.id === userId),
  };
}

/**
 * Hook for document comments
 */
export function useComments(documentId: string) {
  const store = useCollaborationStore();
  const document = store.documents.get(documentId);
  const comments = document?.comments || [];

  const unresolvedComments = comments.filter((c) => !c.resolved);
  const resolvedComments = comments.filter((c) => c.resolved);

  return {
    comments,
    unresolvedComments,
    resolvedComments,
    totalComments: comments.length,
    resolveComment: (commentId: string) => {
      store.resolveComment(documentId, commentId);
    },
  };
}

/**
 * Hook for version history
 */
export function useVersionHistory(documentId: string) {
  const store = useCollaborationStore();
  const document = store.documents.get(documentId);
  const versions = document?.versions || [];

  return {
    versions,
    currentVersion: document?.currentVersion || 0,
    totalVersions: versions.length,
    getVersionById: (versionId: string) =>
      versions.find((v) => v.id === versionId),
  };
}