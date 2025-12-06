// Base API Types
export type ApiResponse<T = any> = {
  success: boolean
  data?: T
  error?: string
  code?: string
  timestamp?: string
}

export type PaginatedResponse<T = any> = {
  data: T[]
  pagination: {
    page: number
    pageSize: number
    total: number
    totalPages: number
    hasMore: boolean
  }
}

export type ErrorResponse = {
  error: {
    code: string
    message: string
    details?: Record<string, any>
    stack?: string
  }
  timestamp: string
}

export type SortOrder = 'asc' | 'desc'

export type PaginationParams = {
  page?: number
  pageSize?: number
  sort?: string
  sortOrder?: SortOrder
}

// Authentication Types
export type LoginRequest = {
  email: string
  password: string
  remember?: boolean
}

export type SignupRequest = {
  email: string
  password: string
  name: string
  field?: string
  institution?: string
}

export type LoginResponse = {
  token: string
  refreshToken: string
  user: UserProfile
  expiresIn: number
}

export type RefreshTokenRequest = {
  refreshToken: string
}

export type RefreshTokenResponse = {
  accessToken: string
  refreshToken?: string
  expiresIn: number
}

export type UserProfile = {
  id: string
  email: string
  name: string
  avatar?: string
  role: 'user' | 'admin' | 'moderator'
  field?: string
  institution?: string
  bio?: string
  createdAt: string
  updatedAt: string
  preferences?: UserPreferences
}

export type UserPreferences = {
  theme: 'light' | 'dark' | 'system'
  notifications: {
    email: boolean
    push: boolean
    sms: boolean
  }
  privacy: {
    profileVisible: boolean
    showEmail: boolean
    showInstitution: boolean
  }
}

// Ghost Researcher Types
export type ResearchProject = {
  id: string
  title: string
  description: string
  ownerId: string
  collaborators: Collaborator[]
  papers: ResearchPaper[]
  hypotheses: Hypothesis[]
  status: 'active' | 'completed' | 'archived'
  visibility: 'private' | 'team' | 'public'
  tags: string[]
  createdAt: string
  updatedAt: string
}

export type ResearchPaper = {
  id: string
  title: string
  authors: string[]
  abstract: string
  doi?: string
  arxivId?: string
  pmid?: string
  url?: string
  publicationDate: string
  journal?: string
  citations: number
  keywords: string[]
  fullText?: string
  pdfUrl?: string
  projectId?: string
  createdAt: string
  updatedAt: string
}

export type Hypothesis = {
  id: string
  title: string
  description: string
  evidence: Evidence[]
  confidence: number
  projectId: string
  status: 'draft' | 'testing' | 'validated' | 'rejected'
  relatedPapers: string[]
  createdAt: string
  updatedAt: string
}

export type Evidence = {
  id: string
  source: string
  text: string
  paperId?: string
  type: 'supporting' | 'contradicting' | 'neutral'
  confidence: number
}

export type Collaborator = {
  id: string
  userId: string
  name: string
  email: string
  role: 'owner' | 'editor' | 'viewer'
  joinedAt: string
}

export type SearchPapersRequest = {
  query: string
  filters?: {
    authors?: string[]
    yearRange?: {
      start: number
      end: number
    }
    journals?: string[]
    keywords?: string[]
  }
  pagination?: PaginationParams
}

export type AnalyzePaperRequest = {
  url?: string
  doi?: string
  file?: File
  extractReferences?: boolean
  generateSummary?: boolean
}

export type GenerateHypothesisRequest = {
  projectId: string
  papers: string[]
  context?: string
  methodology?: string
}

export type ExportResearchRequest = {
  projectId: string
  format: 'pdf' | 'docx' | 'latex' | 'bibtex'
  includeHypotheses?: boolean
  includePapers?: boolean
  includeComments?: boolean
}

// Scientific Tinder Types
export type ScientificProfile = {
  id: string
  userId: string
  name: string
  title: string
  institution: string
  field: string
  specializations: string[]
  bio: string
  researchInterests: string[]
  publications: number
  hIndex?: number
  lookingFor: ('collaboration' | 'mentorship' | 'funding' | 'peer-review')[]
  skills: string[]
  languages: string[]
  location?: {
    city: string
    country: string
  }
  photos: ProfilePhoto[]
  availability: 'available' | 'busy' | 'open-to-chat'
  verified: boolean
  createdAt: string
  updatedAt: string
}

export type ProfilePhoto = {
  id: string
  url: string
  isPrimary: boolean
  order: number
}

export type Match = {
  id: string
  users: [string, string]
  status: 'pending' | 'matched' | 'declined' | 'expired'
  matchedAt?: string
  lastActivity?: string
  conversation?: Conversation
}

export type SwipeAction = {
  targetUserId: string
  action: 'like' | 'pass' | 'superlike'
  message?: string
}

export type Conversation = {
  id: string
  participants: string[]
  messages: Message[]
  createdAt: string
  lastMessageAt: string
  unreadCount: number
}

export type Message = {
  id: string
  conversationId: string
  senderId: string
  content: string
  type: 'text' | 'image' | 'file' | 'system'
  attachments?: Attachment[]
  readBy: string[]
  createdAt: string
  editedAt?: string
}

export type Attachment = {
  id: string
  filename: string
  url: string
  size: number
  mimeType: string
}

export type CollaborationTeam = {
  id: string
  name: string
  description: string
  members: TeamMember[]
  project?: string
  field: string
  status: 'forming' | 'active' | 'completed'
  createdAt: string
  updatedAt: string
}

export type TeamMember = {
  userId: string
  role: 'lead' | 'member' | 'advisor'
  joinedAt: string
  contribution?: string
}

// Chaos Engine Types
export type IdeaCollision = {
  id: string
  title: string
  description: string
  domain1: string
  domain2: string
  technique: CollisionTechnique
  userId: string
  refinements: Refinement[]
  score: IdeaScore
  businessPlan?: BusinessPlan
  status: 'draft' | 'public' | 'in-development' | 'implemented'
  visibility: 'private' | 'public'
  forks: number
  votes: {
    up: number
    down: number
  }
  tags: string[]
  createdAt: string
  updatedAt: string
}

export type CollisionTechnique =
  | 'SCAMPER'
  | 'Biomimicry'
  | 'Opposite Thinking'
  | 'Random Word'
  | 'Six Thinking Hats'
  | 'Morphological Analysis'
  | 'TRIZ'
  | 'Synectics'

export type Refinement = {
  id: string
  type: 'pivot' | 'enhancement' | 'simplification' | 'expansion'
  description: string
  impact: 'low' | 'medium' | 'high'
  implementationDifficulty: 'easy' | 'moderate' | 'hard'
  createdAt: string
}

export type IdeaScore = {
  novelty: number
  feasibility: number
  impact: number
  marketPotential: number
  overall: number
  breakdown: {
    technicalFeasibility: number
    marketSize: number
    competitiveAdvantage: number
    implementationCost: number
    timeToMarket: number
  }
}

export type BusinessPlan = {
  id: string
  ideaId: string
  executiveSummary: string
  marketAnalysis: string
  competitiveAnalysis: string
  businessModel: string
  implementationPlan: string
  financialProjections: {
    startupCost: number
    yearlyRevenue: number[]
    breakEvenMonth: number
    roi: number
  }
  risks: Risk[]
  milestones: Milestone[]
  generatedAt: string
}

export type Risk = {
  type: string
  description: string
  likelihood: 'low' | 'medium' | 'high'
  impact: 'low' | 'medium' | 'high'
  mitigation: string
}

export type Milestone = {
  title: string
  description: string
  targetDate: string
  dependencies: string[]
  estimatedCost: number
}

export type GenerateIdeaRequest = {
  domain1?: string
  domain2?: string
  technique?: CollisionTechnique
  constraints?: string[]
  targetMarket?: string
  problemToSolve?: string
}

export type RefineIdeaRequest = {
  ideaId: string
  refinementType: 'pivot' | 'enhancement' | 'simplification' | 'expansion'
  context?: string
}

export type ScoreIdeaRequest = {
  ideaId: string
  detailed?: boolean
  includeMarketAnalysis?: boolean
}

export type ExportIdeaRequest = {
  ideaId: string
  format: 'pdf' | 'pptx' | 'docx'
  includePlan?: boolean
  includeScore?: boolean
  includeRefinements?: boolean
}

export type CommunityIdea = {
  id: string
  ideaId: string
  author: {
    id: string
    name: string
    avatar?: string
  }
  votes: number
  comments: Comment[]
  forks: Fork[]
  featured: boolean
  trendingScore: number
  createdAt: string
}

export type Comment = {
  id: string
  authorId: string
  authorName: string
  content: string
  parentId?: string
  upvotes: number
  createdAt: string
  editedAt?: string
}

export type Fork = {
  id: string
  originalIdeaId: string
  forkedIdeaId: string
  authorId: string
  changes: string
  createdAt: string
}

// WebSocket Event Types
export type WebSocketEvent<T = any> = {
  type: string
  payload: T
  timestamp: string
  userId?: string
}

// Research WebSocket Events
export type ResearchUpdateEvent = {
  projectId: string
  documentId: string
  changes: any[]
  userId: string
}

export type CursorPositionEvent = {
  projectId: string
  documentId: string
  userId: string
  position: {
    line: number
    column: number
  }
}

export type CommentEvent = {
  projectId: string
  documentId: string
  comment: Comment
}

// Tinder WebSocket Events
export type NewMatchEvent = {
  match: Match
}

export type NewMessageEvent = {
  message: Message
}

export type TypingEvent = {
  conversationId: string
  userId: string
  isTyping: boolean
}

export type UserStatusEvent = {
  userId: string
  status: 'online' | 'offline' | 'away'
  lastSeen?: string
}

// Chaos Engine WebSocket Events
export type IdeaGeneratedEvent = {
  idea: IdeaCollision
}

export type IdeaVoteEvent = {
  ideaId: string
  userId: string
  voteType: 'up' | 'down'
  newCount: {
    up: number
    down: number
  }
}

export type CollaborationUpdateEvent = {
  ideaId: string
  userId: string
  changes: any[]
}

// Error Codes
export enum ErrorCode {
  // Authentication
  AUTH_REQUIRED = 'AUTH_REQUIRED',
  INVALID_CREDENTIALS = 'INVALID_CREDENTIALS',
  TOKEN_EXPIRED = 'TOKEN_EXPIRED',
  INSUFFICIENT_PERMISSIONS = 'INSUFFICIENT_PERMISSIONS',

  // Validation
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  INVALID_INPUT = 'INVALID_INPUT',
  MISSING_REQUIRED_FIELD = 'MISSING_REQUIRED_FIELD',

  // Resource
  NOT_FOUND = 'NOT_FOUND',
  ALREADY_EXISTS = 'ALREADY_EXISTS',
  CONFLICT = 'CONFLICT',

  // Rate Limiting
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',
  QUOTA_EXCEEDED = 'QUOTA_EXCEEDED',

  // Server
  INTERNAL_ERROR = 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE = 'SERVICE_UNAVAILABLE',
  NETWORK_ERROR = 'NETWORK_ERROR',

  // Business Logic
  INVALID_OPERATION = 'INVALID_OPERATION',
  PAYMENT_REQUIRED = 'PAYMENT_REQUIRED',
  SUBSCRIPTION_EXPIRED = 'SUBSCRIPTION_EXPIRED',
}