# API Reference Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Ghost Researcher API](#ghost-researcher-api)
4. [Scientific Tinder API](#scientific-tinder-api)
5. [Chaos Engine API](#chaos-engine-api)
6. [WebSocket Events](#websocket-events)
7. [Error Codes](#error-codes)
8. [Rate Limiting](#rate-limiting)

---

## Overview

Base URL: `https://api.Foundry.com/v1`

All API endpoints return JSON responses with the following structure:

```typescript
{
  "success": boolean,
  "data": any, // Response payload
  "error": string | null,
  "code": string | null,
  "timestamp": string
}
```

### Pagination

Paginated endpoints accept these query parameters:
- `page` (number): Page number (default: 1)
- `pageSize` (number): Items per page (default: 20, max: 100)
- `sort` (string): Sort field
- `sortOrder` (string): 'asc' or 'desc'

Paginated responses include:
```json
{
  "data": [],
  "pagination": {
    "page": 1,
    "pageSize": 20,
    "total": 100,
    "totalPages": 5,
    "hasMore": true
  }
}
```

---

## Authentication

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "remember": true
}

Response:
{
  "token": "jwt_token",
  "refreshToken": "refresh_token",
  "user": { /* UserProfile */ },
  "expiresIn": 900
}
```

### Signup
```http
POST /auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "name": "John Doe",
  "field": "Computer Science",
  "institution": "MIT"
}
```

### Refresh Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refreshToken": "refresh_token"
}

Response:
{
  "accessToken": "new_jwt_token",
  "refreshToken": "new_refresh_token",
  "expiresIn": 900
}
```

### Headers
All authenticated requests must include:
```http
Authorization: Bearer <jwt_token>
```

---

## Ghost Researcher API

### Projects

#### List Projects
```http
GET /research/projects?page=1&pageSize=20&status=active&visibility=private
Authorization: Bearer <token>

Response: PaginatedResponse<ResearchProject>
```

#### Get Project
```http
GET /research/projects/:projectId
Authorization: Bearer <token>

Response: ResearchProject
```

#### Create Project
```http
POST /research/projects
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "AI Research Project",
  "description": "Exploring neural networks",
  "visibility": "private",
  "tags": ["AI", "ML"]
}
```

#### Update Project
```http
PUT /research/projects/:projectId
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Updated Title",
  "status": "completed"
}
```

#### Delete Project
```http
DELETE /research/projects/:projectId
Authorization: Bearer <token>
```

### Paper Search & Analysis

#### Search Papers
```http
POST /research/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "machine learning",
  "filters": {
    "yearRange": { "start": 2020, "end": 2024 },
    "keywords": ["AI", "neural networks"]
  },
  "pagination": { "page": 1, "pageSize": 20 }
}

Response: PaginatedResponse<ResearchPaper>
```

#### Analyze Paper
```http
POST /research/analyze
Authorization: Bearer <token>
Content-Type: multipart/form-data

FormData:
- file: PDF file
- url: Paper URL
- doi: DOI identifier
- extractReferences: true
- generateSummary: true

Response: ResearchPaper with analysis
```

#### Get Paper Details
```http
GET /research/papers/:paperId
Authorization: Bearer <token>

Response: ResearchPaper
```

### Hypothesis Generation

#### Get Hypotheses
```http
GET /research/hypotheses?projectId=xxx&page=1&pageSize=20
Authorization: Bearer <token>

Response: PaginatedResponse<Hypothesis>
```

#### Generate Hypothesis
```http
POST /research/hypotheses
Authorization: Bearer <token>
Content-Type: application/json

{
  "projectId": "project_id",
  "papers": ["paper1_id", "paper2_id"],
  "context": "Research context",
  "methodology": "Experimental approach"
}

Response: Hypothesis[]
```

### Collaboration

#### Get Collaborators
```http
GET /collaboration/team/:projectId
Authorization: Bearer <token>

Response: Collaborator[]
```

#### Invite Collaborator
```http
POST /collaboration/invite
Authorization: Bearer <token>
Content-Type: application/json

{
  "projectId": "project_id",
  "email": "colleague@example.com",
  "role": "editor"
}
```

#### Share Research
```http
POST /collaboration/share
Authorization: Bearer <token>
Content-Type: application/json

{
  "projectId": "project_id",
  "visibility": "public",
  "message": "Check out my research!",
  "recipients": ["email1@example.com"]
}

Response: { "shareUrl": "https://..." }
```

### AI Enhancement

#### Get Writing Suggestions
```http
POST /ai/enhance
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "Original text...",
  "documentType": "abstract",
  "field": "Computer Science",
  "targetJournal": "Nature"
}

Response: {
  "suggestions": ["suggestion1", "suggestion2"],
  "improved": "Enhanced text..."
}
```

---

## Scientific Tinder API

### Profile Management

#### Get My Profile
```http
GET /profiles/me
Authorization: Bearer <token>

Response: ScientificProfile
```

#### Update Profile
```http
PUT /profiles/me
Authorization: Bearer <token>
Content-Type: application/json

{
  "bio": "Updated bio",
  "researchInterests": ["AI", "Quantum Computing"],
  "availability": "available"
}
```

#### Upload Photo
```http
POST /profiles/me/photos
Authorization: Bearer <token>
Content-Type: multipart/form-data

FormData:
- photo: image file
- isPrimary: true

Response: { "url": "https://...", "id": "photo_id" }
```

### Discovery & Matching

#### Discover Profiles
```http
GET /matches/discover?field=physics&verified=true&minHIndex=10
Authorization: Bearer <token>

Response: ScientificProfile[]
```

#### Swipe Action
```http
POST /matches/swipe
Authorization: Bearer <token>
Content-Type: application/json

{
  "targetUserId": "user_id",
  "action": "like",
  "message": "Love your research!"
}

Response: {
  "match": true,
  "matchId": "match_id"
}
```

#### Get Mutual Matches
```http
GET /matches/mutual?page=1&pageSize=20&status=matched
Authorization: Bearer <token>

Response: PaginatedResponse<Match>
```

### Messaging

#### Get Conversations
```http
GET /messages?page=1&pageSize=20
Authorization: Bearer <token>

Response: PaginatedResponse<Conversation>
```

#### Send Message
```http
POST /messages
Authorization: Bearer <token>
Content-Type: application/json

{
  "conversationId": "conversation_id",
  "content": "Hello! Great research!"
}

Response: Message
```

#### Mark as Read
```http
POST /messages/:conversationId/read
Authorization: Bearer <token>
Content-Type: application/json

{
  "messageIds": ["msg1", "msg2"]
}
```

### Teams & Collaboration

#### Create Team
```http
POST /teams
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Quantum Research Team",
  "description": "Exploring quantum computing",
  "field": "Physics",
  "project": "Quantum supremacy research"
}

Response: CollaborationTeam
```

#### Start Collaboration
```http
POST /collaboration/start
Authorization: Bearer <token>
Content-Type: application/json

{
  "matchId": "match_id",
  "title": "Joint Research Proposal",
  "description": "Collaboration details",
  "goals": ["Goal 1", "Goal 2"],
  "timeline": "6 months"
}

Response: { "collaborationId": "collab_id" }
```

### Video & Meetings

#### Create Video Meeting
```http
POST /meetings/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "participants": ["user1_id", "user2_id"],
  "scheduledAt": "2024-01-15T10:00:00Z"
}

Response: {
  "roomId": "room_id",
  "joinUrl": "https://meet...",
  "token": "video_token"
}
```

---

## Chaos Engine API

### Idea Generation

#### Generate Idea
```http
POST /ideas/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "domain1": "Healthcare",
  "domain2": "Blockchain",
  "technique": "SCAMPER",
  "constraints": ["low-cost", "scalable"],
  "targetMarket": "Hospitals"
}

Response: IdeaCollision
```

#### Get My Ideas
```http
GET /ideas?page=1&pageSize=20&status=public&tag=AI
Authorization: Bearer <token>

Response: PaginatedResponse<IdeaCollision>
```

#### Update Idea
```http
PUT /ideas/:ideaId
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Updated Title",
  "description": "Updated description",
  "status": "in-development"
}
```

### Refinement

#### Get Refinement Suggestions
```http
POST /ideas/refine
Authorization: Bearer <token>
Content-Type: application/json

{
  "ideaId": "idea_id",
  "refinementType": "enhancement",
  "context": "Target enterprise market"
}

Response: Refinement[]
```

#### Apply Refinement
```http
POST /ideas/:ideaId/refinements/:refinementId/apply
Authorization: Bearer <token>

Response: IdeaCollision
```

### Scoring & Analysis

#### Score Idea
```http
POST /ideas/score
Authorization: Bearer <token>
Content-Type: application/json

{
  "ideaId": "idea_id",
  "detailed": true,
  "includeMarketAnalysis": true
}

Response: {
  "score": { /* IdeaScore */ },
  "analysis": { /* SWOT */ },
  "recommendations": [],
  "competitors": []
}
```

#### Market Viability Analysis
```http
GET /ideas/:ideaId/market-analysis
Authorization: Bearer <token>

Response: {
  "marketSize": 1000000000,
  "growthRate": 15.5,
  "targetAudience": { /* Demographics */ },
  "competitiveLandscape": { /* Analysis */ },
  "viabilityScore": 85
}
```

### Business Plan

#### Generate Business Plan
```http
POST /ideas/:ideaId/business-plan
Authorization: Bearer <token>
Content-Type: application/json

{
  "includeFinancials": true,
  "detailLevel": "comprehensive",
  "targetInvestors": true
}

Response: BusinessPlan
```

#### Generate Pitch Deck
```http
POST /ideas/:ideaId/pitch-deck
Authorization: Bearer <token>
Content-Type: application/json

{
  "template": "investor"
}

Response: { "downloadUrl": "https://..." }
```

### Community Features

#### Get Leaderboard
```http
GET /community/leaderboard?period=week&page=1&pageSize=20
Authorization: Bearer <token>

Response: PaginatedResponse<CommunityIdea>
```

#### Vote on Idea
```http
POST /community/ideas/:ideaId/vote
Authorization: Bearer <token>
Content-Type: application/json

{
  "voteType": "up"
}

Response: {
  "newCount": { "up": 42, "down": 5 },
  "userVote": "up"
}
```

#### Fork Idea
```http
POST /ideas/:ideaId/fork
Authorization: Bearer <token>
Content-Type: application/json

{
  "changes": "Added AI integration"
}

Response: { "forkedIdeaId": "new_idea_id" }
```

---

## WebSocket Events

### Connection
```javascript
const socket = io('wss://api.Foundry.com', {
  auth: { token: 'jwt_token' },
  transports: ['websocket']
});
```

### Ghost Researcher Events

#### Subscribe to Project Updates
```javascript
// Join project room
socket.emit('join', { room: 'project_123' });

// Listen for updates
socket.on('research:update', (data) => {
  // Handle document update
});

socket.on('research:cursor', (data) => {
  // Handle cursor position
});

socket.on('research:comment', (data) => {
  // Handle new comment
});
```

### Scientific Tinder Events

#### Subscribe to Matches & Messages
```javascript
socket.on('match:new', (data) => {
  // Handle new match
});

socket.on('message:new', (data) => {
  // Handle new message
});

socket.on('message:typing', (data) => {
  // Handle typing indicator
});
```

### Chaos Engine Events

#### Subscribe to Idea Updates
```javascript
socket.on('idea:generated', (data) => {
  // Handle new idea
});

socket.on('idea:vote', (data) => {
  // Handle vote update
});

socket.on('collaboration:update', (data) => {
  // Handle collaborative edit
});
```

---

## Error Codes

| Code | Description | HTTP Status |
|------|------------|------------|
| `AUTH_REQUIRED` | Authentication required | 401 |
| `INVALID_CREDENTIALS` | Invalid email or password | 401 |
| `TOKEN_EXPIRED` | JWT token expired | 401 |
| `INSUFFICIENT_PERMISSIONS` | Lacking required permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `VALIDATION_ERROR` | Request validation failed | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily down | 503 |

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  },
  "timestamp": "2024-01-15T10:00:00Z"
}
```

---

## Rate Limiting

Rate limits are enforced per API key:

| Endpoint Type | Requests | Window |
|--------------|----------|--------|
| Authentication | 10 | 15 min |
| Search | 100 | 1 min |
| Write Operations | 50 | 1 min |
| AI Operations | 20 | 1 min |
| Export | 10 | 1 hour |

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

### Handling Rate Limits
When rate limited, the API returns:
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "details": {
      "retryAfter": 60
    }
  }
}
```

---

## Best Practices

1. **Always handle errors gracefully** - Check for error responses and implement retry logic
2. **Use pagination** - Don't request all data at once
3. **Cache responses** - Implement client-side caching to reduce API calls
4. **Use WebSockets for real-time data** - Don't poll endpoints repeatedly
5. **Compress large payloads** - Use gzip compression for large requests
6. **Implement exponential backoff** - When retrying failed requests
7. **Store tokens securely** - Use HttpOnly cookies or secure storage
8. **Monitor rate limits** - Check rate limit headers and adjust request frequency

---

## SDK Examples

### JavaScript/TypeScript
```typescript
import { ghostResearcherApi, scientificTinderApi, chaosEngineApi } from '@Foundry/sdk';

// Initialize with API key
const api = ghostResearcherApi.init({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.Foundry.com/v1'
});

// Search papers
const papers = await api.searchPapers({
  query: 'quantum computing',
  filters: {
    yearRange: { start: 2020, end: 2024 }
  }
});

// Generate idea
const idea = await chaosEngineApi.generateIdea({
  domain1: 'Healthcare',
  domain2: 'AI',
  technique: 'SCAMPER'
});
```

### Python
```python
from Foundry import GhostResearcher, ScientificTinder, ChaosEngine

# Initialize clients
ghost = GhostResearcher(api_key="your-api-key")
tinder = ScientificTinder(api_key="your-api-key")
chaos = ChaosEngine(api_key="your-api-key")

# Search papers
papers = ghost.search_papers(
    query="machine learning",
    year_range=(2020, 2024)
)

# Discover profiles
profiles = tinder.discover_profiles(
    field="Computer Science",
    verified=True
)

# Generate idea
idea = chaos.generate_idea(
    domain1="Finance",
    domain2="Blockchain",
    technique="Biomimicry"
)
```

---

## Support

For API support, please contact:
- Email: api-support@Foundry.com
- Documentation: https://docs.Foundry.com
- Status Page: https://status.Foundry.com
- Developer Forum: https://forum.Foundry.com/developers