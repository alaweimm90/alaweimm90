# Frontend Applications Suite

Three production-ready Next.js applications for revolutionary research and innovation platforms.

## 🚀 Applications

### 1. Ghost Researcher
**AI-Powered Research Assistant Platform**
- 📁 Location: `/frontend/ghost-researcher`
- 🎯 Purpose: Transform academic research with AI-powered literature review, hypothesis generation, and paper writing assistance
- 🔗 Port: 3000 (default)

#### Key Features:
- Advanced literature review with citation network visualization
- AI-powered hypothesis generation
- Real-time paper writing assistant
- Collaboration tools for research teams
- Project management dashboard
- Export to multiple formats (PDF, BibTeX, CSV)

### 2. Scientific Tinder
**Research Collaboration Matching Platform**
- 📁 Location: `/frontend/scientific-tinder`
- 🎯 Purpose: Connect researchers worldwide using a swipe-based matching system
- 🔗 Port: 3001

#### Key Features:
- Swipe-based researcher matching interface
- AI-powered compatibility scoring
- Real-time messaging and video calls
- Collaboration workspace
- Verified researcher profiles
- Cross-disciplinary matching algorithm

### 3. Chaos Engine
**Domain Collision Idea Generator**
- 📁 Location: `/frontend/chaos-engine`
- 🎯 Purpose: Generate breakthrough innovations by colliding different domains
- 🔗 Port: 3002

#### Key Features:
- Random domain collision generator
- AI-powered idea viability scoring
- Automatic business plan generation
- Pitch deck builder
- Community voting system
- Success story showcase

## 🛠️ Tech Stack

All applications are built with:
- **Framework:** Next.js 14+ with App Router
- **Language:** TypeScript
- **Styling:** Tailwind CSS + Custom Design System
- **State Management:** Zustand + React Context
- **Animation:** Framer Motion
- **Data Fetching:** React Query (TanStack Query)
- **Forms:** React Hook Form + Zod Validation
- **UI Components:** Radix UI Primitives
- **Icons:** Lucide React
- **Charts:** Recharts / Chart.js
- **Real-time:** Socket.io
- **Auth:** NextAuth.js

## 📦 Installation

### Prerequisites
- Node.js 18+
- npm/yarn/pnpm
- Git

### Setup All Applications

```bash
# Clone the repository
git clone <repository-url>
cd Foundry/frontend

# Install dependencies for all apps
cd ghost-researcher && npm install && cd ..
cd scientific-tinder && npm install && cd ..
cd chaos-engine && npm install && cd ..
```

### Individual App Setup

#### Ghost Researcher
```bash
cd frontend/ghost-researcher
npm install
npm run dev
# Open http://localhost:3000
```

#### Scientific Tinder
```bash
cd frontend/scientific-tinder
npm install
npm run dev
# Open http://localhost:3001
```

#### Chaos Engine
```bash
cd frontend/chaos-engine
npm install
npm run dev
# Open http://localhost:3002
```

## 🏗️ Project Structure

Each application follows this structure:
```
app-name/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   ├── globals.css        # Global styles
│   └── [route]/           # Route folders
├── src/
│   ├── components/        # React components
│   │   ├── ui/           # Base UI components
│   │   ├── layouts/      # Layout components
│   │   └── [feature]/    # Feature components
│   ├── hooks/            # Custom React hooks
│   ├── lib/              # Utilities and helpers
│   ├── types/            # TypeScript types
│   ├── context/          # React context providers
│   └── styles/           # Additional styles
├── public/               # Static assets
├── tests/                # Test files
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── next.config.js
└── postcss.config.js
```

## 🎨 Design System

All applications share a unified design system with:
- **Color Palette:** Consistent primary, secondary, and accent colors
- **Typography:** Inter font with responsive sizing
- **Spacing:** 4px grid system
- **Components:** 30+ reusable components
- **Dark Mode:** Full dark mode support with system preference detection
- **Animations:** Smooth micro-interactions using Framer Motion
- **Responsive:** Mobile-first design with breakpoints at 640px, 768px, 1024px, 1280px

## 🔐 Authentication

Each app includes:
- JWT-based authentication
- Session management
- Protected routes
- Role-based access control
- OAuth integration ready

## 📊 Features Overview

### Common Features (All Apps):
- ✅ Modern, polished UI with glass morphism effects
- ✅ Dark/Light mode with system preference
- ✅ Responsive design (mobile-first)
- ✅ Loading states and skeleton screens
- ✅ Error boundaries and graceful error handling
- ✅ Toast notifications
- ✅ Accessibility (WCAG 2.1 AA compliant)
- ✅ PWA support
- ✅ SEO optimized
- ✅ Analytics integration ready

### Performance Optimizations:
- Code splitting by route
- Image optimization with next/image
- Lazy loading components
- Bundle size optimization
- Lighthouse score: 90+

## 🧪 Testing

```bash
# Run unit tests
npm test

# Run tests in watch mode
npm run test:watch

# Run E2E tests
npm run test:e2e

# Type checking
npm run type-check
```

## 📝 Environment Variables

Create `.env.local` in each app directory:

```env
# Authentication
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key

# Database
DATABASE_URL=postgresql://...

# API
NEXT_PUBLIC_API_URL=http://localhost:8000

# WebSocket
NEXT_PUBLIC_WS_URL=http://localhost:3001

# Analytics (optional)
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
```

## 🚀 Deployment

### Production Build
```bash
npm run build
npm start
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Vercel Deployment
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

## 📚 API Integration

All apps are configured to work with backend APIs:
- RESTful API endpoints defined
- GraphQL ready (if needed)
- WebSocket connections for real-time features
- Error handling and retry logic
- Request/response interceptors

## 🔧 Development Commands

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start

# Lint code
npm run lint

# Format code
npm run format

# Type check
npm run type-check
```

## 🎯 Key Pages Implemented

### Ghost Researcher:
- Landing page with hero, features, pricing
- Dashboard with stats and activity
- Research input interface
- Results visualization page
- Hypothesis generator
- Paper writing assistant
- Project management
- Settings & profile

### Scientific Tinder:
- Landing page with animations
- Swipe interface
- Matches grid
- Researcher profiles
- Messaging/chat interface
- Collaboration workspace
- Settings & preferences

### Chaos Engine:
- Landing page with live demo
- Idea generator interface
- Results dashboard
- Business plan generator
- Pitch deck builder
- Community showcase
- Idea comparison tool
- Settings & profile

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

For issues or questions:
- Open an issue on GitHub
- Contact: support@Foundry.com
- Documentation: docs.Foundry.com

---

**Built with ❤️ for researchers and innovators worldwide**