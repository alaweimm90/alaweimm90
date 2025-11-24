# MCP (Model Context Protocol) Servers Guide

## Overview

Model Context Protocol (MCP) was introduced by Anthropic in late 2024 as an open standard that gives Large Language Models secure, controlled access to tools and data sources. This guide documents the top MCPs organized by purpose and category.

---

## Directory & Discovery Resources

- **[mcp.so](https://mcp.so/)** - Community-driven MCP server directory (17,000+ servers)
- **[MCP Market](https://mcpmarket.com/)** - Discover MCPs with filter options
- **[GitHub: awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)** - Curated collection of production-ready and experimental servers
- **[GitHub: Official MCP Servers](https://github.com/modelcontextprotocol/servers)** - Official reference implementations
- **[Claude MCP Servers](https://www.claudemcp.com/servers)** - Community-curated Claude-specific MCPs

---

## Top 50 MCP Servers

### Reference & Core Servers
1. **Everything** - Comprehensive reference server with prompts, resources, and tools
2. **Fetch** - Web content retrieval and conversion optimized for LLM usage
3. **Filesystem** - Secure file operations with configurable access controls
4. **Git** - Tools for reading, searching, and manipulating Git repositories
5. **Memory** - Knowledge graph-based persistent memory system
6. **Sequential Thinking** - Dynamic problem-solving through reflective sequences
7. **Time** - Time and timezone conversion capabilities

### Enterprise & Data Integration (Top Tier)
8. **K2view** - Enterprise data access with patented Micro-Database technology
9. **AWS** - Specialized AWS services integration
10. **Azure** - Access to Azure services (Storage, Cosmos DB, Azure CLI)
11. **GitHub** - GitHub REST API integration for repositories, issues, PRs, CI/CD
12. **MongoDB** - Database integration and management
13. **Cloudflare** - CDN and edge computing services
14. **JetBrains** - IDE and development tools integration
15. **Auth0** - Authentication tenant management
16. **Atlassian** - Jira and Confluence integration
17. **Box** - Intelligent content management

### Vector Search & Knowledge Retrieval
18. **Chroma** - Vector database for semantic search and embeddings
19. **Pinecone** - Vector database for AI applications
20. **Supabase** - Open-source Firebase alternative with Postgres
21. **Vectara** - Generative AI retrieval platform

### Development & Code Tools
22. **GitLab** - GitLab repository management
23. **Semgrep** - Static analysis for code vulnerabilities
24. **Sentry** - Error tracking and monitoring
25. **Context7** - Up-to-date code documentation for frameworks
26. **Brave Search** - Web search for research
27. **Puppeteer** - Browser automation and web scraping
28. **Playwright** - Advanced browser automation

### Analytics & Data Processing
29. **GreptimeDB** - Time-series database
30. **ClickHouse** - Columnar database for analytics
31. **PostgreSQL** - SQL database integration
32. **Redis** - In-memory data store
33. **Financial Datasets** - Financial market data
34. **mcp-octagon** - Specialized protocol functions

### Communication & Collaboration
35. **Slack** - Slack workspace integration
36. **Gmail** - Email integration
37. **Google Drive** - Cloud storage access
38. **Notion** - Workspace and documentation tool
39. **Airtable** - Database and CRM platform
40. **Discord** - Discord server integration

### Search & Web Integration
41. **Serper** - Web search integration
42. **Jina AI** - Web search and content extraction
43. **Firecrawl** - Web scraping and crawling
44. **Zhipu Web Search** - Multi-engine search
45. **Reddit MCP** - Community insights from Reddit
46. **EdgeOne Pages** - HTML content deployment

### Specialized Services
47. **Zapier** - Workflow automation
48. **Salesforce** - CRM integration
49. **Howtocook** - Recipe recommendation
50. **MiniMax** - Text-to-speech, image/video generation

---

## Top 10 MCPs by Purpose

### For Development & Code Management
1. **GitHub MCP** - Repository and CI/CD management
2. **GitLab MCP** - GitLab repository operations
3. **Semgrep** - Security analysis and vulnerability detection
4. **Sentry** - Error tracking and debugging
5. **Context7** - Documentation retrieval
6. **Playwright** - Web automation
7. **Git** - Direct git operations
8. **AWS** - Development infrastructure
9. **Azure** - Cloud development services
10. **JetBrains** - IDE integration

### For Data & Analytics
1. **Chroma** - Vector semantic search
2. **ClickHouse** - OLAP analytics database
3. **PostgreSQL** - SQL operations
4. **GreptimeDB** - Time-series data
5. **Pinecone** - Vector embeddings
6. **MongoDB** - NoSQL document database
7. **Redis** - Caching and data structures
8. **Financial Datasets** - Market data analysis
9. **Supabase** - SQL + Real-time database
10. **K2view** - Enterprise data platform

### For AI & Knowledge Management
1. **Memory** - Knowledge graphs and persistence
2. **Chroma** - Semantic embeddings and search
3. **Vectara** - Generative RAG platform
4. **Sequential Thinking** - Advanced problem-solving
5. **Context7** - Documentation context
6. **Fetch** - Content retrieval and conversion
7. **Jina AI** - Web content extraction
8. **Firecrawl** - Web data collection
9. **Vectara** - Search and retrieval
10. **K2view** - Context-rich data access

### For Communication & Collaboration
1. **Slack** - Team messaging
2. **Gmail** - Email management
3. **Notion** - Documentation and wikis
4. **Airtable** - Database and teams
5. **Discord** - Community interaction
6. **Google Drive** - File collaboration
7. **Atlassian (Jira/Confluence)** - Project management
8. **Box** - Enterprise content management
9. **Zapier** - Workflow automation
10. **Auth0** - Identity management

### For Security & Compliance
1. **Semgrep** - Code security scanning
2. **Sentry** - Application monitoring
3. **Cloudflare** - Security and CDN
4. **AWS** - Security services
5. **Azure** - Identity and security
6. **Auth0** - Authentication and authorization
7. **GitHub** - Repository security
8. **GitLab** - DevSecOps integration
9. **Financial Datasets** - Compliance data
10. **K2view** - Secure data access

### For Content & Web
1. **Fetch** - Web content retrieval
2. **Brave Search** - Web research
3. **Puppeteer** - Browser automation
4. **Playwright** - Advanced web interaction
5. **Firecrawl** - Web scraping
6. **Jina AI** - Content extraction
7. **Serper** - Search integration
8. **Zhipu Web Search** - Multi-engine search
9. **Reddit MCP** - Community content
10. **EdgeOne Pages** - Content deployment

### For Business & Operations
1. **Airtable** - CRM and databases
2. **Salesforce** - Customer relationship
3. **Notion** - Knowledge management
4. **Slack** - Team operations
5. **Zapier** - Process automation
6. **Google Drive** - Document collaboration
7. **Gmail** - Communication
8. **Box** - Content management
9. **Atlassian** - Project tracking
10. **Time** - Schedule coordination

### For Infrastructure & DevOps
1. **AWS** - Cloud infrastructure
2. **Azure** - Cloud services
3. **Kubernetes** - Container orchestration
4. **Docker** - Container management
5. **Cloudflare** - Edge computing
6. **Sentry** - Performance monitoring
7. **GitHub** - CI/CD workflows
8. **GitLab** - DevOps pipelines
9. **PostgreSQL** - Database infrastructure
10. **Redis** - Caching layer

### For ML & AI Operations
1. **Chroma** - Vector databases
2. **Pinecone** - Vector embeddings
3. **Sequential Thinking** - Reasoning systems
4. **Memory** - Knowledge persistence
5. **Vectara** - RAG platforms
6. **GreptimeDB** - Time-series ML data
7. **Firecrawl** - Data collection
8. **Jina AI** - Content understanding
9. **MiniMax** - Multimodal generation
10. **K2view** - Context data for AI

### For Content Creation & Media
1. **MiniMax** - Text-to-speech and generation
2. **Puppeteer** - Web content capture
3. **Playwright** - Browser automation
4. **Fetch** - Content processing
5. **Firecrawl** - Content discovery
6. **Google Drive** - Media storage
7. **Box** - Asset management
8. **Airtable** - Content databases
9. **Notion** - Documentation creation
10. **EdgeOne Pages** - Publishing platform

---

## MCP Server Categories (35+ Categories)

1. **Aggregators** - Multi-MCP management
2. **Aerospace & Astrodynamics** - Space systems
3. **Art & Culture** - Museums and entertainment
4. **Architecture & Design** - Design tools
5. **Biology & Medicine** - Healthcare data
6. **Browser Automation** - Web interaction
7. **Cloud Platforms** - AWS, Azure, GCP
8. **Code Execution** - Sandbox environments
9. **Coding Agents** - Autonomous coding
10. **Command Line** - Shell execution
11. **Communication** - Messaging tools
12. **Customer Data Platforms** - CRM systems
13. **Databases** - SQL, NoSQL stores
14. **Data Platforms** - Analytics engines
15. **Developer Tools** - IDEs and debugging
16. **Data Science** - ML frameworks
17. **Embedded Systems** - IoT devices
18. **File Systems** - Storage access
19. **Finance & Fintech** - Banking tools
20. **Gaming** - Game development
21. **Knowledge & Memory** - RAG systems
22. **Location Services** - Maps and geo
23. **Marketing** - Campaign tools
24. **Monitoring** - System observability
25. **Multimedia Processing** - Audio/video
26. **Search & Extraction** - Web search
27. **Security** - Vulnerability scanning
28. **Social Media** - Platform APIs
29. **Sports** - Sports data
30. **Support & Service** - Help desk systems
31. **Translation** - Language services
32. **Text-to-Speech** - Voice synthesis
33. **Travel & Transport** - Booking systems
34. **Version Control** - Git systems
35. **Workplace & Productivity** - Office tools
36. **Other Tools** - Miscellaneous utilities

---

## Getting Started

### Installation Steps

1. **Install MCP globally:**
   ```bash
   npm install -g @modelcontextprotocol/sdk
   ```

2. **Configure Claude Code settings.json:**
   ```json
   {
     "mcpServers": {
       "filesystem": {
         "command": "npx",
         "args": ["@modelcontextprotocol/server-filesystem"]
       },
       "git": {
         "command": "npx",
         "args": ["@modelcontextprotocol/server-git"]
       }
     }
   }
   ```

3. **Add specific MCPs as needed** - See directory resources above

---

## Resources

- Official Anthropic MCP: https://www.anthropic.com/news/model-context-protocol
- GitHub Repository: https://github.com/modelcontextprotocol/servers
- Community Awesome List: https://github.com/punkpeye/awesome-mcp-servers
- MCP Directory: https://mcp.so/
- Market: https://mcpmarket.com/

---

## Notes

- MCPs enable secure, sandboxed access to external tools and data
- Each MCP runs in its own process for security
- Official servers maintained by Anthropic set best practices
- Community servers vary in maturity - check documentation
- Most MCPs are open-source and available on GitHub or npm

