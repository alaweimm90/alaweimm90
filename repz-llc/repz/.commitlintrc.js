/**
 * REPZ Conventional Commits Configuration
 * Enforces consistent commit message format across the entire platform
 * Following Angular's commit message conventions with REPZ-specific types
 */

module.exports = {
  extends: ['@commitlint/config-conventional'],
  
  // Custom rules for REPZ platform
  rules: {
    // Type enum - allowed commit types
    'type-enum': [2, 'always', [
      // Standard types
      'feat',     // New feature
      'fix',      // Bug fix
      'docs',     // Documentation changes
      'style',    // Code style changes (formatting, etc.)
      'refactor', // Code refactoring
      'test',     // Adding or updating tests
      'chore',    // Maintenance tasks
      'perf',     // Performance improvements
      'ci',       // CI/CD changes
      'build',    // Build system changes
      'revert',   // Reverting changes
      
      // REPZ-specific types
      'design',   // Design system changes
      'content',  // Content updates (copy, images, etc.)
      'config',   // Configuration changes
      'security', // Security fixes
      'deps',     // Dependency updates
      'release',  // Release commits
      'hotfix',   // Critical fixes for production
      'wip',      // Work in progress (use sparingly)
      
      // Platform-specific types
      'auth',     // Authentication related
      'payment',  // Payment/Stripe related
      'coach',    // Coach-specific features
      'client',   // Client-specific features
      'admin',    // Admin panel features
      'mobile',   // Mobile app specific
      'ai',       // AI/ML features
      'analytics' // Analytics and tracking
    ]],
    
    // Scope validation - optional but when used, must match these patterns
    'scope-enum': [1, 'always', [
      // Core areas
      'app',
      'ui',
      'components',
      'pages',
      'hooks',
      'contexts',
      'utils',
      'lib',
      'types',
      'constants',
      
      // Feature areas
      'auth',
      'dashboard',
      'intake',
      'pricing',
      'analytics',
      'testing',
      'mobile',
      'ai',
      
      // Technical areas
      'api',
      'db',
      'build',
      'deploy',
      'config',
      'deps',
      'security',
      
      // Design system
      'tokens',
      'atoms',
      'molecules',
      'organisms',
      'templates',
      'storybook',
      
      // Business logic
      'tiers',
      'subscriptions',
      'coaching',
      'nutrition',
      'workouts',
      'progress',
      'goals',
      
      // Third-party integrations
      'supabase',
      'stripe',
      'openai',
      'capacitor',
      'sentry'
    ]],
    
    // Message format rules
    'type-case': [2, 'always', 'lower-case'],
    'type-empty': [2, 'never'],
    'scope-case': [2, 'always', 'lower-case'],
    'subject-case': [2, 'always', 'sentence-case'],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'subject-max-length': [2, 'always', 100],
    'subject-min-length': [2, 'always', 10],
    
    // Body and footer rules
    'body-leading-blank': [1, 'always'],
    'body-max-line-length': [2, 'always', 100],
    'footer-leading-blank': [1, 'always'],
    'footer-max-line-length': [2, 'always', 100],
    
    // Header rules
    'header-max-length': [2, 'always', 100],
    'header-min-length': [2, 'always', 15]
  },
  
  // Custom prompt configuration for interactive commits
  prompt: {
    questions: {
      type: {
        description: "Select the type of change that you're committing:",
        enum: {
          feat: {
            description: 'A new feature',
            title: 'Features',
            emoji: '✨'
          },
          fix: {
            description: 'A bug fix',
            title: 'Bug Fixes',
            emoji: '🐛'
          },
          docs: {
            description: 'Documentation only changes',
            title: 'Documentation',
            emoji: '📚'
          },
          style: {
            description: 'Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)',
            title: 'Styles',
            emoji: '💎'
          },
          refactor: {
            description: 'A code change that neither fixes a bug nor adds a feature',
            title: 'Code Refactoring',
            emoji: '📦'
          },
          perf: {
            description: 'A code change that improves performance',
            title: 'Performance Improvements',
            emoji: '🚀'
          },
          test: {
            description: 'Adding missing tests or correcting existing tests',
            title: 'Tests',
            emoji: '🚨'
          },
          build: {
            description: 'Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)',
            title: 'Builds',
            emoji: '🛠'
          },
          ci: {
            description: 'Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)',
            title: 'Continuous Integrations',
            emoji: '⚙️'
          },
          chore: {
            description: "Other changes that don't modify src or test files",
            title: 'Chores',
            emoji: '♻️'
          },
          revert: {
            description: 'Reverts a previous commit',
            title: 'Reverts',
            emoji: '🗑'
          },
          design: {
            description: 'Design system and UI component changes',
            title: 'Design System',
            emoji: '🎨'
          },
          security: {
            description: 'Security improvements and fixes',
            title: 'Security',
            emoji: '🔒'
          },
          config: {
            description: 'Configuration file changes',
            title: 'Configuration',
            emoji: '🔧'
          }
        }
      },
      scope: {
        description:
          'What is the scope of this change (e.g. component, page, service)? (optional)'
      },
      subject: {
        description:
          'Write a short, imperative tense description of the change (max 100 chars):\n'
      },
      body: {
        description:
          'Provide a longer description of the change (optional). Use "|" to break new line:\n'
      },
      isBreaking: {
        description: 'Are there any breaking changes?',
        default: false
      },
      breakingBody: {
        description:
          'A BREAKING CHANGE commit requires a body. Please enter a longer description of the commit itself:\n'
      },
      breaking: {
        description: 'Describe the breaking changes:\n'
      },
      isIssueAffected: {
        description: 'Does this change affect any open issues?',
        default: false
      },
      issuesBody: {
        description:
          'If issues are closed, the commit requires a body. Please enter a longer description of the commit itself:\n'
      },
      issues: {
        description: 'Add issue references (e.g. "fix #123", "re #123"):\n'
      }
    }
  },
  
  // Ignore certain commits (like merge commits)
  ignores: [
    (commit) => commit.includes('WIP'),
    (commit) => commit.includes('wip'),
    (commit) => commit.startsWith('Merge'),
    (commit) => commit.startsWith('merge')
  ],
  
  // Default commit type if none specified
  defaultIgnores: true,
  
  // Help URL for commit message format
  helpUrl: 'https://github.com/conventional-changelog/commitlint/#what-is-commitlint'
};