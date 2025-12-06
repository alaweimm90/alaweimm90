/**
 * REPZ Semantic Release Configuration
 * Automates versioning and release processes using conventional commits
 * Supports multiple release channels and comprehensive changelog generation
 */

module.exports = {
  // Git branches configuration
  branches: [
    // Production releases
    '+([0-9])?(.{+([0-9]),x}).x',
    'main',
    
    // Pre-releases
    {
      name: 'develop',
      channel: 'beta',
      prerelease: 'beta'
    },
    {
      name: 'staging',
      channel: 'rc',
      prerelease: 'rc'
    },
    
    // Feature branches for alpha releases
    {
      name: 'feat/*',
      channel: 'alpha',
      prerelease: 'alpha'
    }
  ],

  // Plugins configuration
  plugins: [
    // Analyze commits using conventional commits
    [
      '@semantic-release/commit-analyzer',
      {
        preset: 'angular',
        releaseRules: [
          // Custom release rules for REPZ-specific commit types
          { type: 'feat', release: 'minor' },
          { type: 'fix', release: 'patch' },
          { type: 'perf', release: 'patch' },
          { type: 'security', release: 'patch' },
          { type: 'hotfix', release: 'patch' },
          { type: 'revert', release: 'patch' },
          
          // REPZ-specific types
          { type: 'design', release: 'minor' },
          { type: 'auth', release: 'minor' },
          { type: 'payment', release: 'minor' },
          { type: 'coach', release: 'minor' },
          { type: 'client', release: 'minor' },
          { type: 'admin', release: 'minor' },
          { type: 'mobile', release: 'minor' },
          { type: 'ai', release: 'minor' },
          { type: 'analytics', release: 'minor' },
          
          // Patch-level changes
          { type: 'docs', release: 'patch' },
          { type: 'style', release: 'patch' },
          { type: 'refactor', release: 'patch' },
          { type: 'test', release: 'patch' },
          { type: 'build', release: 'patch' },
          { type: 'ci', release: 'patch' },
          { type: 'config', release: 'patch' },
          { type: 'deps', release: 'patch' },
          { type: 'content', release: 'patch' },
          
          // No release for chores
          { type: 'chore', release: false },
          { type: 'wip', release: false },
          
          // Breaking changes always trigger major release
          { breaking: true, release: 'major' },
          
          // Scope-based rules
          { scope: 'deps', release: 'patch' },
          { scope: 'security', release: 'patch' },
          
          // Critical fixes
          { subject: '*critical*', release: 'patch' },
          { subject: '*urgent*', release: 'patch' },
          { subject: '*hotfix*', release: 'patch' }
        ],
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES', 'BREAKING']
        }
      }
    ],

    // Generate release notes
    [
      '@semantic-release/release-notes-generator',
      {
        preset: 'angular',
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES', 'BREAKING']
        },
        writerOpts: {
          // Custom sections for REPZ changelog
          groupBy: 'type',
          commitGroupsSort: 'title',
          commitsSort: ['scope', 'subject'],
          noteGroupsSort: 'title',
          transform: (commit, context) => {
            const issues = [];

            // Map commit types to readable sections
            const typeMapping = {
              feat: '✨ Features',
              fix: '🐛 Bug Fixes',
              perf: '🚀 Performance Improvements',
              revert: '🗑 Reverts',
              docs: '📚 Documentation',
              style: '💎 Styles',
              refactor: '📦 Code Refactoring',
              test: '🚨 Tests',
              build: '🛠 Build System',
              ci: '⚙️ Continuous Integration',
              chore: '♻️ Chores',
              security: '🔒 Security',
              design: '🎨 Design System',
              config: '🔧 Configuration',
              deps: '📦 Dependencies',
              auth: '🔐 Authentication',
              payment: '💳 Payments',
              coach: '👨‍💼 Coach Features',
              client: '👤 Client Features',
              admin: '👨‍💻 Admin Features',
              mobile: '📱 Mobile',
              ai: '🤖 AI Features',
              analytics: '📊 Analytics',
              content: '📝 Content'
            };

            commit.type = typeMapping[commit.type] || commit.type;

            if (commit.scope === '*') {
              commit.scope = '';
            }

            if (typeof commit.hash === 'string') {
              commit.shortHash = commit.hash.substring(0, 7);
            }

            if (typeof commit.subject === 'string') {
              let url = context.repository
                ? `${context.host}/${context.owner}/${context.repository}`
                : context.repoUrl;
              if (url) {
                url = `${url}/issues/`;
                // Issue URLs.
                commit.subject = commit.subject.replace(/#([0-9]+)/g, (_, issue) => {
                  issues.push(issue);
                  return `[#${issue}](${url}${issue})`;
                });
              }
              if (context.host) {
                // User URLs.
                commit.subject = commit.subject.replace(/\B@([a-z0-9](?:-?[a-z0-9/]){0,38})/g, (_, username) => {
                  if (username.includes('/')) {
                    return `@${username}`;
                  }
                  return `[@${username}](${context.host}/${username})`;
                });
              }
            }

            // Remove references that already appear in the subject
            commit.references = commit.references.filter(reference => {
              if (issues.indexOf(reference.issue) === -1) {
                return true;
              }
              return false;
            });

            return commit;
          }
        }
      }
    ],

    // Update CHANGELOG.md
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md',
        changelogTitle: '# 📋 REPZ Platform Changelog\n\nAll notable changes to the REPZ Coach platform will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n## [Unreleased]'
      }
    ],

    // Update package.json version
    '@semantic-release/npm',

    // Create GitHub release
    [
      '@semantic-release/github',
      {
        successComment: `🎉 This issue has been resolved in version \${nextRelease.version}. The release is available at:

**📦 NPM Package:** https://www.npmjs.com/package/repz-platform/v/\${nextRelease.version}
**🏷️ GitHub Release:** \${releases[0].url}
**📋 Changelog:** https://github.com/repz/repzcoach-pro-ai/blob/main/CHANGELOG.md

Thank you for reporting this issue! 🙏`,
        failComment: '❌ The release from branch `\${branch.name}` had failed due to the following errors:\n- \${errors.map(err => err.message).join("\\n- ")}',
        failTitle: '🚨 Release Failed',
        labels: ['released'],
        assignees: [],
        releasedLabels: ['released on @\${branch.name}'],
        addReleases: 'bottom',
        assets: [
          // Include build artifacts in release
          { path: 'dist.zip', label: 'Production Build' },
          { path: 'storybook-static.zip', label: 'Storybook Build' },
          { path: 'coverage/lcov-report.zip', label: 'Test Coverage Report' }
        ]
      }
    ],

    // Commit updated files back to repository
    [
      '@semantic-release/git',
      {
        assets: [
          'CHANGELOG.md',
          'package.json',
          'package-lock.json',
          'packages/*/package.json'
        ],
        message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
      }
    ]
  ],

  // Environment-specific configuration
  ci: true,
  debug: process.env.NODE_ENV === 'development',
  dryRun: process.env.DRY_RUN === 'true',

  // Git configuration
  repositoryUrl: 'https://github.com/repz/repzcoach-pro-ai.git',
  tagFormat: 'v${version}',

  // Success and failure conditions
  success: [
    '@semantic-release/github',
    '@semantic-release/npm'
  ],
  
  fail: [
    '@semantic-release/github'
  ]
};