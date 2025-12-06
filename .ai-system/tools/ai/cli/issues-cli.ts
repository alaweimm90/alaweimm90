#!/usr/bin/env npx tsx
/**
 * Issues CLI
 * Command-line interface for the issue manager
 */

import { issueManager, AIIssue, IssueCategory } from '@ai/issues.js';

function displayIssues(issues: AIIssue[]): void {
  if (issues.length === 0) {
    console.log('\n✅ No issues to display\n');
    return;
  }

  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║            📋 AI ISSUE TRACKER                               ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');

  for (const issue of issues.slice(0, 10)) {
    const icon =
      issue.priority === 'critical'
        ? '🔴'
        : issue.priority === 'high'
          ? '🟠'
          : issue.priority === 'medium'
            ? '🟡'
            : '🔵';
    const status =
      issue.status === 'resolved' ? '✅' : issue.status === 'in_progress' ? '🔄' : '📌';

    console.log('║                                                              ║');
    console.log(`║  ${icon} ${issue.title.substring(0, 45)}...`.padEnd(65) + '║');
    console.log(`║     ID: ${issue.id} | Status: ${status}`.padEnd(65) + '║');
    console.log(`║     Category: ${issue.category} | Source: ${issue.source}`.padEnd(65) + '║');
    if (issue.githubIssue) {
      console.log(`║     GitHub: #${issue.githubIssue}`.padEnd(65) + '║');
    }
  }

  if (issues.length > 10) {
    console.log('║                                                              ║');
    console.log(`║  ... and ${issues.length - 10} more issues`.padEnd(65) + '║');
  }

  console.log('║                                                              ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
}

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'list':
    case 'open': {
      const category = args[1] as IssueCategory | undefined;
      const issues = issueManager.getOpen(category);
      displayIssues(issues);
      break;
    }

    case 'critical': {
      const issues = issueManager.getByPriority('critical');
      displayIssues(issues);
      break;
    }

    case 'stats': {
      const stats = issueManager.getStats();
      console.log('\n📊 Issue Statistics\n');
      console.log(`Total Issues: ${stats.total}`);
      console.log(`Open: ${stats.open}`);
      console.log(`Resolved: ${stats.resolved}`);
      console.log(`Avg Resolution Time: ${stats.avgResolutionTime}h`);
      console.log('\nBy Category:');
      for (const [cat, count] of Object.entries(stats.byCategory)) {
        if (count > 0) console.log(`  ${cat}: ${count}`);
      }
      console.log('\nBy Priority:');
      for (const [pri, count] of Object.entries(stats.byPriority)) {
        if (count > 0) console.log(`  ${pri}: ${count}`);
      }
      break;
    }

    case 'create': {
      const category = (args[1] as IssueCategory) || 'maintenance';
      const title = args.slice(2).join(' ') || 'New Issue';
      const issue = issueManager.create({
        category,
        title,
        description: 'Created via CLI',
        source: 'cli',
      });
      console.log(`\n✅ Created issue: ${issue.id}\n`);
      break;
    }

    case 'resolve': {
      const issueId = args[1];
      const resolution = args.slice(2).join(' ') || 'Resolved';
      if (issueId) {
        const resolved = issueManager.updateStatus(issueId, 'resolved', resolution);
        console.log(resolved ? `\n✅ Issue ${issueId} resolved\n` : `\n❌ Issue not found\n`);
      } else {
        console.log('Usage: issues resolve <issue-id> [resolution]');
      }
      break;
    }

    case 'gh-create': {
      const issueId = args[1];
      if (issueId) {
        const ghNumber = issueManager.createGitHubIssue(issueId);
        if (ghNumber) {
          console.log(`\n✅ Created GitHub issue #${ghNumber}\n`);
        } else {
          console.log('\n❌ Failed to create GitHub issue\n');
        }
      } else {
        console.log('Usage: issues gh-create <issue-id>');
      }
      break;
    }

    default:
      console.log(`
AI Issue Manager - Automated issue tracking

Commands:
  list [category]     List open issues
  critical            List critical priority issues
  stats               Show issue statistics
  create <cat> <title> Create new issue
  resolve <id> [msg]  Resolve an issue
  gh-create <id>      Create GitHub issue from AI issue

Categories: security, compliance, performance, bug, enhancement, maintenance
      `);
  }
}

main();
