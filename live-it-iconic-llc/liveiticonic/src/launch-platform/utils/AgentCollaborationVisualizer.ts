/**
 * LiveItIconic Launch Platform - Agent Collaboration Visualizer
 *
 * Visualizes agent interactions and collaboration patterns
 */

import { BaseAgent } from '../core/BaseAgent';
import { AgentType } from '../types';
import { EventBus } from '../core/EventBus';

export interface AgentNode {
  id: string;
  name: string;
  type: AgentType;
  category: string;
  status: string;
  messagesSent: number;
  messagesReceived: number;
}

export interface AgentEdge {
  from: string;
  to: string;
  messageCount: number;
  messageTypes: string[];
  lastActivity: Date;
}

export interface CollaborationGraph {
  nodes: AgentNode[];
  edges: AgentEdge[];
  timestamp: Date;
  totalMessages: number;
}

export interface AgentCommunicationStats {
  agentId: string;
  agentName: string;
  totalSent: number;
  totalReceived: number;
  mostFrequentPartner: string | null;
  communicationScore: number;
}

export class AgentCollaborationVisualizer {
  private agents: Map<string, BaseAgent> = new Map();
  private messageLog: Array<{
    from: string;
    to: string | string[];
    type: string;
    timestamp: Date;
  }> = [];
  private maxLogSize: number = 1000;

  /**
   * Register agents for visualization
   */
  registerAgent(agent: BaseAgent): void {
    this.agents.set(agent.getId(), agent);
  }

  /**
   * Register multiple agents
   */
  registerAgents(agents: BaseAgent[]): void {
    agents.forEach(agent => this.registerAgent(agent));
  }

  /**
   * Log a message for visualization
   */
  logMessage(from: string, to: string | string[], type: string): void {
    this.messageLog.push({
      from,
      to,
      type,
      timestamp: new Date(),
    });

    // Limit log size
    if (this.messageLog.length > this.maxLogSize) {
      this.messageLog.shift();
    }
  }

  /**
   * Get agent category from type
   */
  private getAgentCategory(type: AgentType): string {
    const typeStr = type.toString();

    if (
      [
        'competitor_analyst',
        'trend_detector',
        'customer_researcher',
        'pricing_strategist',
        'market_sizer',
      ].includes(typeStr)
    ) {
      return 'Market Intelligence';
    }

    if (
      [
        'brand_architect',
        'copywriter',
        'visual_designer',
        'video_producer',
        'storyteller',
      ].includes(typeStr)
    ) {
      return 'Creative & Branding';
    }

    if (
      [
        'campaign_manager',
        'social_media_strategist',
        'influencer_outreach',
        'pr_manager',
        'email_marketer',
        'content_distributor',
      ].includes(typeStr)
    ) {
      return 'Launch Execution';
    }

    if (
      [
        'analytics_interpreter',
        'conversion_optimizer',
        'seo_specialist',
        'paid_ads_manager',
        'feedback_analyzer',
      ].includes(typeStr)
    ) {
      return 'Optimization';
    }

    return 'Supporting';
  }

  /**
   * Build collaboration graph
   */
  buildGraph(): CollaborationGraph {
    const nodes: AgentNode[] = [];
    const edgeMap: Map<string, AgentEdge> = new Map();

    // Build nodes
    for (const [agentId, agent] of this.agents) {
      const state = agent.getState();

      // Count messages
      const sent = this.messageLog.filter(m => m.from === agentId).length;
      const received = this.messageLog.filter(m => {
        if (Array.isArray(m.to)) {
          return m.to.includes(agentId);
        }
        return m.to === agentId;
      }).length;

      nodes.push({
        id: agentId,
        name: agent.getName(),
        type: agent.getType(),
        category: this.getAgentCategory(agent.getType()),
        status: state.status,
        messagesSent: sent,
        messagesReceived: received,
      });
    }

    // Build edges
    for (const message of this.messageLog) {
      const toList = Array.isArray(message.to) ? message.to : [message.to];

      for (const to of toList) {
        const key = `${message.from}->${to}`;
        const edge = edgeMap.get(key);

        if (edge) {
          edge.messageCount++;
          if (!edge.messageTypes.includes(message.type)) {
            edge.messageTypes.push(message.type);
          }
          edge.lastActivity = message.timestamp;
        } else {
          edgeMap.set(key, {
            from: message.from,
            to,
            messageCount: 1,
            messageTypes: [message.type],
            lastActivity: message.timestamp,
          });
        }
      }
    }

    return {
      nodes,
      edges: Array.from(edgeMap.values()),
      timestamp: new Date(),
      totalMessages: this.messageLog.length,
    };
  }

  /**
   * Get communication statistics
   */
  getCommunicationStats(): AgentCommunicationStats[] {
    const stats: AgentCommunicationStats[] = [];

    for (const [agentId, agent] of this.agents) {
      const sent = this.messageLog.filter(m => m.from === agentId).length;
      const received = this.messageLog.filter(m => {
        if (Array.isArray(m.to)) {
          return m.to.includes(agentId);
        }
        return m.to === agentId;
      }).length;

      // Find most frequent communication partner
      const partners: Map<string, number> = new Map();

      for (const message of this.messageLog) {
        if (message.from === agentId) {
          const toList = Array.isArray(message.to) ? message.to : [message.to];
          for (const to of toList) {
            partners.set(to, (partners.get(to) || 0) + 1);
          }
        }
      }

      let mostFrequentPartner: string | null = null;
      let maxCount = 0;

      for (const [partner, count] of partners) {
        if (count > maxCount) {
          maxCount = count;
          mostFrequentPartner = partner;
        }
      }

      const communicationScore = sent + received;

      stats.push({
        agentId,
        agentName: agent.getName(),
        totalSent: sent,
        totalReceived: received,
        mostFrequentPartner,
        communicationScore,
      });
    }

    return stats.sort((a, b) => b.communicationScore - a.communicationScore);
  }

  /**
   * Generate ASCII visualization
   */
  generateASCIIVisualization(): string {
    const graph = this.buildGraph();
    const stats = this.getCommunicationStats();

    let viz = '\n';
    viz += '╔════════════════════════════════════════════════════════════╗\n';
    viz += '║        Agent Collaboration Network Visualization          ║\n';
    viz += '╚════════════════════════════════════════════════════════════╝\n\n';

    // Overview
    viz += '📊 NETWORK OVERVIEW\n\n';
    viz += `  Total Agents: ${graph.nodes.length}\n`;
    viz += `  Total Messages: ${graph.totalMessages}\n`;
    viz += `  Active Connections: ${graph.edges.length}\n`;
    viz += `  Timestamp: ${graph.timestamp.toLocaleString()}\n\n`;

    // Agent breakdown by category
    viz += '🤖 AGENTS BY CATEGORY\n\n';
    const byCategory: Map<string, number> = new Map();
    graph.nodes.forEach(node => {
      byCategory.set(node.category, (byCategory.get(node.category) || 0) + 1);
    });

    for (const [category, count] of byCategory) {
      viz += `  ${category}: ${count} agents\n`;
    }
    viz += '\n';

    // Most active communicators
    viz += '📡 TOP COMMUNICATORS\n\n';
    const topCommunicators = stats.slice(0, 5);
    topCommunicators.forEach((stat, idx) => {
      viz += `  ${idx + 1}. ${stat.agentName}\n`;
      viz += `     Sent: ${stat.totalSent} | Received: ${stat.totalReceived}\n`;
      if (stat.mostFrequentPartner) {
        const partner = this.agents.get(stat.mostFrequentPartner);
        viz += `     Most frequent partner: ${partner?.getName() || 'Unknown'}\n`;
      }
    });
    viz += '\n';

    // Strongest connections
    viz += '🔗 STRONGEST CONNECTIONS\n\n';
    const strongestEdges = graph.edges
      .filter(e => e.messageCount > 0)
      .sort((a, b) => b.messageCount - a.messageCount)
      .slice(0, 5);

    strongestEdges.forEach((edge, idx) => {
      const fromAgent = this.agents.get(edge.from);
      const toAgent = this.agents.get(edge.to);
      viz += `  ${idx + 1}. ${fromAgent?.getName() || 'Unknown'} → ${toAgent?.getName() || 'Unknown'}\n`;
      viz += `     Messages: ${edge.messageCount}\n`;
      viz += `     Types: ${edge.messageTypes.join(', ')}\n`;
    });
    viz += '\n';

    // Category interactions
    viz += '🔀 CATEGORY INTERACTIONS\n\n';
    const categoryInteractions: Map<string, Set<string>> = new Map();

    for (const edge of graph.edges) {
      const fromNode = graph.nodes.find(n => n.id === edge.from);
      const toNode = graph.nodes.find(n => n.id === edge.to);

      if (fromNode && toNode && fromNode.category !== toNode.category) {
        const key = fromNode.category;
        if (!categoryInteractions.has(key)) {
          categoryInteractions.set(key, new Set());
        }
        categoryInteractions.get(key)!.add(toNode.category);
      }
    }

    for (const [category, connections] of categoryInteractions) {
      viz += `  ${category}:\n`;
      viz += `    → ${Array.from(connections).join(', ')}\n`;
    }
    viz += '\n';

    viz += '═══════════════════════════════════════════════════════════\n';

    return viz;
  }

  /**
   * Export graph data as JSON
   */
  exportGraph(): string {
    const graph = this.buildGraph();
    return JSON.stringify(graph, null, 2);
  }

  /**
   * Export DOT format for Graphviz
   */
  exportDOT(): string {
    const graph = this.buildGraph();

    let dot = 'digraph AgentCollaboration {\n';
    dot += '  rankdir=LR;\n';
    dot += '  node [shape=box, style=rounded];\n\n';

    // Add nodes with colors by category
    const categoryColors: Record<string, string> = {
      'Market Intelligence': 'lightblue',
      'Creative & Branding': 'lightgreen',
      'Launch Execution': 'lightyellow',
      'Optimization': 'lightcoral',
      'Supporting': 'lightgray',
    };

    for (const node of graph.nodes) {
      const color = categoryColors[node.category] || 'white';
      dot += `  "${node.name}" [fillcolor=${color}, style=filled];\n`;
    }

    dot += '\n';

    // Add edges with weights
    for (const edge of graph.edges) {
      const fromNode = graph.nodes.find(n => n.id === edge.from);
      const toNode = graph.nodes.find(n => n.id === edge.to);

      if (fromNode && toNode) {
        const weight = Math.min(edge.messageCount, 10);
        dot += `  "${fromNode.name}" -> "${toNode.name}" [label="${edge.messageCount}", penwidth=${weight}];\n`;
      }
    }

    dot += '}\n';

    return dot;
  }

  /**
   * Clear message log
   */
  clearLog(): void {
    this.messageLog = [];
  }

  /**
   * Get message log
   */
  getMessageLog() {
    return [...this.messageLog];
  }
}

// Singleton instance
export const collaborationVisualizer = new AgentCollaborationVisualizer();
