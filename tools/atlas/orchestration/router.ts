// ATLAS Task Router - Intelligent task routing and load balancing 
 
import { Agent, Task, agentRegistry } from '../agents/registry'; 
import { OrchestrationConfig } from '../types/index'; 
 
export interface RoutingDecision { 
  agentId: string; 
  confidence: number; 
  reasoning: string; 
  estimatedCost: number; 
  estimatedTime: number; 
} 
 
export class TaskRouter { 
  constructor(config: OrchestrationConfig) { this.config = config; } 
} 
