// ATLAS Fallback Chain - Resilience and fault tolerance 
 
import { Task, TaskType } from '../types/index'; 
 
export enum CircuitState { 
  CLOSED = 'closed', 
  OPEN = 'open', 
  HALF_OPEN = 'half_open' 
} 
 
export class CircuitBreaker { 
  private state: CircuitState = CircuitState.CLOSED; 
} 
