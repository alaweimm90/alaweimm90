import * as path from 'path';
import { ensureDir, loadJson, saveJson } from '@ai/utils/file-persistence.js';
import { AgentCapability } from '@atlas/types/index.js';

export interface AgentTeam {
  id: string;
  name: string;
  members: string[];
  capabilities: AgentCapability[];
  routingStrategy: 'capability' | 'load_balance' | 'cost' | 'latency';
  createdAt: string;
  updatedAt?: string;
}

interface TeamState {
  teams: AgentTeam[];
  lastUpdated: string;
  version: string;
}

const ROOT = process.cwd();
const ATLAS_DIR = path.join(ROOT, '.atlas');
const TEAM_FILE = path.join(ATLAS_DIR, 'team-registry.json');

class TeamRegistry {
  private state: TeamState;
  constructor() {
    ensureDir(ATLAS_DIR);
    const def: TeamState = { teams: [], lastUpdated: new Date().toISOString(), version: '1.0.0' };
    this.state = loadJson<TeamState>(TEAM_FILE, def) || def;
  }
  private save(): void {
    this.state.lastUpdated = new Date().toISOString();
    saveJson(TEAM_FILE, this.state);
  }
  list(): AgentTeam[] {
    return [...this.state.teams];
  }
  get(id: string): AgentTeam | undefined {
    return this.state.teams.find((t) => t.id === id);
  }
  register(team: Omit<AgentTeam, 'createdAt'>): AgentTeam {
    const existing = this.state.teams.findIndex((t) => t.id === team.id);
    const full: AgentTeam = { ...team, createdAt: new Date().toISOString() };
    if (existing >= 0)
      this.state.teams[existing] = {
        ...full,
        createdAt: this.state.teams[existing].createdAt,
        updatedAt: new Date().toISOString(),
      };
    else this.state.teams.push(full);
    this.save();
    return full;
  }
  remove(id: string): boolean {
    const i = this.state.teams.findIndex((t) => t.id === id);
    if (i < 0) return false;
    this.state.teams.splice(i, 1);
    this.save();
    return true;
  }
}

export const teamRegistry = new TeamRegistry();
export type { AgentCapability };
