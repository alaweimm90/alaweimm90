import { Agent } from './types';
import {
  Ringmaster,
  BlueprintBoss,
  DealMaker,
  EthicsEnforcer,
  PuzzleProdigy,
  QuantumQuokka,
  MLMagician,
  AnalogyAlchemist,
  ProofPirate,
  VerificationVigilante,
  BenchmarkBandit,
  CodeCowboy,
  NoveltyNinja,
  SkepticSorcerer,
} from './agents';

const instances: Agent[] = [
  new Ringmaster(),
  new BlueprintBoss(),
  new DealMaker(),
  new EthicsEnforcer(),
  new PuzzleProdigy(),
  new QuantumQuokka(),
  new MLMagician(),
  new AnalogyAlchemist(),
  new ProofPirate(),
  new VerificationVigilante(),
  new BenchmarkBandit(),
  new CodeCowboy(),
  new NoveltyNinja(),
  new SkepticSorcerer(),
];

export const AgentRegistry = {
  list: () => instances,
  byName: (name: Agent['name']) => instances.find(a => a.name === name) || null,
};