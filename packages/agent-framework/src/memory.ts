import { ContextualMemory, MemoryEntry } from './interfaces';

export class InMemoryContextualMemory implements ContextualMemory {
  shortTerm: MemoryEntry[] = [];
  longTerm: MemoryEntry[] = [];
  episodic: MemoryEntry[] = [];
  private maxShortTermSize = 100;
  private maxLongTermSize = 1000;

  add(entry: MemoryEntry): void {
    // Add to episodic memory
    this.episodic.push(entry);

    // Add to short-term memory with size limit
    this.shortTerm.push(entry);
    if (this.shortTerm.length > this.maxShortTermSize) {
      this.shortTerm.shift();
    }

    // Consolidate to long-term if important
    if (entry.confidence > 0.8) {
      this.longTerm.push(entry);
      if (this.longTerm.length > this.maxLongTermSize) {
        // Remove oldest low-confidence entries
        this.longTerm.sort((a, b) => b.confidence - a.confidence);
        this.longTerm = this.longTerm.slice(0, this.maxLongTermSize);
      }
    }
  }

  retrieve(query: string): MemoryEntry[] {
    const allMemories = [...this.shortTerm, ...this.longTerm, ...this.episodic];
    return allMemories.filter(entry =>
      entry.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase())) ||
      JSON.stringify(entry.content).toLowerCase().includes(query.toLowerCase())
    );
  }

  consolidate(): void {
    // Move high-confidence short-term memories to long-term
    const toConsolidate = this.shortTerm.filter(entry => entry.confidence > 0.7);
    this.longTerm.push(...toConsolidate);

    // Clean up duplicates and maintain size limits
    const uniqueEntries = new Map<string, MemoryEntry>();
    [...this.longTerm].forEach(entry => {
      uniqueEntries.set(entry.id, entry);
    });
    this.longTerm = Array.from(uniqueEntries.values()).slice(0, this.maxLongTermSize);
  }

  getStats() {
    return {
      shortTermCount: this.shortTerm.length,
      longTermCount: this.longTerm.length,
      episodicCount: this.episodic.length,
      totalMemories: this.shortTerm.length + this.longTerm.length + this.episodic.length
    };
  }
}