"""CLI for cross-IDE sync"""
import sys
from syncer import PromptSyncer

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py [sync|watch] [ide1,ide2,...]")
        print("\nCommands:")
        print("  sync          - Sync once to all IDEs")
        print("  sync amazonq  - Sync to specific IDE")
        print("  watch         - Watch and auto-sync on changes")
        return
    
    command = sys.argv[1]
    syncer = PromptSyncer()
    
    if command == "sync":
        ides = sys.argv[2].split(',') if len(sys.argv) > 2 else None
        results = syncer.sync_all(ides)
        
        print("\n[SYNC] Results:")
        for ide, result in results.items():
            if result["status"] == "success":
                print(f"  {ide}: {result['files']} files")
            else:
                print(f"  {ide}: {result['message']}")
    
    elif command == "watch":
        print("[WATCH] Starting file watcher...")
        print("Press Ctrl+C to stop")
        syncer.watch()
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
