"""Sync prompts across IDEs"""
import shutil
from pathlib import Path
from typing import List
from config import IDEConfig

class PromptSyncer:
    def __init__(self):
        self.config = IDEConfig()
        self.source = self.config.source / "prompts"
    
    def sync_all(self, ides: List[str] = None) -> dict:
        """Sync prompts to all or specific IDEs"""
        results = {}
        ide_paths = self.config.get_ide_paths()
        
        if ides:
            ide_paths = {k: v for k, v in ide_paths.items() if k in ides}
        
        for ide, paths in ide_paths.items():
            results[ide] = self._sync_to_ide(paths[0])
        
        return results
    
    def _sync_to_ide(self, target: Path) -> dict:
        """Sync source prompts to target IDE"""
        if not self.source.exists():
            return {"status": "error", "message": "Source not found"}
        
        target.mkdir(parents=True, exist_ok=True)
        
        synced = 0
        for src_file in self.source.rglob("*.md"):
            rel_path = src_file.relative_to(self.source)
            dest_file = target / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            synced += 1
        
        return {"status": "success", "files": synced}
    
    def watch(self):
        """Watch for changes and auto-sync"""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class SyncHandler(FileSystemEventHandler):
                def __init__(self, syncer):
                    self.syncer = syncer
                
                def on_modified(self, event):
                    if event.src_path.endswith('.md'):
                        print(f"[SYNC] Detected change: {event.src_path}")
                        self.syncer.sync_all()
            
            observer = Observer()
            observer.schedule(SyncHandler(self), str(self.source), recursive=True)
            observer.start()
            print(f"[WATCH] Monitoring {self.source}")
            
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        except ImportError:
            print("[ERROR] watchdog not installed. Run: pip install watchdog")

if __name__ == "__main__":
    syncer = PromptSyncer()
    results = syncer.sync_all()
    
    print("\n[SYNC] Results:")
    for ide, result in results.items():
        if result["status"] == "success":
            print(f"  {ide}: {result['files']} files")
        else:
            print(f"  {ide}: {result['message']}")
