"""Test cross-IDE sync"""
from syncer import PromptSyncer
from config import IDEConfig

def test_sync():
    print("\n[TEST] Cross-IDE Sync")
    
    config = IDEConfig()
    print(f"  Source: {config.source}")
    print(f"  Source exists: {config.source.exists()}")
    
    # Show IDE paths
    print("\n  IDE Paths:")
    for ide, paths in config.get_ide_paths().items():
        print(f"    {ide}: {paths[0]}")
    
    # Test sync
    syncer = PromptSyncer()
    print(f"\n  Syncing from: {syncer.source}")
    
    results = syncer.sync_all()
    
    print("\n  Sync Results:")
    for ide, result in results.items():
        if result["status"] == "success":
            print(f"    [OK] {ide}: {result['files']} files synced")
        else:
            print(f"    [ERR] {ide}: {result['message']}")
    
    total = sum(r['files'] for r in results.values() if r['status'] == 'success')
    print(f"\n[OK] Synced {total} total files across {len(results)} IDEs")

if __name__ == "__main__":
    test_sync()
