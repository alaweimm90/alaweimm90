"""CLI for marketplace"""
from registry import PromptRegistry
from installer import PromptInstaller
import sys as system

def main():
    if len(system.argv) < 2:
        print("Usage: python cli.py [search|publish|install|rate|list]")
        print("\nCommands:")
        print("  search <query>           - Search marketplace")
        print("  publish <name> <file>    - Publish prompt")
        print("  install <prompt_id>      - Install prompt")
        print("  rate <prompt_id> <1-5>   - Rate prompt")
        print("  list                     - List installed")
        return
    
    command = system.argv[1]
    registry = PromptRegistry()
    installer = PromptInstaller()
    
    if command == "search":
        if len(system.argv) < 3:
            print("Error: Provide search query")
            return
        
        query = " ".join(system.argv[2:])
        results = registry.search(query)
        
        print(f"\n[SEARCH] Found {len(results)} prompts")
        for r in results[:10]:
            print(f"\n  {r['id']}")
            print(f"    Description: {r.get('description', 'N/A')}")
            print(f"    Rating: {r.get('rating', 0):.1f}/5.0")
            print(f"    Downloads: {r.get('downloads', 0)}")
            print(f"    Tags: {', '.join(r.get('tags', []))}")
    
    elif command == "publish":
        if len(system.argv) < 4:
            print("Error: Provide name and file path")
            return
        
        name = system.argv[2]
        file_path = system.argv[3]
        
        prompt_id = registry.publish({
            'name': name,
            'author': 'local',
            'description': f'Published from {file_path}',
            'tags': ['custom'],
            'version': '1.0.0',
            'category': 'community'
        })
        
        print(f"\n[PUBLISH] {prompt_id}")
    
    elif command == "install":
        if len(system.argv) < 3:
            print("Error: Provide prompt ID")
            return
        
        prompt_id = system.argv[2]
        success = installer.install(prompt_id)
        
        if success:
            print(f"\n[INSTALL] {prompt_id} installed")
        else:
            print(f"\n[ERROR] Prompt not found")
    
    elif command == "rate":
        if len(system.argv) < 4:
            print("Error: Provide prompt ID and rating (1-5)")
            return
        
        prompt_id = system.argv[2]
        rating = float(system.argv[3])
        review = " ".join(system.argv[4:]) if len(system.argv) > 4 else None
        
        registry.rate(prompt_id, rating, review)
        print(f"\n[RATE] {prompt_id}: {rating}/5.0")
    
    elif command == "list":
        installed = installer.list_installed()
        
        print(f"\n[INSTALLED] {len(installed)} prompts")
        
        # Group by category
        by_category = {}
        for p in installed:
            cat = p['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(p['name'])
        
        for cat, prompts in sorted(by_category.items()):
            print(f"\n  {cat}:")
            for name in sorted(prompts)[:5]:
                print(f"    - {name}")
            if len(prompts) > 5:
                print(f"    ... and {len(prompts)-5} more")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
