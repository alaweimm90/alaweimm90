"""Test marketplace functionality"""
from registry import PromptRegistry
from installer import PromptInstaller

def test_marketplace():
    print("\n[TEST] Marketplace")
    
    # Test 1: Publish prompts
    print("\n  Test 1: Publishing")
    registry = PromptRegistry()
    
    prompts_to_publish = [
        {'name': 'code-optimizer', 'author': 'alawein', 'description': 'Optimize code performance', 'tags': ['optimization', 'performance']},
        {'name': 'test-generator', 'author': 'alawein', 'description': 'Generate unit tests', 'tags': ['testing', 'automation']},
        {'name': 'api-designer', 'author': 'community', 'description': 'Design REST APIs', 'tags': ['api', 'design']},
    ]
    
    for prompt in prompts_to_publish:
        prompt_id = registry.publish({**prompt, 'version': '1.0.0', 'category': 'community'})
        print(f"    Published: {prompt_id}")
    
    # Test 2: Search
    print("\n  Test 2: Search")
    results = registry.search('optimization')
    print(f"    Found {len(results)} prompts for 'optimization'")
    for r in results:
        print(f"      - {r['name']}: {r['description']}")
    
    # Test 3: Rating
    print("\n  Test 3: Rating")
    prompt_id = 'alawein/code-optimizer'
    registry.rate(prompt_id, 4.5, "Great prompt!")
    registry.rate(prompt_id, 5.0, "Excellent!")
    
    prompt = registry.get(prompt_id)
    print(f"    {prompt_id}: {prompt['rating']:.1f}/5.0 ({len(prompt['reviews'])} reviews)")
    
    # Test 4: Downloads
    print("\n  Test 4: Downloads")
    registry.download(prompt_id)
    registry.download(prompt_id)
    prompt = registry.get(prompt_id)
    print(f"    {prompt_id}: {prompt['downloads']} downloads")
    
    # Test 5: List installed
    print("\n  Test 5: Installed Prompts")
    installer = PromptInstaller()
    installed = installer.list_installed()
    print(f"    Total installed: {len(installed)}")
    
    # Group by category
    categories = {}
    for p in installed:
        cat = p['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"      {cat}: {count}")
    
    print("\n[OK] Marketplace test complete")

if __name__ == "__main__":
    test_marketplace()
