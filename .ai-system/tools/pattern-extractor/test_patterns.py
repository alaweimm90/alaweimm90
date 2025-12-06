"""Test pattern extraction"""
from extractor import PatternExtractor
from library import PatternLibrary

def test_extraction():
    print("\n[TEST] Pattern Extraction")
    
    extractor = PatternExtractor()
    patterns = extractor.extract_all()
    
    print(f"  Extracted {len(patterns)} patterns")
    for p in patterns:
        print(f"    - {p.name}: {p.frequency} occurrences")
        print(f"      Examples: {p.examples[:2]}")
    
    # Test library
    library = PatternLibrary()
    library.save(patterns)
    print(f"\n  Saved to library")
    
    # Test retrieval
    loaded = library.list_all()
    print(f"  Library contains: {loaded}")
    
    # Test template generation
    print(f"\n  Template for 'section-structure':")
    template = extractor.generate_template("section-structure")
    print(template[:100] + "...")
    
    print("\n[OK] Pattern extraction test complete")

if __name__ == "__main__":
    test_extraction()
