#!/usr/bin/env python3
"""
PromptForge - Offline Prompt Pattern Extraction

Extracts reusable prompt patterns from notes/documentation using regex.
No ML required - works completely offline.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .extractor import PatternExtractor
from .library import PromptLibrary
from .models import ExtractionResult


def extract_from_file(file_path: Path, extractor: PatternExtractor):
    """Extract patterns from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return extractor.extract_from_text(content, str(file_path))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def extract_from_directory(directory: Path, extractor: PatternExtractor) -> ExtractionResult:
    """Extract patterns from all markdown/text files in directory"""
    patterns = []
    files_processed = 0

    # Find all markdown and text files
    for ext in ['*.md', '*.txt', '*.markdown']:
        for file_path in directory.rglob(ext):
            print(f"Processing: {file_path}")
            file_patterns = extract_from_file(file_path, extractor)
            patterns.extend(file_patterns)
            files_processed += 1

    result = ExtractionResult(
        total_patterns=len(patterns),
        patterns=patterns,
        files_processed=files_processed,
        timestamp=datetime.now().isoformat()
    )

    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PromptForge - Offline Prompt Pattern Extraction"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract from single file')
    extract_parser.add_argument('--input', required=True, help='Input file path')
    extract_parser.add_argument('--output', required=True, help='Output JSON file')
    extract_parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory')
    scan_parser.add_argument('--directory', required=True, help='Directory to scan')
    scan_parser.add_argument('--output', required=True, help='Output JSON file')
    scan_parser.add_argument('--min-confidence', type=float, default=0.4, help='Minimum confidence')

    # Template command
    template_parser = subparsers.add_parser('template', help='Get template by name')
    template_parser.add_argument('--input', required=True, help='Patterns JSON file')
    template_parser.add_argument('--name', help='Pattern name to retrieve')
    template_parser.add_argument('--type', help='Filter by pattern type')
    template_parser.add_argument('--tag', help='Filter by tag')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    extractor = PatternExtractor()

    if args.command == 'extract':
        # Extract from single file
        file_path = Path(args.input)
        patterns = extract_from_file(file_path, extractor)

        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= args.min_confidence]

        result = ExtractionResult(
            total_patterns=len(patterns),
            patterns=patterns,
            files_processed=1,
            timestamp=datetime.now().isoformat()
        )

        # Save to JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\nExtracted {len(patterns)} patterns from {file_path}")
        print(f"Saved to: {args.output}")

    elif args.command == 'scan':
        # Scan directory
        directory = Path(args.directory)
        result = extract_from_directory(directory, extractor)

        # Filter by confidence
        result.patterns = [p for p in result.patterns if p.confidence >= args.min_confidence]
        result.total_patterns = len(result.patterns)

        # Deduplicate
        library = PromptLibrary()
        library.add_patterns(result.patterns)
        library.deduplicate()
        result.patterns = library.patterns
        result.total_patterns = len(library.patterns)

        # Save to JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\nScanned {result.files_processed} files")
        print(f"Extracted {result.total_patterns} unique patterns")
        print(f"Saved to: {args.output}")

    elif args.command == 'template':
        # Load patterns
        with open(args.input, 'r') as f:
            data = json.load(f)

        library = PromptLibrary()
        library.patterns = [
            PromptPattern.from_dict(p) for p in data['patterns']
        ]

        # Filter patterns
        if args.name:
            pattern = library.get_template(args.name)
            if pattern:
                print(f"\nPattern: {pattern.pattern_name}")
                print(f"Type: {pattern.pattern_type}")
                print(f"Confidence: {pattern.confidence}")
                print(f"Variables: {', '.join(pattern.variables) if pattern.variables else 'None'}")
                print(f"\nTemplate:\n{pattern.template}")
            else:
                print(f"Pattern '{args.name}' not found")

        elif args.type:
            patterns = library.filter_by_type(args.type)
            print(f"\nFound {len(patterns)} patterns of type '{args.type}':")
            for p in patterns:
                print(f"  - {p.pattern_name} (confidence: {p.confidence})")

        elif args.tag:
            patterns = library.filter_by_tag(args.tag)
            print(f"\nFound {len(patterns)} patterns with tag '{args.tag}':")
            for p in patterns:
                print(f"  - {p.pattern_name} (confidence: {p.confidence})")

        else:
            # Show top patterns by reusability
            top_patterns = library.rank_by_reusability()[:10]
            print(f"\nTop 10 most reusable patterns:")
            for i, p in enumerate(top_patterns, 1):
                print(f"{i}. {p.pattern_name} (type: {p.pattern_type}, confidence: {p.confidence})")


if __name__ == "__main__":
    main()
