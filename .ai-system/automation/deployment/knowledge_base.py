#!/usr/bin/env python3
"""
Knowledge Base Deployment System
Organizes and deploys knowledge bases with web interfaces.
"""

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# File type categories
FILE_CATEGORIES = {
    "documents": [".md", ".txt", ".pdf", ".doc", ".docx", ".rtf"],
    "code": [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb"],
    "data": [".json", ".yaml", ".yml", ".xml", ".csv", ".sql"],
    "images": [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"],
    "archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
    "config": [".env", ".ini", ".toml", ".cfg"],
    "web": [".html", ".css", ".scss", ".less"],
    "notebooks": [".ipynb"],
    "media": [".mp3", ".mp4", ".wav", ".avi", ".mov"],
    "other": []
}


@dataclass
class OrganizationResult:
    """Result of organizing files."""
    total_files: int = 0
    organized_files: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    file_map: Dict[str, str] = field(default_factory=dict)


class KnowledgeBaseDeployer:
    """Organize and deploy knowledge bases."""

    def __init__(self, source_path: Path, output_path: Path = None):
        self.source_path = Path(source_path)
        self.output_path = output_path or self.source_path / "ORGANIZED"
        self.web_path = self.output_path.parent / "knowledge_base_web"
        self.index_data: Dict[str, Any] = {}

    def categorize_file(self, file_path: Path) -> str:
        """Determine category for a file."""
        ext = file_path.suffix.lower()

        for category, extensions in FILE_CATEGORIES.items():
            if ext in extensions:
                return category

        return "other"

    def organize_files(self, dry_run: bool = False) -> OrganizationResult:
        """Organize files into categories."""
        result = OrganizationResult()

        # Create category directories
        for category in FILE_CATEGORIES.keys():
            category_path = self.output_path / category
            if not dry_run:
                category_path.mkdir(parents=True, exist_ok=True)

        # Process files
        for file_path in self.source_path.iterdir():
            if file_path.is_file():
                result.total_files += 1

                # Skip hidden files and system files
                if file_path.name.startswith('.') or file_path.name.startswith('~'):
                    continue

                category = self.categorize_file(file_path)
                dest_path = self.output_path / category / file_path.name

                # Handle duplicates
                if dest_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = self.output_path / category / f"{stem}_{counter}{suffix}"
                        counter += 1

                try:
                    if not dry_run:
                        shutil.copy2(file_path, dest_path)

                    result.organized_files += 1
                    result.categories[category] = result.categories.get(category, 0) + 1
                    result.file_map[str(file_path)] = str(dest_path)

                except Exception as e:
                    result.errors.append(f"Failed to copy {file_path.name}: {e}")

        return result

    def build_index(self) -> Dict[str, Any]:
        """Build searchable index of organized files."""
        index = {
            "created_at": datetime.now().isoformat(),
            "source": str(self.source_path),
            "categories": {},
            "total_files": 0,
            "total_size": 0
        }

        for category_path in self.output_path.iterdir():
            if category_path.is_dir():
                category_name = category_path.name
                files = []

                for file_path in category_path.rglob("*"):
                    if file_path.is_file():
                        file_info = {
                            "name": file_path.name,
                            "path": str(file_path.relative_to(self.output_path)),
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat()
                        }
                        files.append(file_info)
                        index["total_files"] += 1
                        index["total_size"] += file_info["size"]

                if files:
                    index["categories"][category_name] = {
                        "count": len(files),
                        "files": files
                    }

        self.index_data = index

        # Save index
        index_path = self.output_path / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        return index

    def generate_web_interface(self) -> Path:
        """Generate a web interface for the knowledge base."""
        self.web_path.mkdir(parents=True, exist_ok=True)

        # Build index if not already done
        if not self.index_data:
            self.build_index()

        # Generate HTML
        html_content = self._generate_html()

        index_html = self.web_path / "index.html"
        index_html.write_text(html_content, encoding='utf-8')

        # Copy organized files to web directory
        web_files = self.web_path / "files"
        if web_files.exists():
            shutil.rmtree(web_files)
        shutil.copytree(self.output_path, web_files, dirs_exist_ok=True)

        # Generate search data
        self._generate_search_data()

        return self.web_path

    def _generate_html(self) -> str:
        """Generate the main HTML interface."""
        categories_html = ""

        for category, data in self.index_data.get("categories", {}).items():
            files_html = ""
            for f in data["files"][:20]:  # Limit to 20 files per category
                size_kb = f["size"] / 1024
                files_html += f'''
                    <li>
                        <a href="files/{f['path']}" target="_blank">{f['name']}</a>
                        <span class="file-size">({size_kb:.1f} KB)</span>
                    </li>'''

            if data["count"] > 20:
                files_html += f'<li class="more">...and {data["count"] - 20} more files</li>'

            icon = self._get_category_icon(category)
            categories_html += f'''
            <div class="category" data-category="{category}">
                <h3>{icon} {category.title()}</h3>
                <p class="count">{data['count']} files</p>
                <ul class="file-list category-files">{files_html}
                </ul>
            </div>'''

        total_size_mb = self.index_data.get("total_size", 0) / (1024 * 1024)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Organized Knowledge Base - Searchable file repository">
    <title>Knowledge Base</title>
    <style>
        :root {{
            --primary: #3498db;
            --primary-dark: #2980b9;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #2c3e50;
            --text-muted: #7f8c8d;
            --border: #e9ecef;
            --success: #27ae60;
            --warning: #f39c12;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
        }}

        /* Skip link for accessibility */
        .skip-link {{
            position: absolute;
            left: -9999px;
            top: auto;
            width: 1px;
            height: 1px;
            overflow: hidden;
        }}
        .skip-link:focus {{
            position: fixed;
            top: 0;
            left: 0;
            width: auto;
            height: auto;
            padding: 1rem;
            background: var(--primary);
            color: white;
            z-index: 9999;
            text-decoration: none;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            margin-bottom: 2rem;
        }}

        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}

        .stats {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}

        .stat {{
            background: var(--bg);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }}

        .stat h3 {{
            font-size: 1.75rem;
            color: var(--primary);
        }}

        .stat p {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        .search-container {{
            margin-bottom: 2rem;
        }}

        .search-box {{
            width: 100%;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            border: 2px solid var(--border);
            border-radius: 8px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}

        .search-box:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }}

        .categories {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.5rem;
        }}

        .category {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .category:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .category h3 {{
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--primary);
        }}

        .category .count {{
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }}

        .file-list {{
            list-style: none;
            max-height: 250px;
            overflow-y: auto;
        }}

        .file-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .file-list li:last-child {{
            border-bottom: none;
        }}

        .file-list a {{
            color: var(--primary);
            text-decoration: none;
            word-break: break-word;
        }}

        .file-list a:hover {{
            text-decoration: underline;
        }}

        .file-list a:focus {{
            outline: 2px solid var(--primary);
            outline-offset: 2px;
        }}

        .file-size {{
            color: var(--text-muted);
            font-size: 0.75rem;
            white-space: nowrap;
            margin-left: 0.5rem;
        }}

        .more {{
            color: var(--text-muted);
            font-style: italic;
        }}

        .hidden {{
            display: none !important;
        }}

        footer {{
            margin-top: 3rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            .categories {{
                grid-template-columns: 1fr;
            }}

            .stats {{
                flex-direction: column;
            }}

            .stat {{
                width: 100%;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            * {{
                transition: none !important;
            }}
        }}
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>

    <div class="container">
        <header>
            <h1>📚 Knowledge Base</h1>
            <p>Your organized file repository - searchable and accessible</p>

            <div class="stats" role="region" aria-label="Statistics">
                <div class="stat">
                    <h3>{self.index_data.get('total_files', 0)}</h3>
                    <p>Total Files</p>
                </div>
                <div class="stat">
                    <h3>{len(self.index_data.get('categories', {}))}</h3>
                    <p>Categories</p>
                </div>
                <div class="stat">
                    <h3>{total_size_mb:.1f} MB</h3>
                    <p>Total Size</p>
                </div>
                <div class="stat">
                    <h3>100%</h3>
                    <p>Organized</p>
                </div>
            </div>
        </header>

        <main id="main-content">
            <div class="search-container">
                <label for="searchBox" class="visually-hidden">Search files</label>
                <input
                    type="search"
                    id="searchBox"
                    class="search-box"
                    placeholder="🔍 Search files by name or category..."
                    aria-label="Search files"
                >
            </div>

            <div class="categories" role="region" aria-label="File categories">
                {categories_html}
            </div>
        </main>

        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} |
               <a href="https://github.com/alawein">GitHub</a>
            </p>
        </footer>
    </div>

    <script>
        // Search functionality
        const searchBox = document.getElementById('searchBox');
        const categories = document.querySelectorAll('.category');

        searchBox.addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();

            categories.forEach(category => {{
                const categoryName = category.dataset.category.toLowerCase();
                const files = category.querySelectorAll('.file-list li');
                let hasVisibleFiles = false;

                files.forEach(file => {{
                    const fileName = file.textContent.toLowerCase();
                    const matches = fileName.includes(query) || categoryName.includes(query);
                    file.classList.toggle('hidden', !matches && query !== '');
                    if (matches || query === '') hasVisibleFiles = true;
                }});

                category.classList.toggle('hidden', !hasVisibleFiles && query !== '');
            }});
        }});

        // Keyboard navigation
        searchBox.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                this.value = '';
                this.dispatchEvent(new Event('input'));
            }}
        }});
    </script>
</body>
</html>'''

    def _get_category_icon(self, category: str) -> str:
        """Get emoji icon for category."""
        icons = {
            "documents": "📄",
            "code": "💻",
            "data": "📊",
            "images": "🖼️",
            "archives": "📦",
            "config": "⚙️",
            "web": "🌐",
            "notebooks": "📓",
            "media": "🎬",
            "other": "📁"
        }
        return icons.get(category, "📁")

    def _generate_search_data(self):
        """Generate search index for client-side search."""
        search_data = []

        for category, data in self.index_data.get("categories", {}).items():
            for f in data["files"]:
                search_data.append({
                    "name": f["name"],
                    "category": category,
                    "path": f"files/{f['path']}",
                    "size": f["size"]
                })

        search_js = self.web_path / "search-data.js"
        search_js.write_text(
            f"const SEARCH_DATA = {json.dumps(search_data)};",
            encoding='utf-8'
        )

    def deploy(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run full deployment pipeline."""
        print(f"{'[DRY RUN] ' if dry_run else ''}Deploying Knowledge Base...")
        print(f"Source: {self.source_path}")
        print(f"Output: {self.output_path}")
        print("-" * 50)

        # Organize files
        print("Organizing files...")
        org_result = self.organize_files(dry_run=dry_run)
        print(f"  Organized: {org_result.organized_files}/{org_result.total_files} files")
        for cat, count in org_result.categories.items():
            print(f"    {cat}: {count}")

        if org_result.errors:
            print(f"  Errors: {len(org_result.errors)}")
            for err in org_result.errors[:5]:
                print(f"    - {err}")

        # Build index
        print("\nBuilding index...")
        index = self.build_index()
        print(f"  Total files: {index['total_files']}")
        print(f"  Total size: {index['total_size'] / (1024*1024):.1f} MB")

        # Generate web interface
        if not dry_run:
            print("\nGenerating web interface...")
            web_path = self.generate_web_interface()
            print(f"  Web interface: {web_path}")

        return {
            "success": True,
            "organized": org_result.organized_files,
            "total": org_result.total_files,
            "categories": org_result.categories,
            "web_path": str(self.web_path) if not dry_run else None,
            "index": index
        }


if __name__ == "__main__":
    import sys

    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads"

    deployer = KnowledgeBaseDeployer(source)
    result = deployer.deploy(dry_run="--dry-run" in sys.argv)

    print("\n" + "=" * 50)
    print("DEPLOYMENT COMPLETE")
    print("=" * 50)
    print(f"Files organized: {result['organized']}")
    if result['web_path']:
        print(f"Web interface: {result['web_path']}")
        print(f"Open: file://{result['web_path']}/index.html")
