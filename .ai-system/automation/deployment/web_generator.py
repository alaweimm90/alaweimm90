#!/usr/bin/env python3
"""
Web Interface Generator
Generate accessible, modern web interfaces for various content types.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class WebConfig:
    """Configuration for web interface generation."""
    title: str
    description: str = ""
    theme: str = "light"  # light, dark, auto
    primary_color: str = "#3498db"
    logo_url: Optional[str] = None
    footer_text: str = ""
    enable_search: bool = True
    enable_dark_mode: bool = True
    accessibility_level: str = "AA"


class WebInterfaceGenerator:
    """Generate accessible web interfaces."""

    def __init__(self, output_path: Path, config: WebConfig = None):
        self.output_path = Path(output_path)
        self.config = config or WebConfig(title="Web Interface")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(
        self,
        sections: List[Dict[str, Any]],
        stats: Dict[str, Any] = None
    ) -> Path:
        """Generate a dashboard interface."""

        stats_html = ""
        if stats:
            stats_items = ""
            for label, value in stats.items():
                stats_items += f'''
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>'''
            stats_html = f'<div class="stats-grid">{stats_items}</div>'

        sections_html = ""
        for section in sections:
            content = section.get("content", "")
            if isinstance(content, list):
                items = "".join(f"<li>{item}</li>" for item in content)
                content = f"<ul>{items}</ul>"

            sections_html += f'''
            <section class="dashboard-section" aria-labelledby="section-{section.get('id', 'default')}">
                <h2 id="section-{section.get('id', 'default')}">{section.get('title', 'Section')}</h2>
                <div class="section-content">{content}</div>
            </section>'''

        html = self._generate_base_html(
            body_content=f'''
            <header class="dashboard-header">
                <h1>{self.config.title}</h1>
                <p>{self.config.description}</p>
                {stats_html}
            </header>
            <main id="main-content" class="dashboard-main">
                {sections_html}
            </main>
            ''',
            extra_styles=self._dashboard_styles()
        )

        output_file = self.output_path / "dashboard.html"
        output_file.write_text(html, encoding='utf-8')
        return output_file

    def generate_file_browser(
        self,
        files: List[Dict[str, Any]],
        base_path: str = "files"
    ) -> Path:
        """Generate a file browser interface."""

        # Group files by category/folder
        grouped = {}
        for f in files:
            category = f.get("category", "other")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(f)

        categories_html = ""
        for category, category_files in grouped.items():
            files_html = ""
            for f in category_files[:50]:  # Limit per category
                size = f.get("size", 0)
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                files_html += f'''
                <tr>
                    <td><a href="{base_path}/{f.get('path', f.get('name'))}">{f.get('name', 'Unknown')}</a></td>
                    <td>{size_str}</td>
                    <td>{f.get('modified', '-')}</td>
                </tr>'''

            categories_html += f'''
            <div class="file-category" data-category="{category}">
                <h3>{self._get_icon(category)} {category.title()} ({len(category_files)} files)</h3>
                <table class="file-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Size</th>
                            <th>Modified</th>
                        </tr>
                    </thead>
                    <tbody>{files_html}</tbody>
                </table>
            </div>'''

        html = self._generate_base_html(
            body_content=f'''
            <header>
                <h1>📁 {self.config.title}</h1>
                <p>{self.config.description}</p>
            </header>
            <main id="main-content">
                <div class="search-bar">
                    <input type="search" id="fileSearch" placeholder="Search files..." aria-label="Search files">
                </div>
                <div class="file-browser">
                    {categories_html}
                </div>
            </main>
            ''',
            extra_styles=self._file_browser_styles(),
            extra_scripts=self._file_browser_scripts()
        )

        output_file = self.output_path / "files.html"
        output_file.write_text(html, encoding='utf-8')
        return output_file

    def generate_documentation(
        self,
        docs: List[Dict[str, Any]],
        sidebar: bool = True
    ) -> Path:
        """Generate a documentation interface."""

        nav_html = ""
        content_html = ""

        for i, doc in enumerate(docs):
            doc_id = doc.get("id", f"doc-{i}")
            nav_html += f'<li><a href="#{doc_id}">{doc.get("title", "Untitled")}</a></li>'

            content_html += f'''
            <article id="{doc_id}" class="doc-article">
                <h2>{doc.get("title", "Untitled")}</h2>
                <div class="doc-content">{doc.get("content", "")}</div>
            </article>'''

        sidebar_html = f'''
        <nav class="doc-sidebar" aria-label="Documentation navigation">
            <ul>{nav_html}</ul>
        </nav>''' if sidebar else ""

        html = self._generate_base_html(
            body_content=f'''
            <header class="doc-header">
                <h1>📖 {self.config.title}</h1>
            </header>
            <div class="doc-layout">
                {sidebar_html}
                <main id="main-content" class="doc-main">
                    {content_html}
                </main>
            </div>
            ''',
            extra_styles=self._documentation_styles()
        )

        output_file = self.output_path / "docs.html"
        output_file.write_text(html, encoding='utf-8')
        return output_file

    def generate_portfolio(
        self,
        projects: List[Dict[str, Any]],
        about: Dict[str, Any] = None
    ) -> Path:
        """Generate a portfolio interface."""

        about_html = ""
        if about:
            about_html = f'''
            <section class="about-section" aria-labelledby="about-heading">
                <h2 id="about-heading">About</h2>
                <div class="about-content">
                    <h3>{about.get('name', '')}</h3>
                    <p>{about.get('bio', '')}</p>
                    <div class="social-links">
                        {self._generate_social_links(about.get('links', {}))}
                    </div>
                </div>
            </section>'''

        projects_html = ""
        for project in projects:
            tags = "".join(f'<span class="tag">{tag}</span>' for tag in project.get("tags", []))
            links = ""
            if project.get("url"):
                links += f'<a href="{project["url"]}" class="project-link">View Project</a>'
            if project.get("github"):
                links += f'<a href="{project["github"]}" class="project-link">GitHub</a>'

            projects_html += f'''
            <article class="project-card">
                <h3>{project.get('title', 'Project')}</h3>
                <p>{project.get('description', '')}</p>
                <div class="project-tags">{tags}</div>
                <div class="project-links">{links}</div>
            </article>'''

        html = self._generate_base_html(
            body_content=f'''
            <header class="portfolio-header">
                <h1>{self.config.title}</h1>
                <p>{self.config.description}</p>
            </header>
            <main id="main-content">
                {about_html}
                <section class="projects-section" aria-labelledby="projects-heading">
                    <h2 id="projects-heading">Projects</h2>
                    <div class="projects-grid">
                        {projects_html}
                    </div>
                </section>
            </main>
            ''',
            extra_styles=self._portfolio_styles()
        )

        output_file = self.output_path / "portfolio.html"
        output_file.write_text(html, encoding='utf-8')
        return output_file

    def _generate_base_html(
        self,
        body_content: str,
        extra_styles: str = "",
        extra_scripts: str = ""
    ) -> str:
        """Generate base HTML structure with accessibility features."""

        dark_mode_toggle = '''
        <button id="darkModeToggle" class="dark-mode-toggle" aria-label="Toggle dark mode">
            🌙
        </button>''' if self.config.enable_dark_mode else ""

        dark_mode_script = '''
        <script>
            const toggle = document.getElementById('darkModeToggle');
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

            if (localStorage.getItem('darkMode') === 'true' ||
                (localStorage.getItem('darkMode') === null && prefersDark)) {
                document.body.classList.add('dark-mode');
                toggle.textContent = '☀️';
            }

            toggle.addEventListener('click', () => {
                document.body.classList.toggle('dark-mode');
                const isDark = document.body.classList.contains('dark-mode');
                localStorage.setItem('darkMode', isDark);
                toggle.textContent = isDark ? '☀️' : '🌙';
            });
        </script>''' if self.config.enable_dark_mode else ""

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{self.config.description}">
    <title>{self.config.title}</title>
    <style>
        :root {{
            --primary: {self.config.primary_color};
            --primary-dark: color-mix(in srgb, {self.config.primary_color} 80%, black);
            --bg: #ffffff;
            --bg-secondary: #f8f9fa;
            --text: #2c3e50;
            --text-muted: #6c757d;
            --border: #dee2e6;
            --shadow: rgba(0,0,0,0.1);
        }}

        body.dark-mode {{
            --bg: #1a1a2e;
            --bg-secondary: #16213e;
            --text: #eaeaea;
            --text-muted: #a0a0a0;
            --border: #333;
            --shadow: rgba(0,0,0,0.3);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
            transition: background-color 0.3s, color 0.3s;
        }}

        /* Accessibility */
        .skip-link {{
            position: absolute;
            left: -9999px;
            z-index: 999;
            padding: 1rem;
            background: var(--primary);
            color: white;
            text-decoration: none;
        }}
        .skip-link:focus {{
            left: 50%;
            transform: translateX(-50%);
            top: 0;
        }}

        *:focus {{
            outline: 3px solid var(--primary);
            outline-offset: 2px;
        }}

        .visually-hidden {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            border: 0;
        }}

        /* Dark mode toggle */
        .dark-mode-toggle {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.25rem;
            z-index: 100;
        }}

        /* Container */
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* Header */
        header {{
            text-align: center;
            padding: 3rem 2rem;
            background: var(--bg-secondary);
            margin-bottom: 2rem;
        }}

        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            header h1 {{ font-size: 1.75rem; }}
            .container {{ padding: 1rem; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            * {{ transition: none !important; }}
        }}

        {extra_styles}
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    {dark_mode_toggle}

    <div class="container">
        {body_content}

        <footer>
            <p>{self.config.footer_text or f'Generated on {datetime.now().strftime("%Y-%m-%d")}'}</p>
        </footer>
    </div>

    {dark_mode_script}
    {extra_scripts}
</body>
</html>'''

    def _dashboard_styles(self) -> str:
        return '''
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }
        .stat-card {
            background: var(--bg);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px var(--shadow);
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }
        .stat-label {
            color: var(--text-muted);
            font-size: 0.875rem;
        }
        .dashboard-section {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .dashboard-section h2 {
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary);
        }
        '''

    def _file_browser_styles(self) -> str:
        return '''
        .search-bar {
            margin-bottom: 2rem;
        }
        .search-bar input {
            width: 100%;
            padding: 1rem;
            font-size: 1rem;
            border: 2px solid var(--border);
            border-radius: 8px;
            background: var(--bg);
            color: var(--text);
        }
        .search-bar input:focus {
            border-color: var(--primary);
        }
        .file-category {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .file-category h3 {
            margin-bottom: 1rem;
        }
        .file-table {
            width: 100%;
            border-collapse: collapse;
        }
        .file-table th, .file-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .file-table th {
            background: var(--bg);
            font-weight: 600;
        }
        .file-table a {
            color: var(--primary);
            text-decoration: none;
        }
        .file-table a:hover {
            text-decoration: underline;
        }
        '''

    def _file_browser_scripts(self) -> str:
        return '''
        <script>
            document.getElementById('fileSearch').addEventListener('input', function(e) {
                const query = e.target.value.toLowerCase();
                document.querySelectorAll('.file-table tbody tr').forEach(row => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(query) ? '' : 'none';
                });
            });
        </script>
        '''

    def _documentation_styles(self) -> str:
        return '''
        .doc-layout {
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 2rem;
        }
        .doc-sidebar {
            position: sticky;
            top: 2rem;
            height: fit-content;
        }
        .doc-sidebar ul {
            list-style: none;
        }
        .doc-sidebar li {
            margin-bottom: 0.5rem;
        }
        .doc-sidebar a {
            color: var(--text);
            text-decoration: none;
            padding: 0.5rem;
            display: block;
            border-radius: 4px;
        }
        .doc-sidebar a:hover {
            background: var(--bg-secondary);
        }
        .doc-article {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .doc-article h2 {
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary);
        }
        @media (max-width: 768px) {
            .doc-layout {
                grid-template-columns: 1fr;
            }
            .doc-sidebar {
                position: static;
            }
        }
        '''

    def _portfolio_styles(self) -> str:
        return '''
        .portfolio-header {
            text-align: center;
            padding: 4rem 2rem;
        }
        .about-section, .projects-section {
            margin-bottom: 3rem;
        }
        .about-section h2, .projects-section h2 {
            margin-bottom: 1.5rem;
        }
        .projects-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        .project-card {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            transition: transform 0.2s;
        }
        .project-card:hover {
            transform: translateY(-4px);
        }
        .project-card h3 {
            margin-bottom: 0.5rem;
        }
        .project-tags {
            margin: 1rem 0;
        }
        .tag {
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .project-links {
            margin-top: 1rem;
        }
        .project-link {
            display: inline-block;
            color: var(--primary);
            text-decoration: none;
            margin-right: 1rem;
        }
        .project-link:hover {
            text-decoration: underline;
        }
        .social-links a {
            display: inline-block;
            margin-right: 1rem;
            color: var(--primary);
        }
        '''

    def _generate_social_links(self, links: Dict[str, str]) -> str:
        icons = {
            "github": "GitHub",
            "linkedin": "LinkedIn",
            "twitter": "Twitter",
            "email": "Email"
        }
        html = ""
        for platform, url in links.items():
            label = icons.get(platform, platform.title())
            html += f'<a href="{url}" target="_blank" rel="noopener">{label}</a>'
        return html

    def _get_icon(self, category: str) -> str:
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


if __name__ == "__main__":
    # Demo
    output = Path("./demo_web")
    config = WebConfig(
        title="Demo Interface",
        description="A demonstration of the web generator",
        primary_color="#6366f1"
    )

    generator = WebInterfaceGenerator(output, config)

    # Generate dashboard
    dashboard = generator.generate_dashboard(
        sections=[
            {"id": "overview", "title": "Overview", "content": "Welcome to the dashboard"},
            {"id": "stats", "title": "Statistics", "content": ["Item 1", "Item 2", "Item 3"]}
        ],
        stats={"Files": 100, "Categories": 10, "Size": "50 MB"}
    )

    print(f"Generated: {dashboard}")
