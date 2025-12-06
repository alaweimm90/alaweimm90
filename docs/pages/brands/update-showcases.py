#!/usr/bin/env python3
"""Add showcase banners to all brand pages and remove external links."""
import os
import re

brands_dir = os.path.dirname(os.path.abspath(__file__))

showcase_banner = '''  <div class="showcase-banner" style="background: linear-gradient(90deg, var(--primary, #38bdf8), #a855f7); color: white; text-align: center; padding: 0.5rem 1rem; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em;">
    🎨 DESIGN SHOWCASE — This is a concept mockup. <a href="../" style="color: white; text-decoration: underline;">View all brand designs →</a>
  </div>
'''

for brand in os.listdir(brands_dir):
    brand_path = os.path.join(brands_dir, brand, 'index.html')
    if os.path.isfile(brand_path):
        with open(brand_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has showcase banner
        if 'showcase-banner' in content:
            print(f'✓ {brand} - already has banner')
            continue

        # Add banner after <body...>
        new_content = re.sub(r'(<body[^>]*>)', r'\1\n' + showcase_banner, content, count=1)

        # Remove external domain links (replace with #)
        new_content = re.sub(r'href="https?://[^"]+"', 'href="#"', new_content)

        with open(brand_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f'✅ {brand} - updated')

print('\nDone!')
