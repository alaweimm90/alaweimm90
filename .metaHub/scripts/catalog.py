#!/usr/bin/env python3
"""catalog.py - Organization Catalog Generator"""

import json
from pathlib import Path

class CatalogBuilder:
    def __init__(self, org_path='organizations/'):
        self.org_path = Path(org_path)
        self.catalog = {'version': '1.0', 'organizations': []}
        
    def scan_organizations(self):
        """Scan and catalog all repos"""
        for org_dir in self.org_path.iterdir():
            if org_dir.is_dir():
                org_entry = {'name': org_dir.name, 'repos': []}
                for repo_dir in org_dir.iterdir():
                    if repo_dir.is_dir():
                        repo_entry = {'name': repo_dir.name, 'path': str(repo_dir)}
                        org_entry['repos'].append(repo_entry)
                self.catalog['organizations'].append(org_entry)
        return self.catalog
    
    def generate_json(self, output_file='catalog.json'):
        """Generate catalog.json"""
        with open(output_file, 'w') as f:
            json.dump(self.catalog, f, indent=2)

if __name__ == '__main__':
    builder = CatalogBuilder()
    catalog = builder.scan_organizations()
    builder.generate_json()
    print(f"Generated catalog with {len(catalog['organizations'])} organizations")
