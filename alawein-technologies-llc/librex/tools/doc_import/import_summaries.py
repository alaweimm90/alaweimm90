from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime

TECH_TERMS: tuple[str, ...] = ("method", "theory", "algorithm", "optimization", "qap")
BANNER_PHRASE: str = "Notation Standards"
BANNER: str = (
    "\n---\n**Notation Standards**: See [NOTATION_STANDARDS.md](NOTATION_STANDARDS.md) "
    "for consistent mathematical notation across all Librex.QAP documentation.\n---\n\n"
)

def is_technical_doc(path: Path) -> bool:
    return any(term in str(path).lower() for term in TECH_TERMS)

def inject_banner(content: str) -> str:
    if content.startswith("#"):
        pos = content.find("\n\n")
        if pos != -1:
            return content[: pos + 2] + BANNER + content[pos + 2 :]
    return BANNER + content

def write_provenance(dst: Path, src: Path) -> None:
    ts = datetime.now().isoformat()
    header = f"Source: {src}\nImported: {ts}\n\n"
    try:
        content = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = src.read_text(encoding="utf-8", errors="ignore")
    if is_technical_doc(dst):
        content = inject_banner(content)
    dst.write_text(header + content, encoding="utf-8")

def build_index(dest: Path) -> None:
    index = dest / "INDEX.md"
    items = collect_items(dest)
    order = [
        "Papers",
        "ArXiv",
        "Proposals",
        "Patents & IP",
        "Legal & Compliance",
        "Financial",
        "Methods & Taxonomy",
        "Theory & Literature",
        "Tutorials & Notebooks",
        "Planning",
        "General",
    ]
    lines = ["# External Summaries\n"]
    for cat in order:
        group = [x for x in items if x[0] == cat]
        if not group:
            continue
        lines.append(f"## {cat}\n")
        for _, title, rel in sorted(group, key=lambda x: x[1].lower()):
            lines.append(f"- [{title}]({rel})")
        lines.append("")
    index.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def fix_existing(dest: Path) -> int:
    count = 0
    for p in sorted(dest.glob("*.md")):
        if p.name == "INDEX.md":
            continue
        if not is_technical_doc(p):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if BANNER_PHRASE in text:
            continue
        p.write_text(inject_banner(text), encoding="utf-8")
        count += 1
    build_index(dest)
    return count

def extract_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""

def categorize(name: str, title: str) -> str:
    if "arxiv" in name:
        return "ArXiv"
    if "paper" in name or "unified" in title:
        return "Papers"
    if "proposal" in name:
        return "Proposals"
    if "patent" in name or name.startswith("ip_") or "ip " in title:
        return "Patents & IP"
    if "legal" in name or "quality_assurance" in name:
        return "Legal & Compliance"
    if "financial" in name or "projections" in name:
        return "Financial"
    if name.startswith("method") or "taxonomy" in name or "inventory" in name or "methods" in name or "tutorial" in name:
        if "tutorial" in name:
            return "Tutorials & Notebooks"
        return "Methods & Taxonomy"
    if "theory" in name or "literature" in name:
        return "Theory & Literature"
    if "timeline" in name or "plan" in name:
        return "Planning"
    return "General"

def collect_items(dest: Path) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for p in iter_md(dest):
        text = p.read_text(encoding="utf-8", errors="ignore")
        title = extract_title(text) or p.stem
        cat = categorize(p.name.lower(), title.lower())
        rel = p.relative_to(dest).as_posix()
        items.append((cat, title, rel))
    return items

def iter_md(dest: Path) -> list[Path]:
    files = [p for p in dest.rglob("*.md") if p.name != "INDEX.md"]
    return sorted(files, key=lambda x: x.name.lower())

def category_folder(cat: str) -> str:
    m = {
        "Papers": "papers",
        "ArXiv": "arxiv",
        "Proposals": "proposals",
        "Patents & IP": "patents_ip",
        "Legal & Compliance": "legal_compliance",
        "Financial": "financial",
        "Methods & Taxonomy": "methods_taxonomy",
        "Theory & Literature": "theory_literature",
        "Tutorials & Notebooks": "tutorials_notebooks",
        "Planning": "planning",
        "General": "general",
    }
    return m.get(cat, "general")

def organize(dest: Path) -> int:
    moved = 0
    for cat, title, rel in collect_items(dest):
        p = dest / rel
        folder = dest / category_folder(cat)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / p.name
        if p != target:
            p.rename(target)
            moved += 1
    build_index(dest)
    return moved

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="import_summaries",
        description="Import external summaries, enforce notation banner, and maintain index.",
    )
    parser.add_argument("paths", nargs="*", help="Source markdown files to import")
    parser.add_argument("--fix-existing", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--organize", action="store_true")
    parser.add_argument("--dest", help="Destination directory for summaries")
    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else Path(__file__).resolve().parents[2] / "docs" / "integration" / "summaries"
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        missing = [
            p
            for p in sorted(dest.glob("*.md"))
            if p.name != "INDEX.md"
            and is_technical_doc(p)
            and BANNER_PHRASE not in p.read_text(encoding="utf-8", errors="ignore")
        ]
        for p in missing:
            print(f"Missing banner: {p}")
        print(f"Total missing: {len(missing)}")
        return 0

    if args.relabel:
        build_index(dest)
        print(f"Index: {dest / 'INDEX.md'}")
        return 0

    if args.organize:
        moved = organize(dest)
        print(f"Organized: {moved}")
        print(f"Index: {dest / 'INDEX.md'}")
        return 0

    if args.fix_existing:
        changed = fix_existing(dest)
        print(f"Fixed: {changed}")
        print(f"Index: {dest / 'INDEX.md'}")
        return 0

    if not args.paths:
        print("No source paths provided.")
        return 1

    for arg in args.paths:
        src = Path(arg)
        if not src.exists() or not src.is_file():
            print(f"Skip: {arg}")
            continue
        out = dest / src.name
        write_provenance(out, src)
        print(f"Imported: {out}")
    build_index(dest)
    print(f"Index: {dest / 'INDEX.md'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
