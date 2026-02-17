#!/usr/bin/env python3
"""Download test corpus for shoal.

Fetches Wikipedia articles as markdown and Project Gutenberg books as plain text.
Articles and books are defined in README.md section 11.

Usage:
    python scripts/download_corpus.py
"""

import os
import re
import time

import html2text
import requests

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

# Wikipedia articles from README section 11
WIKIPEDIA_ARTICLES = [
    # Archipelago 0: Natural sciences and taxonomy
    "Photosynthesis",
    "Tyrannosaurus",
    "Corvidae",
    "Penicillin",
    "Mitochondrion",
    "Great Barrier Reef",
    # Archipelago 1: Physical world and materiality
    "Earthquake",
    "Jupiter",
    "Granite",
    "Atmospheric river",
    "Volcanic eruption",
    # Archipelago 2: Abstract processes and systems
    "Fourier transform",
    "Game theory",
    "Algorithm",
    "Natural selection",
    "Supply and demand",
    "Thermodynamics",
    # Archipelago 3: Social order and assessment
    "French Revolution",
    "United Nations",
    "Supreme Court of the United States",
    "Olympic Games",
    "Renaissance",
    # Cross-domain edge cases
    "Coffee",
    "Silk Road",
    "DNA",
    "Climate change",
    # Phase 2 stress test: "Yamamoto" disambiguation corpus
    "Isoroku Yamamoto",
    "Yamamoto (crater)",
    "Ichi the Killer (manga)",
    "Lunar craters",
    "Manga",
]

# Project Gutenberg books
GUTENBERG_BOOKS = [
    {"id": 76, "filename": "huckleberry_finn.txt", "title": "Adventures of Huckleberry Finn"},
    {"id": 2009, "filename": "origin_of_species.txt", "title": "On the Origin of Species"},
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ShoalCorpusDownloader/1.0 (educational/research)"})


def download_wikipedia_article(title: str) -> str:
    """Download a Wikipedia article via MediaWiki API and convert to markdown."""
    resp = SESSION.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "redirects": "true",
            "disabletoc": "true",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(f"API error: {data['error'].get('info', data['error'])}")

    html_content = data["parse"]["text"]["*"]
    display_title = data["parse"]["title"]

    # Remove reference sections, navboxes, and other non-content elements
    # These are common Wikipedia HTML patterns that don't add value as corpus text
    html_content = re.sub(
        r'<div class="reflist.*?</div>', "", html_content, flags=re.DOTALL
    )
    html_content = re.sub(
        r'<div class="navbox.*?</div>', "", html_content, flags=re.DOTALL
    )
    html_content = re.sub(r"<sup.*?</sup>", "", html_content, flags=re.DOTALL)

    # Configure html2text for clean markdown output
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_images = True
    h.ignore_emphasis = False
    h.unicode_snob = True
    h.skip_internal_links = True
    h.inline_links = False
    h.ignore_links = True

    markdown = h.handle(html_content)

    # Clean up excessive blank lines
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

    # Add title as H1
    return f"# {display_title}\n\n{markdown.strip()}\n"


def download_gutenberg_book(book_id: int) -> str:
    """Download a Project Gutenberg book and strip boilerplate."""
    # Try the standard cache URL first
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    resp = SESSION.get(url)
    resp.raise_for_status()
    text = resp.text

    # Strip Project Gutenberg header (everything before the START marker)
    start_match = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE,
    )
    if start_match:
        text = text[start_match.end() :]

    # Strip Project Gutenberg footer (everything after the END marker)
    end_match = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE,
    )
    if end_match:
        text = text[: end_match.start()]

    return text.strip() + "\n"


def title_to_filename(title: str) -> str:
    """Convert a Wikipedia article title to a filename."""
    return title.lower().replace(" ", "_") + ".md"


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Download Wikipedia articles
    total = len(WIKIPEDIA_ARTICLES)
    print(f"Downloading {total} Wikipedia articles to {DOCS_DIR}/")
    print("-" * 60)

    wiki_ok, wiki_fail = 0, 0
    for i, title in enumerate(WIKIPEDIA_ARTICLES, 1):
        filename = title_to_filename(title)
        filepath = os.path.join(DOCS_DIR, filename)

        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  [{i:2d}/{total}] {title} — exists ({size:,} bytes), skipping")
            wiki_ok += 1
            continue

        print(f"  [{i:2d}/{total}] {title}...", end=" ", flush=True)
        try:
            content = download_wikipedia_article(title)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"OK ({len(content):,} chars)")
            wiki_ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            wiki_fail += 1

        # Be polite to Wikipedia — 1 second between requests
        if i < total:
            time.sleep(1)

    # Download Gutenberg books
    print(f"\nDownloading {len(GUTENBERG_BOOKS)} Project Gutenberg books...")
    print("-" * 60)

    gut_ok, gut_fail = 0, 0
    for book in GUTENBERG_BOOKS:
        filepath = os.path.join(DOCS_DIR, book["filename"])

        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  {book['title']} — exists ({size:,} bytes), skipping")
            gut_ok += 1
            continue

        print(f"  {book['title']} (Gutenberg #{book['id']})...", end=" ", flush=True)
        try:
            content = download_gutenberg_book(book["id"])
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"OK ({len(content):,} chars)")
            gut_ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            gut_fail += 1

    # Summary
    print("\n" + "=" * 60)
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    total_size = sum(os.path.getsize(os.path.join(DOCS_DIR, f)) for f in files)
    md_count = sum(1 for f in files if f.endswith(".md"))
    txt_count = sum(1 for f in files if f.endswith(".txt"))

    print(f"Summary: {len(files)} files ({md_count} .md, {txt_count} .txt)")
    print(f"  Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    print(f"  Wikipedia: {wiki_ok} OK, {wiki_fail} failed")
    print(f"  Gutenberg: {gut_ok} OK, {gut_fail} failed")

    if wiki_fail or gut_fail:
        print("\nSome downloads failed. Re-run the script to retry (existing files are skipped).")


if __name__ == "__main__":
    main()
