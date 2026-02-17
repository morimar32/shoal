#!/usr/bin/env python3
"""Fetch Fandom wiki articles as markdown and save to docs/.

Uses the same MediaWiki API as Wikipedia, with per-wiki base URLs.
Reuses html2text for HTML-to-markdown conversion.

Usage:
    python scripts/fetch_fandom.py
"""

import os
import re
import time

import html2text
import requests

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

# (base_api_url, page_title, output_filename)
FANDOM_ARTICLES = [
    # StarCraft wiki
    ("https://starcraft.fandom.com/api.php", "Yamato_cannon", "yamato_cannon"),
    ("https://starcraft.fandom.com/api.php", "Battlecruiser_(StarCraft_II)", "battlecruiser_starcraft_ii"),
    ("https://starcraft.fandom.com/api.php", "Terran", "terran"),
    ("https://starcraft.fandom.com/api.php", "Starport", "starport"),
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ShoalCorpusBuilder/1.0 (educational/research)"})


def download_fandom_article(api_url: str, page_title: str) -> str:
    """Download a Fandom wiki article via MediaWiki API and convert to markdown."""
    resp = SESSION.get(
        api_url,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
            "redirects": "true",
            "disabletoc": "true",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(f"API error: {data['error'].get('info', data['error'])}")

    html_content = data["parse"]["text"]["*"]
    display_title = data["parse"]["title"]

    # Remove Fandom-specific cruft: navboxes, reference lists, infoboxes, aside elements
    html_content = re.sub(r'<div class="reflist.*?</div>', "", html_content, flags=re.DOTALL)
    html_content = re.sub(r'<div class="navbox.*?</div>', "", html_content, flags=re.DOTALL)
    html_content = re.sub(r'<aside[^>]*>.*?</aside>', "", html_content, flags=re.DOTALL)
    html_content = re.sub(r'<sup.*?</sup>', "", html_content, flags=re.DOTALL)
    # Remove Fandom "References" footer junk
    html_content = re.sub(r'<ol class="references">.*?</ol>', "", html_content, flags=re.DOTALL)

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

    return f"# {display_title}\n\n{markdown.strip()}\n"


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    total = len(FANDOM_ARTICLES)
    print(f"Downloading {total} Fandom wiki articles to {DOCS_DIR}/")
    print("-" * 60)

    ok, fail = 0, 0
    for i, (api_url, page_title, filename) in enumerate(FANDOM_ARTICLES, 1):
        filepath = os.path.join(DOCS_DIR, f"{filename}.md")

        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  [{i:2d}/{total}] {page_title} — exists ({size:,} bytes), skipping")
            ok += 1
            continue

        wiki_name = api_url.split("//")[1].split(".")[0]
        print(f"  [{i:2d}/{total}] {wiki_name}:{page_title}...", end=" ", flush=True)
        try:
            content = download_fandom_article(api_url, page_title)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"OK ({len(content):,} chars)")
            ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            fail += 1

        if i < total:
            time.sleep(1)

    print(f"\nDone: {ok} OK, {fail} failed")
    if fail:
        print("Re-run the script to retry (existing files are skipped).")


if __name__ == "__main__":
    main()
