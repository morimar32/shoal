"""Fetch Wikipedia articles as markdown and save to docs/.

Uses two Wikipedia APIs:
1. Parse API for section structure (titles + levels)
2. TextExtracts API for clean plaintext per section
Then stitches them into markdown with proper # headers.
"""

import re
import time
from pathlib import Path

import requests

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "ShoalCorpusBuilder/1.0 (https://github.com/shoal; educational project)",
}

# Articles to fetch: (wiki_title, output_filename)
ARTICLES = [
    ("Chess", "chess"),
    ("Monarchy_of_the_United_Kingdom", "monarchy_of_the_united_kingdom"),
    ("Mercury_(planet)", "mercury_planet"),
    ("Mercury_(element)", "mercury_element"),
    ("Crane_(machine)", "crane_machine"),
    ("Crane_(bird)", "crane_bird"),
    ("Apple_Inc.", "apple_inc"),
    ("Apple", "apple_fruit"),
    ("Python_(programming_language)", "python_programming_language"),
    ("Pythonidae", "python_snake"),
]


def _get_sections(title: str) -> list[dict]:
    """Get section structure from parse API."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections",
        "format": "json",
        "formatversion": "2",
    }
    resp = requests.get(_API, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"API error: {data['error']}")
    return data["parse"]["sections"], data["parse"]["title"]


def _get_section_text(title: str, section_index: int) -> str:
    """Get clean plaintext for a specific section using TextExtracts."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exsectionformat": "plain",
        "explaintext": "1",
        "exlimit": "1",
        "format": "json",
        "formatversion": "2",
    }
    if section_index > 0:
        params["exsectionformat"] = "plain"
        params["exchars"] = "100000"
        # Use the parse API with section parameter for non-intro sections
        parse_params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "section": str(section_index),
            "format": "json",
            "formatversion": "2",
        }
        resp = requests.get(_API, params=parse_params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return ""
        wikitext = data["parse"]["wikitext"]
        return _clean_wikitext(wikitext)
    else:
        # For intro (section 0), use TextExtracts - it handles it cleanly
        params["exintro"] = "1"
        resp = requests.get(_API, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages = data["query"]["pages"]
        if not pages:
            return ""
        return pages[0].get("extract", "")


def _clean_wikitext(wikitext: str) -> str:
    """Convert wikitext to clean plaintext, stripping all markup."""
    text = wikitext

    # Remove header line (we add our own)
    text = re.sub(r"^=+\s*.*?\s*=+\s*\n?", "", text)

    # Remove nested templates (multi-pass for nesting)
    for _ in range(10):
        new = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if new == text:
            break
        text = new

    # Remove leftover {{ or }}
    text = text.replace("{{", "").replace("}}", "")

    # Remove ref tags (both self-closing and paired)
    text = re.sub(r"<ref[^>]*/\s*>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)

    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove files/images
    text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Remove category links
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Convert wiki links: [[target|display]] -> display, [[target]] -> target
    text = re.sub(r"\[\[[^\]]*?\|([^\]]+?)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+?)\]\]", r"\1", text)

    # Convert external links
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # Remove bold/italic wiki markup
    text = re.sub(r"'{2,5}", "", text)

    # Remove table markup
    text = re.sub(r"^\{\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|\}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^!.*$", "", text, flags=re.MULTILINE)

    # Remove list markers but keep text
    text = re.sub(r"^\*+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^;+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^:+\s*", "", text, flags=re.MULTILINE)

    # Remove __NOTOC__ etc
    text = re.sub(r"__[A-Z]+__", "", text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fetch_article_markdown(title: str) -> str:
    """Fetch a Wikipedia article and return as markdown with section headers."""
    sections_data, display_title = _get_sections(title)

    parts = [f"# {display_title}\n"]

    # Get intro (section 0)
    intro = _get_section_text(title, 0)
    if intro.strip():
        parts.append(intro.strip())
        parts.append("")

    # Get each section
    for sec in sections_data:
        level = int(sec["level"]) + 1  # wiki level 2 -> markdown ##, etc.
        sec_title = sec["line"]
        sec_index = int(sec["index"])

        # Cap at h6
        level = min(level, 6)

        header = f"{'#' * level} {sec_title}"

        text = _get_section_text(title, sec_index)
        if text.strip():
            parts.append(header)
            parts.append("")
            parts.append(text.strip())
            parts.append("")

    result = "\n".join(parts)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"


def main():
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    for wiki_title, filename in ARTICLES:
        outpath = docs_dir / f"{filename}.md"
        print(f"Fetching {wiki_title}...", end=" ", flush=True)
        try:
            md = fetch_article_markdown(wiki_title)
            outpath.write_text(md, encoding="utf-8")
            words = len(md.split())
            print(f"OK ({words} words, {len(md)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
        # Be polite to Wikipedia — 1 req per section, so pause between articles
        time.sleep(2)

    print("\nDone!")


if __name__ == "__main__":
    main()
