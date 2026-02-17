"""Command-line interface for shoal."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ._engine import Engine
from ._models import DocFormat, SearchResponse

# ── ANSI color helpers ──

_NO_COLOR = os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty()

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"

# Hierarchy colors
_ARCH = "\033[35m"        # magenta — archipelago (broadest)
_ARCH_BOLD = "\033[1;35m"
_ISLAND = "\033[36m"      # cyan — island (mid-level)
_ISLAND_BOLD = "\033[1;36m"
_REEF = "\033[32m"        # green — reef (finest)
_REEF_BOLD = "\033[1;32m"

# Other colors
_TITLE = "\033[1;33m"     # bold yellow — document titles
_SCORE = "\033[1;37m"     # bold white — scores
_QUERY = "\033[1;97m"     # bright bold white — the query text
_NUM = "\033[1;34m"       # bold blue — rank numbers
_GOOD = "\033[32m"        # green — good values
_MED = "\033[33m"         # yellow — medium values
_LOW = "\033[31m"         # red — low values
_LABEL = "\033[90m"       # dark gray — labels
_PREVIEW = "\033[37m"     # light gray — text previews
_BAR = "\033[90m"         # dark gray — bars/lines
_Z_CHUNK = "\033[1;32m"   # bold green — chunk z-scores
_Z_QUERY = "\033[1;36m"   # bold cyan — query z-scores
_SECTION = "\033[36m"     # cyan — section paths
_WARN = "\033[1;33m"      # bold yellow — warnings

# Unicode characters
_HLINE = "\u2500"       # ─
_BLOCK_FULL = "\u2588"  # █
_BLOCK_LIGHT = "\u2591" # ░
_DIAMOND = "\u25c6"     # ◆
_ARROW = "\u2192"       # →
_TREE_BRANCH = "\u251c" # ├
_TREE_LAST = "\u2514"   # └
_TREE_PIPE = "\u2502"   # │

_ARCH_NAMES = [
    "natural sciences & taxonomy",
    "physical world & materiality",
    "abstract processes & systems",
    "social order & assessment",
    "specialized activities & practices",
]

_ARCH_SYMBOLS = ["N", "P", "A", "S", "Sp"]


def _c(code: str, text: str) -> str:
    """Apply ANSI color code to text, respecting NO_COLOR."""
    if _NO_COLOR:
        return text
    return f"{code}{text}{_RESET}"


def _conf_color(confidence: float) -> str:
    """Color-code a confidence value."""
    if confidence >= 3.0:
        return _GOOD
    elif confidence >= 1.0:
        return _MED
    return _LOW


def _cov_color(coverage: float) -> str:
    """Color-code a coverage value."""
    if coverage >= 0.8:
        return _GOOD
    elif coverage >= 0.5:
        return _MED
    return _LOW


def _score_bar(score: float, max_score: float, width: int = 20) -> str:
    """Render a horizontal bar showing relative score."""
    if max_score <= 0:
        filled = 0
    else:
        filled = min(width, round(width * score / max_score))
    bar = _BLOCK_FULL * filled + _BLOCK_LIGHT * (width - filled)
    return _c(_BAR, bar)


def _dominant_arch(arch_scores: list[float]) -> tuple[int, str]:
    """Return the index and name of the dominant archipelago."""
    if not arch_scores:
        return 0, _ARCH_NAMES[0]
    idx = max(range(len(arch_scores)), key=lambda i: arch_scores[i])
    return idx, _ARCH_NAMES[idx]


def _arch_sparkline(arch_scores: list[float]) -> str:
    """Render a compact sparkline of arch scores with labels."""
    if not arch_scores or len(arch_scores) < 5:
        return ""
    total = sum(arch_scores) or 1.0
    parts = []
    for i, (sym, score) in enumerate(zip(_ARCH_SYMBOLS, arch_scores)):
        pct = score / total * 100
        if pct >= 30:
            parts.append(_c(_ARCH_BOLD, f"{sym}:{pct:.0f}%"))
        elif pct >= 15:
            parts.append(_c(_ARCH, f"{sym}:{pct:.0f}%"))
        else:
            parts.append(_c(_LABEL, f"{sym}:{pct:.0f}%"))
    return " ".join(parts)


def _format_section_path(path: list[str]) -> str:
    """Format a section path as 'A > B > C'."""
    if not path:
        return ""
    return " > ".join(path)


# ── Commands ──

def _detect_format(path: Path) -> DocFormat:
    """Detect document format from file extension."""
    if path.suffix.lower() == ".md":
        return DocFormat.markdown
    return DocFormat.plaintext


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a single file."""
    path = Path(args.path)
    if not path.is_file():
        print(f"Error: {path} is not a file", file=sys.stderr)
        sys.exit(1)

    fmt = _detect_format(path)
    title = args.title or path.stem.replace("_", " ").title()
    content = path.read_text(encoding="utf-8")
    tags = args.tags.split(",") if args.tags else []

    with Engine(db_path=args.db) as engine:
        result = engine.ingest(
            title=title,
            content=content,
            format=fmt,
            tags=tags,
            source_path=str(path),
        )
        print(f"Ingested: {result.title} ({result.n_sections} sections, {result.n_chunks} chunks, id={result.id})")


def _cmd_ingest_dir(args: argparse.Namespace) -> None:
    """Ingest all .md and .txt files in a directory."""
    dirpath = Path(args.dir)
    if not dirpath.is_dir():
        print(f"Error: {dirpath} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        p for p in dirpath.iterdir()
        if p.is_file() and p.suffix.lower() in (".md", ".txt")
    )
    if not files:
        print(f"No .md or .txt files found in {dirpath}")
        return

    tags = args.tags.split(",") if args.tags else []

    with Engine(db_path=args.db) as engine:
        for path in files:
            fmt = _detect_format(path)
            title = path.stem.replace("_", " ").title()
            content = path.read_text(encoding="utf-8")
            result = engine.ingest(
                title=title,
                content=content,
                format=fmt,
                tags=tags,
                source_path=str(path),
            )
            print(f"  {result.title}: {result.n_sections} sections, {result.n_chunks} chunks (id={result.id})")
        print(f"\nIngested {len(files)} files.")


def _cmd_search(args: argparse.Namespace) -> None:
    """Search for relevant chunks."""
    tags = args.tags.split(",") if args.tags else None

    with Engine(db_path=args.db) as engine:
        response = engine.search(
            args.query,
            top_k=args.top_k,
            tags=tags,
            min_confidence=args.min_confidence,
            include_scores=args.scores,
        )
        _print_search_results(response, args.query, args.scores)


def _print_search_results(response: SearchResponse, query: str, show_scores: bool) -> None:
    """Pretty-print search results with color-coded hierarchy."""
    qi = response.query_info

    # ── Query header ──
    print()
    print(f"  {_c(_LABEL, 'Query:')} {_c(_QUERY, query)}")
    print(f"  {_c(_LABEL, 'confidence')} {_c(_conf_color(qi.confidence), f'{qi.confidence:.2f}')}  "
          f"{_c(_LABEL, 'coverage')} {_c(_cov_color(qi.coverage), f'{qi.coverage:.2f}')}  "
          f"{_c(_LABEL, 'words')} {_c(_DIM, f'{qi.matched_words}/{qi.total_words}')}")

    # ── Low-confidence warning ──
    if qi.confidence < 0.1:
        explain_hint = f'  Run `shoal explain "{query}"` for per-word diagnostics.'
        print()
        print(f"  {_c(_WARN, f'{_DIAMOND} WARNING: Very low query confidence ({qi.confidence:.4f}).')}")
        print(f"  {_c(_WARN, '  The scorer has almost no semantic signal for this query.')}")
        print(f"  {_c(_WARN, '  Results are driven by incidental reef overlap, not topical matching.')}")
        print(f"  {_c(_WARN, explain_hint)}")
    elif qi.confidence < 0.5:
        explain_hint = f'  Run `shoal explain "{query}"` for diagnostics.'
        print()
        print(f"  {_c(_WARN, f'{_DIAMOND} Low query confidence ({qi.confidence:.2f}). Results may be inaccurate.')}")
        print(f"  {_c(_WARN, explain_hint)}")
    print()

    # ── Query reef profile ──
    print(f"  {_c(_LABEL, 'Query reefs:')}")
    for i, reef in enumerate(qi.top_reefs[:5]):
        z = reef["z_score"]
        name = reef["reef_name"]
        marker = f" {_DIAMOND}" if i == 0 else "  "
        print(f"  {marker} {_c(_REEF, name)}{_c(_LABEL, ':')} {_c(_SCORE, f'{z:.2f}')}")

    if qi.top_islands:
        print(f"  {_c(_LABEL, 'Query islands:')}")
        for isl in qi.top_islands[:3]:
            agg_z = isl["aggregate_z"]
            isl_name = isl["island_name"]
            print(f"    {_c(_ISLAND, isl_name)}{_c(_LABEL, ':')} "
                  f"{_c(_SCORE, f'{agg_z:.2f}')}")

    print()
    hline = _HLINE * 72
    print(f"  {_c(_BAR, hline)}")

    if not response.results:
        print(f"\n  {_c(_LABEL, 'No results found.')}")
        print()
        return

    max_score = response.results[0].match_score if response.results else 1.0

    for i, r in enumerate(response.results, 1):
        # ── Rank + document title ──
        print()
        rank_str = _c(_NUM, f" {i:2d}.")
        title_str = _c(_TITLE, r.document_title)
        print(f"{rank_str} {title_str}")

        # ── Section path ──
        if r.section_path:
            path_str = _format_section_path(r.section_path)
            print(f"     {_c(_LABEL, '[')} {_c(_SECTION, path_str)} {_c(_LABEL, ']')}")

        # ── Score bar + match score ──
        bar = _score_bar(r.match_score, max_score)
        score_str = _c(_SCORE, f"{r.match_score:.1f}")
        shared_str = _c(_SCORE, str(r.n_shared_reefs))
        print(f"     {bar} {_c(_LABEL, 'score')} {score_str}  "
              f"{_c(_LABEL, 'shared')} {shared_str}")

        # ── Confidence + coverage ──
        conf_str = _c(_conf_color(r.confidence), f"{r.confidence:.2f}")
        cov_str = _c(_cov_color(r.coverage), f"{r.coverage:.2f}")
        print(f"     {_c(_LABEL, 'conf')} {conf_str}  {_c(_LABEL, 'cov')} {cov_str}  "
              f"{_c(_LABEL, 'chunk')} {_c(_DIM, f'#{r.chunk_index}')}")

        # ── Reef (top reef of chunk) ──
        print(f"     {_c(_LABEL, 'reef')}  {_c(_REEF_BOLD, r.top_reef_name)}")

        # ── Archipelago sparkline ──
        if r.arch_scores:
            arch_idx, arch_name = _dominant_arch(r.arch_scores)
            sparkline = _arch_sparkline(r.arch_scores)
            print(f"     {_c(_LABEL, 'arch')}  {_c(_ARCH_BOLD, arch_name)}  "
                  f"{_c(_LABEL, '[')} {sparkline} {_c(_LABEL, ']')}")

        # ── Shared reef details (with --scores) ──
        if show_scores and r.shared_reefs:
            print(f"     {_c(_LABEL, 'shared reefs:')}")
            for sr in r.shared_reefs:
                contrib = sr.chunk_z * sr.query_z
                print(f"       {_c(_REEF, sr.reef_name)}")
                print(f"         {_c(_LABEL, 'chunk_z')} {_c(_Z_CHUNK, f'{sr.chunk_z:.2f}')}  "
                      f"{_c(_LABEL, 'query_z')} {_c(_Z_QUERY, f'{sr.query_z:.2f}')}  "
                      f"{_c(_LABEL, _ARROW)} {_c(_SCORE, f'{contrib:.1f}')}")

        # ── Full text ──
        print(f"     {_c(_BAR, _HLINE * 40)}")
        for line in r.text.splitlines():
            print(f"     {_c(_PREVIEW, line)}")
        print(f"     {_c(_BAR, _HLINE * 40)}")

    print()
    hline = _HLINE * 72
    print(f"  {_c(_BAR, hline)}")
    print(f"  {_c(_LABEL, f'{len(response.results)} results')}")
    print()


def _cmd_sections(args: argparse.Namespace) -> None:
    """Display the section tree for a document."""
    with Engine(db_path=args.db) as engine:
        doc = engine.storage.get_document(args.doc_id)
        if doc is None:
            print(f"Error: document {args.doc_id} not found", file=sys.stderr)
            sys.exit(1)

        sections = engine.storage.get_sections_for_document(args.doc_id)
        if not sections:
            print(f"No sections found for document '{doc['title']}' (id={args.doc_id})")
            return

        print(f"\n  {_c(_TITLE, doc['title'])} {_c(_LABEL, f'(id={args.doc_id})')}")
        print()

        # Build tree structure for display
        # Group sections by parent_id
        children: dict[int | None, list[dict]] = {}
        for s in sections:
            parent = s["parent_id"]
            children.setdefault(parent, []).append(s)

        def _print_tree(parent_id: int | None, prefix: str = "  ") -> None:
            kids = children.get(parent_id, [])
            for i, s in enumerate(kids):
                is_last = (i == len(kids) - 1)
                connector = _TREE_LAST if is_last else _TREE_BRANCH
                chunk_info = _c(_LABEL, f"({s['n_chunks']} chunks)")
                depth_info = _c(_DIM, f"d{s['depth']}")
                print(f"{prefix}{_c(_BAR, connector + _HLINE)} {_c(_SECTION, s['title'])} "
                      f"{chunk_info} {depth_info}")
                child_prefix = prefix + ("   " if is_last else f"{_c(_BAR, _TREE_PIPE)}  ")
                _print_tree(s["id"], child_prefix)

        _print_tree(None)
        print()


def _cmd_explain(args: argparse.Namespace) -> None:
    """Explain how a query is understood by the scorer."""
    with Engine(db_path=args.db) as engine:
        info = engine.explain_query(args.query)

    print()
    print(f"  {_c(_LABEL, 'Query:')} {_c(_QUERY, info['query'])}")
    conf_val = f"{info['confidence']:.4f}"
    cov_val = f"{info['coverage']:.2f}"
    words_val = f"{info['matched_words']}/{info['total_words']}"
    print(f"  {_c(_LABEL, 'confidence')} {_c(_conf_color(info['confidence']), conf_val)}  "
          f"{_c(_LABEL, 'coverage')} {_c(_cov_color(info['coverage']), cov_val)}  "
          f"{_c(_LABEL, 'words')} {_c(_DIM, words_val)}")
    print()

    # ── Warnings ──
    if info["warnings"]:
        for warning in info["warnings"]:
            print(f"  {_c(_WARN, f'{_DIAMOND} {warning}')}")
        print()

    # ── Per-word breakdown ──
    print(f"  {_c(_LABEL, 'Per-word breakdown:')}")
    print()
    for wd in info["word_details"]:
        conf = wd["confidence"]
        conf_str = _c(_conf_color(conf), f"{conf:.4f}")

        # Signal strength indicator
        if conf >= 0.5:
            indicator = _c(_GOOD, "STRONG")
        elif conf >= 0.1:
            indicator = _c(_MED, "WEAK  ")
        else:
            indicator = _c(_LOW, "NONE  ")

        print(f"    {indicator}  {_c(_BOLD, wd['word']):20s}  "
              f"{_c(_LABEL, 'conf')} {conf_str}")
        for reef in wd["top_reefs"][:2]:
            z = reef["z_score"]
            print(f"             {_c(_REEF, reef['reef_name'])}{_c(_LABEL, ':')} "
                  f"{_c(_SCORE, f'{z:.2f}')}")
        print()

    # ── Combined query reef profile ──
    hline = _HLINE * 60
    print(f"  {_c(_BAR, hline)}")
    print(f"  {_c(_LABEL, 'Combined query reef profile:')}")
    for reef in info["top_reefs"][:7]:
        z = reef["z_score"]
        n_words = reef["n_contributing_words"]
        word_label = "words" if n_words != 1 else "word"
        word_count = f"({n_words} {word_label})"
        print(f"    {_c(_REEF, reef['reef_name'])}{_c(_LABEL, ':')} "
              f"{_c(_SCORE, f'z={z:.2f}')}  "
              f"{_c(_DIM, word_count)}")

    if info["top_islands"]:
        print(f"  {_c(_LABEL, 'Combined query islands:')}")
        for isl in info["top_islands"][:3]:
            agg_z = f"{isl['aggregate_z']:.2f}"
            print(f"    {_c(_ISLAND, isl['island_name'])}{_c(_LABEL, ':')} "
                  f"{_c(_SCORE, agg_z)}")

    print()


def _cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    with Engine(db_path=args.db) as engine:
        s = engine.status()
        print(f"Status: {s['status']}")
        print(f"Documents: {s['n_documents']}")
        print(f"Sections: {s['n_sections']}")
        print(f"Chunks: {s['n_chunks']}")
        print(f"Lagoon version: {s['lagoon_version']}")
        print(f"DB path: {s['db_path']}")


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the shoal CLI."""
    parser = argparse.ArgumentParser(
        prog="shoal",
        description="Shoal: retrieval engine built on lagoon",
    )
    parser.add_argument("--db", default="shoal.db", help="Database path (default: shoal.db)")

    subparsers = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a single file")
    p_ingest.add_argument("path", help="Path to file (.md or .txt)")
    p_ingest.add_argument("--title", help="Document title (default: filename)")
    p_ingest.add_argument("--tags", help="Comma-separated tags")

    # ingest-dir
    p_ingest_dir = subparsers.add_parser("ingest-dir", help="Ingest all files in a directory")
    p_ingest_dir.add_argument("dir", help="Directory containing .md and .txt files")
    p_ingest_dir.add_argument("--tags", help="Comma-separated tags")

    # search
    p_search = subparsers.add_parser("search", help="Search for relevant chunks")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    p_search.add_argument("--tags", help="Filter by comma-separated tags")
    p_search.add_argument("--min-confidence", type=float, help="Minimum chunk confidence")
    p_search.add_argument("--scores", action="store_true", help="Show shared reef details")

    # explain
    p_explain = subparsers.add_parser("explain", help="Explain how a query is scored")
    p_explain.add_argument("query", help="Query text to analyze")

    # sections
    p_sections = subparsers.add_parser("sections", help="Show section tree for a document")
    p_sections.add_argument("doc_id", type=int, help="Document ID")

    # status
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "ingest": _cmd_ingest,
        "ingest-dir": _cmd_ingest_dir,
        "search": _cmd_search,
        "explain": _cmd_explain,
        "sections": _cmd_sections,
        "status": _cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
