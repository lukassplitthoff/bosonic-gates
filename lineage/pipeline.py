"""
Bosonic QC Lineage pipeline — single entry-point for all steps.

Steps (run individually or combined):
  verify    ORCID resolution + edge fact-checking    → verification_report.md
  discover  Co-author candidate discovery             → candidates_report.md
  monitor   New authors citing landmark papers        → new_authors_report.md
  fetch     Bibliometric data from OpenAlex           → contributions.json
  render    Generate interactive HTML tree            → lineage_tree.html

Usage examples:
  python pipeline.py                     # full run: verify → fetch → render
  python pipeline.py --all               # same as above
  python pipeline.py --verify --render   # verify then re-render
  python pipeline.py --discover          # co-author candidate discovery
  python pipeline.py --monitor           # new authors citing landmark papers
  python pipeline.py --verify --apply    # verify + write ORCIDs to mentorship.yaml
  python pipeline.py --fetch --render    # refresh bibliometrics and re-render
  python pipeline.py --render            # re-render without fetching anything new

Flags:
  --verify    Run ORCID verifier
  --discover  Run co-author discovery
  --monitor   Run arXiv/OpenAlex citation monitor
  --fetch     Fetch/update bibliometrics
  --render    Regenerate HTML
  --all       Shorthand for --verify --fetch --render (no --discover/--monitor, which are slow)
  --apply     With --verify: write ORCID IDs and verified flags to mentorship.yaml
  --top N     With --discover/--monitor: show top N candidates (default 20)
  --days N    With --monitor: look-back window in days (default 90)
"""

import argparse
import sys
import time
from pathlib import Path


def _banner(title: str) -> None:
    width = 60
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def _check_yaml() -> bool:
    if not Path("mentorship.yaml").exists():
        print("ERROR: mentorship.yaml not found. Run from the lineage/ directory.")
        return False
    return True


def run_verify(apply: bool) -> int:
    _banner("Step 1/4 — ORCID verification")
    import verify
    try:
        verify.main(apply=apply)
        return 0
    except Exception as e:
        print(f"verify failed: {e}")
        return 1


def run_discover(top_n: int) -> int:
    _banner("Co-author discovery")
    import discover
    try:
        discover.main(top_n=top_n)
        return 0
    except Exception as e:
        print(f"discover failed: {e}")
        return 1


def run_monitor(top_n: int, days: int) -> int:
    _banner("Citation monitor (new authors citing landmark papers)")
    import monitor
    try:
        monitor.main(days=days, top_n=top_n)
        return 0
    except Exception as e:
        print(f"monitor failed: {e}")
        return 1


def run_fetch() -> int:
    _banner("Step 2/4 — Bibliometrics (OpenAlex)")
    import build_contributions
    try:
        build_contributions.main()
        return 0
    except Exception as e:
        print(f"fetch failed: {e}")
        return 1


def run_render() -> int:
    _banner("Step 3/4 — HTML render")
    import render_tree
    try:
        render_tree.build()
        return 0
    except Exception as e:
        print(f"render failed: {e}")
        return 1


def open_browser() -> None:
    import subprocess, platform
    html = Path("lineage_tree.html").resolve()
    if not html.exists():
        return
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "", str(html)], shell=False)
        elif system == "Darwin":
            subprocess.Popen(["open", str(html)])
        else:
            subprocess.Popen(["xdg-open", str(html)])
    except Exception:
        print(f"  Open manually: {html}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verify",   action="store_true", help="Run ORCID verifier")
    parser.add_argument("--discover", action="store_true", help="Run co-author discovery")
    parser.add_argument("--monitor",  action="store_true", help="Run citation monitor (new authors)")
    parser.add_argument("--fetch",    action="store_true", help="Fetch/update bibliometrics")
    parser.add_argument("--render",   action="store_true", help="Regenerate lineage_tree.html")
    parser.add_argument("--all",      action="store_true", help="Verify + fetch + render")
    parser.add_argument("--apply",    action="store_true", help="Write ORCID updates to mentorship.yaml")
    parser.add_argument("--top",      type=int, default=20, metavar="N", help="Candidates to show with --discover/--monitor")
    parser.add_argument("--days",     type=int, default=90, metavar="N", help="Look-back window in days for --monitor")
    parser.add_argument("--open",     action="store_true", help="Open lineage_tree.html in browser when done")
    args = parser.parse_args()

    # Default: --all if no step flags given
    if not any([args.verify, args.discover, args.monitor, args.fetch, args.render, args.all]):
        args.all = True

    if args.all:
        args.verify = args.fetch = args.render = True

    if not _check_yaml():
        sys.exit(1)

    t0 = time.time()
    errors = []

    if args.verify:
        rc = run_verify(apply=args.apply)
        if rc:
            errors.append("verify")

    if args.discover:
        rc = run_discover(top_n=args.top)
        if rc:
            errors.append("discover")

    if args.monitor:
        rc = run_monitor(top_n=args.top, days=args.days)
        if rc:
            errors.append("monitor")

    if args.fetch:
        rc = run_fetch()
        if rc:
            errors.append("fetch")

    if args.render:
        rc = run_render()
        if rc:
            errors.append("render")

    elapsed = time.time() - t0
    _banner(f"Done in {elapsed:.1f}s")

    # Show output files
    for fname in ("lineage_tree.html", "verification_report.md", "candidates_report.md", "new_authors_report.md"):
        p = Path(fname)
        if p.exists():
            kb = p.stat().st_size // 1024
            print(f"  OK  {fname}  ({kb} KB)")

    if errors:
        print(f"\n  WARNING: Errors in: {', '.join(errors)}")
        sys.exit(1)

    if args.open or (args.render and not errors):
        open_browser()


if __name__ == "__main__":
    main()

