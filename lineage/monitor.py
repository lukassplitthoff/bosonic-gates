"""
Monitor recent papers citing landmark bosonic QC works and surface new authors.

Uses OpenAlex "cited-by" queries filtered to the last N days (default 90).
For each citing paper, collects authors not already in mentorship.yaml and
scores them by paper count + citation count.

Landmark papers hardcoded below — add more as the field publishes new milestones.

Outputs:
  new_authors_report.md   ranked list of new authors with paper context
  cache/monitor/          raw API responses (cached)

Usage:
  python monitor.py                 # last 90 days
  python monitor.py --days 180      # last 180 days
  python monitor.py --top 30        # show top 30 new authors
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

import yaml

OPENALEX_BASE = "https://api.openalex.org"
CACHE_DIR     = os.path.join("cache", "monitor")
EMAIL         = "lukas.splitthoff@gmail.com"
SLEEP         = 0.4

# Landmark papers by OpenAlex work ID.
# These are the most-cited bosonic QC papers — every new group entering the field
# cites at least one of them.
LANDMARK_PAPERS = [
    # --- Cat qubit / bosonic encoding ---
    ("W2068006129", "Mirrahimi 2014 -- Dynamically protected cat-qubits"),
    ("W2497055544", "Ofek 2016 Nature -- Extending lifetime beyond break-even"),
    ("W1969695012", "Leghtas 2015 Science -- Confining light to quantum manifold"),
    ("W3048796215", "Grimm 2020 -- Stabilization and operation of a Kerr-cat qubit"),
    ("W3082857413", "Campagne-Ibarcq 2020 Nature -- GKP/ECD error correction"),
    # --- Circuit QED / transmon foundations ---
    ("W1963734567", "Koch 2007 PRA -- Transmon qubit design"),
    ("W2060904747", "Blais 2004 PRA -- Cavity QED for superconducting circuits"),
    ("W2068163719", "Wallraff 2004 Nature -- Strong coupling in circuit QED"),
    # --- Error correction milestones ---
    ("W2502126115", "Ofek 2016 Nature -- Break-even QEC"),
]

# Quantum-relevance keywords for quick title/concept filter
QC_KEYWORDS = [
    "qubit", "quantum", "superconducting", "bosonic", "josephson", "transmon",
    "cavity", "resonator", "decoherence", "error correction", "cat qubit",
    "kerr", "parametric", "snail", "microwave", "oscillator", "fock",
]


def _get(url: str) -> dict:
    sep = "&" if "?" in url else "?"
    url += f"{sep}mailto={EMAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": "bosonic-lineage-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _cached(path: str, fetch_fn):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    result = fetch_fn()
    if result is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f)
    return result


def fetch_citing_papers(work_id: str, since_date: str) -> list[dict]:
    """Fetch papers citing work_id published on or after since_date (YYYY-MM-DD)."""
    path = os.path.join(CACHE_DIR, f"{work_id}_cites_{since_date}.json")

    def fetch():
        papers, page, per = [], 1, 100
        while True:
            url = (
                f"{OPENALEX_BASE}/works"
                f"?filter=cites:{work_id},from_publication_date:{since_date}"
                f"&per_page={per}&page={page}&sort=cited_by_count:desc"
            )
            data = _get(url)
            batch = data.get("results", [])
            papers.extend(batch)
            if len(batch) < per or len(papers) >= 500:
                break
            page += 1
            time.sleep(SLEEP)
        return papers

    return _cached(path, fetch) or []


def fetch_author_profile(oa_id: str) -> dict:
    clean_id = oa_id.split("/")[-1]
    path = os.path.join(CACHE_DIR, f"{clean_id}_profile.json")

    def fetch():
        data = _get(f"{OPENALEX_BASE}/authors/{clean_id}")
        time.sleep(SLEEP)
        return data

    return _cached(path, fetch) or {}


def _inst(profile: dict) -> str:
    single = profile.get("last_known_institution")
    if single:
        return single.get("display_name", "")
    arr = profile.get("last_known_institutions") or []
    return arr[0].get("display_name", "") if arr else ""


def is_quantum(paper: dict) -> bool:
    title = (paper.get("title") or "").lower()
    concepts = [c.get("display_name", "").lower() for c in paper.get("concepts", [])[:8]]
    return any(kw in title or any(kw in c for c in concepts) for kw in QC_KEYWORDS)


def main(days: int = 90, top_n: int = 20) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    since_date = (date.today() - timedelta(days=days)).isoformat()

    with open("mentorship.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    persons     = data.get("persons", [])
    known_names = {p["name"].lower() for p in persons}
    known_orcids = {
        re.sub(r"https?://orcid\.org/", "", p["orcid"]).strip()
        for p in persons if p.get("orcid")
    }

    # author_id -> {score, papers, name, connected_landmark}
    author_scores: dict[str, dict] = {}

    print(f"\n-- Scanning {len(LANDMARK_PAPERS)} landmark papers (citing since {since_date}) --")

    for work_id, label in LANDMARK_PAPERS:
        print(f"  {label[:60]}...", end=" ", flush=True)
        papers = fetch_citing_papers(work_id, since_date)
        time.sleep(SLEEP)
        print(f"{len(papers)} citing papers")

        for paper in papers:
            if not is_quantum(paper):
                continue
            citations = paper.get("cited_by_count", 0)
            title     = paper.get("title", "")
            year      = (paper.get("publication_date") or "")[:4]

            for auth in paper.get("authorships", [])[:15]:
                a = auth.get("author", {})
                aid = a.get("id", "")
                if not aid:
                    continue
                aname = (a.get("display_name") or "").lower()
                if aname in known_names:
                    continue

                if aid not in author_scores:
                    author_scores[aid] = {
                        "name": a.get("display_name", ""),
                        "score": 0,
                        "papers": [],
                        "landmarks": set(),
                    }
                entry = author_scores[aid]
                entry["score"]    += 10 + min(citations, 1000)
                entry["landmarks"].add(label)
                if len(entry["papers"]) < 3:
                    entry["papers"].append({"title": title, "year": year, "citations": citations})

    # Rank and enrich top candidates
    ranked = sorted(author_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    print(f"\n-- Fetching profiles for top {min(len(ranked), top_n * 3)} candidates --")

    enriched = []
    for oa_id, entry in ranked[: top_n * 3]:
        if len(enriched) >= top_n:
            break
        profile = fetch_author_profile(oa_id)
        if not profile:
            continue

        display_name  = profile.get("display_name", entry["name"])
        cited_by      = profile.get("cited_by_count", 0)
        works_count   = profile.get("works_count", 0)
        institution   = _inst(profile)

        # Deduplicate: skip if ORCID matches known person
        profile_orcid = re.sub(r"https?://orcid\.org/", "", profile.get("orcid") or "").strip()
        if profile_orcid and profile_orcid in known_orcids:
            continue
        if display_name.lower() in known_names:
            continue
        if cited_by < 50:
            continue

        enriched.append({
            "oa_id":       oa_id,
            "name":        display_name,
            "institution": institution,
            "cited_by":    cited_by,
            "works":       works_count,
            "score":       entry["score"],
            "landmarks":   sorted(entry["landmarks"]),
            "papers":      entry["papers"],
        })

    # Write report
    lines = [
        "# New Author Monitor Report",
        f"\n**Date:** {date.today()}  ",
        f"**Window:** last {days} days (since {since_date})  ",
        f"**Landmark papers scanned:** {len(LANDMARK_PAPERS)}  ",
        f"**New authors surfaced:** {len(enriched)}\n",
        "---\n",
        "## How to add someone\n",
        "1. Verify their PhD lineage via ORCID / lab page",
        "2. Add to `mentorship.yaml` under `persons:` + `mentorship_edges:`",
        "3. Run `python pipeline.py --all`\n",
        "---\n",
        "## Top New Authors\n",
        "| # | Name | Institution | Cited by | Score | Landmark connections |",
        "|---|------|-------------|----------|-------|----------------------|",
    ]

    for i, c in enumerate(enriched, 1):
        lm = " / ".join(l.split("--")[0].strip() for l in c["landmarks"])
        lines.append(
            f"| {i} | **{c['name']}** | {c['institution']} | {c['cited_by']:,} "
            f"| {c['score']:,} | {lm} |"
        )

    lines += ["\n---\n", "## Details\n"]
    for i, c in enumerate(enriched, 1):
        lines += [
            f"### {i}. {c['name']}",
            f"- **Institution:** {c['institution']}",
            f"- **Cited by:** {c['cited_by']:,} | **Works:** {c['works']}",
            f"- **OpenAlex:** <{c['oa_id']}>",
            f"- **Citing landmarks:** {', '.join(l.split('--')[0].strip() for l in c['landmarks'])}",
            "- **Recent relevant papers:**",
        ]
        for p in c["papers"]:
            lines.append(f"  - *{p['title']}* ({p['year']}) — {p['citations']} citations")
        lines.append("")

    with open("new_authors_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWritten new_authors_report.md ({len(enriched)} authors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=90, help="Look-back window in days (default: 90)")
    parser.add_argument("--top",  type=int, default=20, help="Number of authors to show (default: 20)")
    args = parser.parse_args()
    main(days=args.days, top_n=args.top)
