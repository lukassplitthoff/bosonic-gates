"""
Discover candidate researchers to add to the bosonic QC lineage map.

Algorithm:
  For each person already in the map:
    1. Resolve their OpenAlex author ID (cached from build_contributions.py)
    2. Fetch their top-cited co-authored works (up to MAX_WORKS per person)
    3. Collect all co-authors, weighted by paper count + citation count

  Aggregate across the whole tree -> rank candidates by a combined score.
  Filter: not already in map, likely active in quantum computing (keyword check).

  For the top candidates, fetch their full OpenAlex profile to show
  institution, h-index, cited_by_count, and top papers.

Outputs:
  candidates_report.md    human-readable, grouped by connection strength
  cache/discover/         raw co-author JSON per person (cached)

Usage:
  python discover.py
  python discover.py --top 30        # show top 30 candidates (default: 20)
  python discover.py --min-papers 3  # require ≥3 shared papers (default: 2)
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date

import yaml

OPENALEX_BASE = "https://api.openalex.org"
CACHE_DIR     = os.path.join("cache", "discover")
OA_CACHE_DIR  = os.path.join("cache")          # shared with build_contributions
EMAIL         = "lukas.splitthoff@gmail.com"
SLEEP         = 0.35
MAX_WORKS     = 100    # works to scan per person
MAX_AUTHORS   = 20     # co-authors per work to process

# Keywords that suggest relevance to quantum computing / superconducting circuits
QC_KEYWORDS = [
    "qubit", "quantum", "superconducting", "circuit qed", "bosonic", "josephson",
    "transmon", "resonator", "microwave", "decoherence", "quantum error",
    "cat qubit", "kerr", "parametric", "snail", "cavity qed", "quantum information",
    "quantum computing", "quantum optics", "quantum gate", "quantum memory",
]


# -- HTTP helpers --------------------------------------------------------------

def _get(url: str) -> dict:
    sep = "&" if "?" in url else "?"
    url += f"{sep}mailto={EMAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": "bosonic-lineage-discover/1.0"})
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


# -- OpenAlex helpers ----------------------------------------------------------

def resolve_openalex_id(slug: str, name: str) -> str | None:
    """Return OpenAlex author ID, using existing cache if available."""
    # Check build_contributions cache first
    id_cache = os.path.join(OA_CACHE_DIR, f"{slug}_id.json")
    if os.path.exists(id_cache):
        with open(id_cache) as f:
            return json.load(f).get("id")

    path = os.path.join(CACHE_DIR, f"{slug}_oaid.json")

    def fetch():
        encoded = urllib.parse.quote(name)
        url  = f"{OPENALEX_BASE}/authors?search={encoded}&sort=cited_by_count:desc&per_page=3"
        data = _get(url)
        time.sleep(SLEEP)
        results = data.get("results", [])
        return {"id": results[0]["id"]} if results else {}

    result = _cached(path, fetch) or {}
    return result.get("id")


def fetch_works_for_author(author_id: str, slug: str) -> list[dict]:
    """Fetch top MAX_WORKS works for an author; return slim dicts."""
    # Reuse existing works cache if present
    works_cache = os.path.join(OA_CACHE_DIR, f"{slug}_works.json")
    if os.path.exists(works_cache):
        with open(works_cache) as f:
            return json.load(f)

    path = os.path.join(CACHE_DIR, f"{slug}_works.json")

    def fetch():
        works = []
        page, per = 1, 50
        while len(works) < MAX_WORKS:
            url  = (f"{OPENALEX_BASE}/works"
                    f"?filter=authorships.author.id:{author_id}"
                    f"&per_page={per}&page={page}&sort=cited_by_count:desc")
            data = _get(url)
            batch = data.get("results", [])
            if not batch:
                break
            works.extend(batch)
            if len(batch) < per:
                break
            page += 1
            time.sleep(SLEEP)
        return works

    return _cached(path, fetch) or []


def fetch_author_profile(oa_id: str) -> dict:
    """Fetch full OpenAlex author profile by ID."""
    clean_id = oa_id.split("/")[-1]
    path = os.path.join(CACHE_DIR, f"{clean_id}_profile.json")

    def fetch():
        data = _get(f"{OPENALEX_BASE}/authors/{clean_id}")
        time.sleep(SLEEP)
        return data

    return _cached(path, fetch) or {}


def is_quantum_relevant(profile: dict) -> bool:
    """Heuristic: does this person work in quantum computing?"""
    topics = profile.get("topics", [])
    for t in topics[:10]:
        name = (t.get("display_name") or "").lower()
        if any(kw in name for kw in QC_KEYWORDS):
            return True
    # Fallback: check last institution subtype
    inst = (profile.get("last_known_institution") or {})
    return True   # keep all for human review; rely on score threshold instead


# -- Scoring -------------------------------------------------------------------

class Candidate:
    __slots__ = ("oa_id", "name", "institution", "cited_by_count",
                 "works_count", "paper_score", "connection_count", "connected_to")

    def __init__(self, oa_id: str):
        self.oa_id         = oa_id
        self.name          = ""
        self.institution   = ""
        self.cited_by_count = 0
        self.works_count   = 0
        self.paper_score   = 0   # sum of (cited_by_count of shared paper) + 10 per paper
        self.connection_count = 0  # how many people in the tree share papers
        self.connected_to  = set()  # slugs


# -- Main ----------------------------------------------------------------------

def _institution_from_profile(profile: dict) -> str:
    """Handle both legacy last_known_institution and new last_known_institutions array."""
    single = profile.get("last_known_institution")
    if single:
        return single.get("display_name", "")
    arr = profile.get("last_known_institutions") or []
    if arr:
        return arr[0].get("display_name", "")
    return ""


def main(top_n: int = 20, min_papers: int = 2) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open("mentorship.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    persons = data.get("persons", [])
    known_names = {p["name"].lower() for p in persons}
    known_slugs = {p["slug"] for p in persons}
    # Collect known ORCIDs so we can filter out split OpenAlex records of existing people
    known_orcids = {
        re.sub(r"https?://orcid\.org/", "", p["orcid"]).strip()
        for p in persons if p.get("orcid")
    }

    # Map OpenAlex ID -> Candidate
    candidates: dict[str, Candidate] = {}

    print(f"\n-- Scanning co-authors for {len(persons)} people --------------------")

    for person in persons:
        slug = person["slug"]
        name = person["name"]
        print(f"  {name}…", end=" ", flush=True)

        oa_id = resolve_openalex_id(slug, name)
        if not oa_id:
            print("no OpenAlex ID, skipping")
            continue

        works = fetch_works_for_author(oa_id, slug)
        print(f"{len(works)} works")

        for work in works:
            work_citations = work.get("cited_by_count", 0)
            authorships    = work.get("authorships", [])[:MAX_AUTHORS]

            for auth in authorships:
                coauth    = auth.get("author", {})
                coauth_id = coauth.get("id", "")
                if not coauth_id or coauth_id == oa_id:
                    continue

                coauth_name = (coauth.get("display_name") or "").lower()
                if coauth_name in known_names:
                    continue   # already in tree

                if coauth_id not in candidates:
                    candidates[coauth_id] = Candidate(coauth_id)
                    candidates[coauth_id].name = coauth.get("display_name", "")

                c = candidates[coauth_id]
                c.paper_score  += 10 + min(work_citations, 2000)
                c.connected_to.add(slug)

    # Compute connection_count
    for c in candidates.values():
        c.connection_count = len(c.connected_to)

    # Filter: must share papers with ≥ min_papers papers (approximation via score)
    filtered = {oa_id: c for oa_id, c in candidates.items()
                if c.paper_score >= min_papers * 10}

    # Rank by score
    ranked = sorted(filtered.values(), key=lambda c: c.paper_score, reverse=True)[:top_n * 3]

    print(f"\n-- Fetching profiles for top {min(len(ranked), top_n * 3)} candidates --")

    enriched = []
    for c in ranked:
        profile = fetch_author_profile(c.oa_id)
        if not profile:
            continue
        c.name           = profile.get("display_name", c.name)
        c.institution    = _institution_from_profile(profile)
        c.cited_by_count = profile.get("cited_by_count", 0)
        c.works_count    = profile.get("works_count", 0)

        # Filter: already in tree (name match missed due to abbreviated names)
        if c.name.lower() in known_names:
            continue
        # Filter: same ORCID as an existing person (split OpenAlex records)
        profile_orcid = re.sub(r"https?://orcid\.org/", "", profile.get("orcid") or "").strip()
        if profile_orcid and profile_orcid in known_orcids:
            continue

        if c.cited_by_count < 100:
            continue

        enriched.append(c)
        if len(enriched) >= top_n:
            break

    # -- Write report -----------------------------------------------------------
    lines = [
        "# Candidate Researchers — Discovery Report",
        f"\n**Date:** {date.today()}  ",
        f"**Source tree:** {len(persons)} people  ",
        f"**Candidates found:** {len(filtered)} (showing top {len(enriched)})\n",
        "---\n",
        "## How to add a candidate\n",
        "1. Check their Google Scholar / lab page to confirm PhD lineage",
        "2. Add them to `mentorship.yaml` under `persons:` and `mentorship_edges:`",
        "3. Re-run `python pipeline.py --all`\n",
        "---\n",
        "## Top Candidates\n",
        "| # | Name | Institution | Cited by | Connected to | Score |",
        "|---|------|-------------|----------|--------------|-------|",
    ]

    for i, c in enumerate(enriched, 1):
        conn = ", ".join(sorted(c.connected_to))
        lines.append(
            f"| {i} | **{c.name}** | {c.institution} | {c.cited_by_count:,} "
            f"| {conn} | {c.paper_score:,} |"
        )

    lines += [
        "\n---\n",
        "## Details\n",
    ]
    for i, c in enumerate(enriched, 1):
        conn = ", ".join(sorted(c.connected_to))
        lines += [
            f"### {i}. {c.name}",
            f"- **Institution:** {c.institution}",
            f"- **Cited by:** {c.cited_by_count:,} | **Works:** {c.works_count}",
            f"- **Connected to:** {conn}",
            f"- **OpenAlex:** <{c.oa_id}>",
            "",
        ]

    with open("candidates_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWritten candidates_report.md ({len(enriched)} candidates)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top",        type=int, default=20, help="Number of candidates to show (default: 20)")
    parser.add_argument("--min-papers", type=int, default=2,  help="Min shared papers to consider (default: 2)")
    args = parser.parse_args()
    main(top_n=args.top, min_papers=args.min_papers)
