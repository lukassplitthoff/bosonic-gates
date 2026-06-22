"""
Fetch bibliometric data from OpenAlex for all persons in mentorship.yaml.

Outputs: contributions.json
Cache:   cache/<slug>.json  (one file per author, keyed by OpenAlex author ID)

Usage:
    python build_contributions.py

Requirements: pyyaml (pip install pyyaml)
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import yaml

CACHE_DIR = "cache"
OPENALEX_BASE = "https://api.openalex.org"
EMAIL = "lukas.splitthoff@gmail.com"   # polite pool — higher rate limits
MAX_WORKS = 200                         # cap per author to keep runtime short
SLEEP_BETWEEN = 0.4                     # seconds between API calls


def _get(url: str) -> dict:
    url = url + ("&" if "?" in url else "?") + f"mailto={EMAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": "bosonic-gates-lineage/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def resolve_author(slug: str, name: str, orcid: str | None) -> str | None:
    """Return an OpenAlex author ID for this person, or None on failure."""
    cache_path = os.path.join(CACHE_DIR, f"{slug}_id.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f).get("id")

    author_id = None
    if orcid:
        url = f"{OPENALEX_BASE}/authors?filter=orcid:{orcid}"
        try:
            data = _get(url)
            if data["results"]:
                author_id = data["results"][0]["id"]
                print(f"  [{slug}] resolved via ORCID: {author_id}")
        except Exception as e:
            print(f"  [{slug}] ORCID lookup failed: {e}")
        time.sleep(SLEEP_BETWEEN)

    if not author_id:
        encoded = urllib.parse.quote(name)
        url = f"{OPENALEX_BASE}/authors?search={encoded}&sort=cited_by_count:desc"
        try:
            data = _get(url)
            if data["results"]:
                author_id = data["results"][0]["id"]
                print(f"  [{slug}] resolved via name search: {author_id}")
        except Exception as e:
            print(f"  [{slug}] name search failed: {e}")
        time.sleep(SLEEP_BETWEEN)

    if author_id:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"id": author_id}, f)

    return author_id


def author_position(authorship: list, author_id: str) -> str:
    """Return 'first', 'last', or 'middle'."""
    ids = [a["author"]["id"] for a in authorship]
    if not ids:
        return "middle"
    if ids[0] == author_id:
        return "first"
    if ids[-1] == author_id:
        return "last"
    return "middle"


def fetch_works(author_id: str, slug: str) -> list:
    """Fetch up to MAX_WORKS works for author_id; use cache when available."""
    cache_path = os.path.join(CACHE_DIR, f"{slug}_works.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    works = []
    page = 1
    per_page = 50
    while len(works) < MAX_WORKS:
        url = (
            f"{OPENALEX_BASE}/works"
            f"?filter=authorships.author.id:{author_id}"
            f"&per_page={per_page}&page={page}"
            f"&sort=cited_by_count:desc"
        )
        try:
            data = _get(url)
        except Exception as e:
            print(f"  [{slug}] works fetch failed on page {page}: {e}")
            break
        batch = data.get("results", [])
        if not batch:
            break
        works.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(works, f)
    return works


def process_person(person: dict) -> dict:
    slug = person["slug"]
    name = person["name"]
    orcid = person.get("orcid")

    print(f"Processing {name} ({slug})…")
    author_id = resolve_author(slug, name, orcid)
    if not author_id:
        print(f"  [{slug}] could not resolve — skipping")
        return {"n_first": 0, "n_last": 0, "lead_papers": [], "flagged": True}

    works = fetch_works(author_id, slug)
    n_first = n_last = 0
    lead_papers = []

    for w in works:
        pos = author_position(w.get("authorships", []), author_id)
        if pos == "first":
            n_first += 1
        elif pos == "last":
            n_last += 1
        if pos in ("first", "last"):
            lead_papers.append({
                "title": w.get("title", ""),
                "year": w.get("publication_year"),
                "citations": w.get("cited_by_count", 0),
                "doi": w.get("doi"),
            })

    lead_papers.sort(key=lambda p: p["citations"], reverse=True)
    flagged = person.get("flagged", False)

    return {
        "n_first": n_first,
        "n_last": n_last,
        "lead_papers": lead_papers[:20],
        "flagged": flagged,
    }


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open("mentorship.yaml") as f:
        data = yaml.safe_load(f)

    persons = data.get("persons", [])
    out_path = "contributions.json"

    # Load existing results so a re-run only fetches missing entries
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {}

    for person in persons:
        slug = person["slug"]
        if slug in results:
            print(f"Skipping {slug} (already in contributions.json)")
            continue
        results[slug] = process_person(person)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Written contributions.json for {len(results)} persons.")


if __name__ == "__main__":
    main()
