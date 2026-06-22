"""
Verify mentorship edges and person data using ORCID and OpenAlex.

For each person:
  1. If orcid: null  -> search OpenAlex for their ORCID
  2. If ORCID found  -> query ORCID public API for education + employment records
  3. Cross-check PhD/postdoc edges against education/employment history
  4. Flag stale affiliations vs. current ORCID employment

Outputs:
  verification_report.md   human-readable pass/fail table
  cache/orcid/             JSON responses (one file per person)

With --apply:
  Writes found ORCID IDs and confirmed verified: true flags back to mentorship.yaml
  (yaml.dump is used, which drops YAML comments — commit first!)

Usage:
  python verify.py            # dry-run: report only
  python verify.py --apply    # report + update mentorship.yaml
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import yaml

OPENALEX_BASE = "https://api.openalex.org"
ORCID_BASE    = "https://pub.orcid.org/v3.0"
CACHE_DIR     = os.path.join("cache", "orcid")
EMAIL         = "lukas.splitthoff@gmail.com"
SLEEP         = 0.35   # seconds between API calls


# -- HTTP helpers --------------------------------------------------------------

def _get(url: str, headers: dict | None = None) -> dict:
    headers = headers or {}
    if "openalex" in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}mailto={EMAIL}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "bosonic-lineage-verify/1.0", **headers}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _cached(path: str, fetch_fn):
    """Return cached JSON, or call fetch_fn(), cache, and return."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    result = fetch_fn()
    if result:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f)
    return result


# -- OpenAlex ORCID lookup -----------------------------------------------------

def openalex_lookup(slug: str, name: str) -> dict:
    """Search OpenAlex for the person; return dict with orcid, openalex_id, institution."""
    path = os.path.join(CACHE_DIR, f"{slug}_oa.json")

    def fetch():
        encoded = urllib.parse.quote(name)
        url = f"{OPENALEX_BASE}/authors?search={encoded}&sort=cited_by_count:desc&per_page=5"
        data = _get(url)
        time.sleep(SLEEP)
        results = data.get("results", [])
        if not results:
            return {}
        best = results[0]
        return {
            "openalex_id": best.get("id", ""),
            "orcid": best.get("orcid") or "",
            "display_name": best.get("display_name", ""),
            "institution": (best.get("last_known_institution") or {}).get("display_name", ""),
            "cited_by_count": best.get("cited_by_count", 0),
        }

    return _cached(path, fetch) or {}


# -- ORCID API -----------------------------------------------------------------

def fetch_orcid(orcid: str) -> dict:
    """Fetch and cache the full ORCID record for an ORCID ID."""
    clean = re.sub(r"https?://orcid\.org/", "", orcid).strip()
    path  = os.path.join(CACHE_DIR, f"{clean.replace('/', '_')}.json")

    def fetch():
        url  = f"{ORCID_BASE}/{clean}/record"
        data = _get(url, headers={"Accept": "application/json"})
        time.sleep(SLEEP)
        return data

    try:
        return _cached(path, fetch) or {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}   # private or non-existent profile
        raise


def _year(node) -> int | None:
    try:
        return int(node["year"]["value"])
    except (TypeError, KeyError):
        return None


def parse_education(record: dict) -> list[dict]:
    out = []
    try:
        groups = record["activities-summary"]["educations"]["affiliation-group"]
    except (KeyError, TypeError):
        return out
    for grp in groups:
        for s in grp.get("summaries", []):
            edu = s.get("education-summary", {})
            out.append({
                "institution": edu.get("organization", {}).get("name", ""),
                "role":        edu.get("role-title", ""),
                "department":  edu.get("department-name", ""),
                "start":       _year(edu.get("start-date") or {}),
                "end":         _year(edu.get("end-date") or {}),
            })
    return out


def parse_employment(record: dict) -> list[dict]:
    out = []
    try:
        groups = record["activities-summary"]["employments"]["affiliation-group"]
    except (KeyError, TypeError):
        return out
    for grp in groups:
        for s in grp.get("summaries", []):
            emp = s.get("employment-summary", {})
            end_node = emp.get("end-date")
            out.append({
                "institution": emp.get("organization", {}).get("name", ""),
                "role":        emp.get("role-title", ""),
                "start":       _year(emp.get("start-date") or {}),
                "end":         _year(end_node or {}),
                "current":     end_node is None,
            })
    return out


# -- Matching helpers -----------------------------------------------------------

# Keywords that uniquely identify institutions
_INST_KEYWORDS = [
    "yale", "eth", "caltech", "chalmers", "sherbrooke", "innsbruck",
    "inria", "mines", "amazon", "aws", "alice", "nord", "rigetti",
    "singapore", "illinois", "chicago", "delft", "paris", "ens",
    "harvard", "mit", "montreal", "uiuc", "maryland",
    "stanford", "princeton", "texas", "austin", "toronto",
    "munich", "oxford", "cambridge", "berkeley", "colorado",
    "pasadena", "tokyo", "sydney", "melbourne", "zurich",
    "nvidia", "ibm", "google", "microsoft", "intel",
]

def inst_match(a: str, b: str) -> bool:
    """Fuzzy institution match based on shared keyword."""
    if not a or not b:
        return False
    al, bl = a.lower(), b.lower()
    return any(kw in al and kw in bl for kw in _INST_KEYWORDS)


def is_phd(role: str) -> bool:
    if not role:
        return False
    rl = role.lower()
    return any(k in rl for k in ["phd", "ph.d", "doctor of philosophy", "doctorate", "doctor"])


# -- Edge verification ---------------------------------------------------------

def verify_edge(edge: dict, person_map: dict, records: dict) -> tuple[str, str]:
    """
    Returns (status, reason) where status ∈ {"confirmed", "unconfirmed", "unknown"}.
    """
    advisor_slug  = edge["advisor"]
    advisee_slug  = edge["advisee"]
    etype         = edge["type"]
    advisor_affil = person_map.get(advisor_slug, {}).get("affiliation", "")
    advisee_orcid = person_map.get(advisee_slug, {}).get("orcid")

    if not advisee_orcid:
        return "unknown", "advisee has no ORCID"

    rec = records.get(advisee_slug)
    if not rec:
        return "unknown", "ORCID record empty or private"

    edu  = parse_education(rec)
    empl = parse_employment(rec)

    if etype in ("phd", "phd_co"):
        phds = [e for e in edu if is_phd(e["role"])]
        if not phds:
            # Fall back to any education with matching institution
            for e in edu:
                if inst_match(e["institution"], advisor_affil):
                    return "confirmed", f"education at {e['institution']} ({e['start']}–{e['end']})"
            return "unknown", "no PhD education entry in ORCID record"
        for e in phds:
            if inst_match(e["institution"], advisor_affil):
                return "confirmed", f"PhD at {e['institution']} ({e['start']}–{e['end']})"
        insts = [e["institution"] for e in phds]
        return "unconfirmed", f"PhD institution(s) {insts} don't match advisor's '{advisor_affil}'"

    if etype == "postdoc":
        for e in empl:
            if inst_match(e["institution"], advisor_affil):
                end = "present" if e["current"] else str(e["end"])
                return "confirmed", f"employment at {e['institution']} ({e['start']}–{end})"
        return "unconfirmed", f"no employment matching advisor's '{advisor_affil}'"

    return "unknown", f"edge type '{etype}' not automatically verifiable"


def check_affiliation(person: dict, rec: dict) -> str | None:
    """Return a warning string if ORCID current employment doesn't match YAML, else None."""
    empl    = parse_employment(rec)
    current = [e for e in empl if e["current"]]
    if not current:
        return None
    yaml_affil = person.get("affiliation", "")
    if any(inst_match(e["institution"], yaml_affil) for e in current):
        return None
    current_insts = [e["institution"] for e in current]
    return f"YAML: '{yaml_affil}' — ORCID current: {current_insts}"


# -- YAML in-place updater (preserves comments with targeted regex) -------------

def _update_orcid_in_yaml(text: str, slug: str, orcid: str) -> str:
    """Replace `orcid: null` with `orcid: "..."` in the person block for `slug`."""
    # Find the slug line, then replace the next orcid: null within ~20 lines
    pattern = re.compile(
        r"(slug:\s+" + re.escape(slug) + r".*?)(orcid:\s*null)",
        re.DOTALL
    )
    return pattern.sub(lambda m: m.group(1) + f'orcid: "{orcid}"', text, count=1)


def _update_verified_in_yaml(text: str, advisor: str, advisee: str) -> str:
    """Replace verified: false with verified: true for a specific edge."""
    # Find the edge block anchored by advisor: + advisee: proximity
    pattern = re.compile(
        r"(advisor:\s+" + re.escape(advisor) + r".*?advisee:\s+" + re.escape(advisee) + r".*?)(verified:\s*false)",
        re.DOTALL
    )
    return pattern.sub(lambda m: m.group(1) + "verified: true", text, count=1)


# -- Main ----------------------------------------------------------------------

def main(apply: bool = False) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open("mentorship.yaml", encoding="utf-8") as f:
        raw_yaml = f.read()
    data = yaml.safe_load(raw_yaml)

    persons = data.get("persons", [])
    edges   = data.get("mentorship_edges", [])
    person_map = {p["slug"]: p for p in persons}

    found_orcids: dict[str, str] = {}   # slug -> new orcid
    records:      dict[str, dict] = {}  # slug -> orcid record

    # -- Step 1: ORCID resolution -----------------------------------------------
    print("\n-- Step 1: ORCID resolution -----------------------------------------")
    for person in persons:
        slug  = person["slug"]
        name  = person["name"]
        orcid = person.get("orcid") or ""

        if not orcid:
            print(f"  searching {name!r}…", end=" ", flush=True)
            result = openalex_lookup(slug, name)
            orcid  = result.get("orcid", "")
            if orcid:
                print(f"found {orcid}")
                found_orcids[slug] = orcid
                person_map[slug] = {**person, "orcid": orcid}
            else:
                print("not found")
        else:
            print(f"  {slug}: already has ORCID {orcid}")

        if orcid:
            rec = fetch_orcid(orcid)
            if rec:
                records[slug] = rec
            else:
                print(f"    WARNING ORCID record for {slug} is private or empty")

    # -- Step 2: Edge verification ----------------------------------------------
    print("\n-- Step 2: Edge verification ----------------------------------------")
    edge_results = []
    for edge in edges:
        status, reason = verify_edge(edge, person_map, records)
        edge_results.append({**edge, "status": status, "reason": reason})
        icon = {"confirmed": "OK", "unconfirmed": "FAIL", "unknown": "?"}[status]
        print(f"  {icon} {edge['advisor']} -> {edge['advisee']} ({edge['type']}): {reason}")

    # -- Step 3: Affiliation check ----------------------------------------------
    print("\n-- Step 3: Affiliation check ----------------------------------------")
    affil_warnings = []
    for person in persons:
        rec  = records.get(person["slug"])
        if not rec:
            continue
        warn = check_affiliation(person, rec)
        if warn:
            print(f"  WARNING {person['slug']}: {warn}")
            affil_warnings.append({"slug": person["slug"], "warning": warn})
        else:
            print(f"  OK {person['slug']}")

    # -- Write report -----------------------------------------------------------
    confirmed   = [r for r in edge_results if r["status"] == "confirmed"]
    unconfirmed = [r for r in edge_results if r["status"] == "unconfirmed"]
    unknown     = [r for r in edge_results if r["status"] == "unknown"]

    lines = [
        "# Lineage Verification Report",
        f"\n**Date:** {date.today()}  ",
        f"**Edges:** {len(edge_results)} total — "
        f"{len(confirmed)} ✅ confirmed, {len(unconfirmed)} ❌ unconfirmed, {len(unknown)} ❓ unknown  ",
        f"**New ORCIDs found:** {len(found_orcids)}",
        "\n---\n",
        "## Mentorship Edges\n",
        "| Edge | Type | Status | Detail |",
        "|------|------|--------|--------|",
    ]
    for r in edge_results:
        icon = {"confirmed": "✅", "unconfirmed": "❌", "unknown": "❓"}[r["status"]]
        lines.append(f"| {r['advisor']} -> {r['advisee']} | {r['type']} | {icon} | {r['reason']} |")

    if affil_warnings:
        lines += ["\n## Affiliation Mismatches\n"]
        for w in affil_warnings:
            lines.append(f"- WARNING️ **{w['slug']}**: {w['warning']}")

    if found_orcids:
        lines += ["\n## Newly Resolved ORCID IDs\n"]
        for slug, orcid in found_orcids.items():
            lines.append(f"- **{slug}**: `{orcid}`")
        if not apply:
            lines.append("\n_Run `python verify.py --apply` to write these to mentorship.yaml._")

    with open("verification_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nWritten verification_report.md")

    # -- Apply updates to YAML -------------------------------------------------
    if apply:
        print("\n-- Applying updates to mentorship.yaml ------------------------------")
        updated = raw_yaml

        for slug, orcid in found_orcids.items():
            before = updated
            updated = _update_orcid_in_yaml(updated, slug, orcid)
            if updated != before:
                print(f"  orcid set: {slug} = {orcid}")
            else:
                print(f"  WARNING could not patch orcid for {slug} (regex missed)")

        for r in edge_results:
            if r["status"] == "confirmed" and not r.get("verified", False):
                before  = updated
                updated = _update_verified_in_yaml(updated, r["advisor"], r["advisee"])
                if updated != before:
                    print(f"  verified=true: {r['advisor']} -> {r['advisee']}")

        with open("mentorship.yaml", "w", encoding="utf-8") as f:
            f.write(updated)
        print("  Saved mentorship.yaml (comments preserved)")

    # Print summary
    print(f"\nSummary: {len(confirmed)}/{len(edge_results)} edges confirmed",
          f"| {len(found_orcids)} new ORCIDs | {len(affil_warnings)} affiliation warnings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Write ORCID IDs and verified flags back to mentorship.yaml")
    args = parser.parse_args()
    main(apply=args.apply)

