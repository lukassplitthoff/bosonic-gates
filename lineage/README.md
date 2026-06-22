# Bosonic QC Academic Lineage

A self-contained interactive tree diagram of PhD/postdoc mentorship relationships among key researchers in circuit QED and bosonic quantum computing, with community-contribution metrics sourced from OpenAlex.

## Quick start

```bash
cd lineage

# 1. Install the only extra dependency
pip install pyyaml

# 2. Fetch bibliometric data from OpenAlex (writes contributions.json)
python build_contributions.py

# 3. Generate the interactive HTML tree
python render_tree.py

# 4. Open in your browser
open lineage_tree.html        # macOS
start lineage_tree.html       # Windows
xdg-open lineage_tree.html    # Linux
```

> **Note:** `lineage_tree.html` loads D3 v7 from the CDN (`d3js.org`) on first open — an internet connection is required.

## File overview

| File | Purpose |
|------|---------|
| `mentorship.yaml` | Human-curated data: persons, companies, mentorship edges, affiliation edges |
| `build_contributions.py` | Fetches OpenAlex bibliometrics for all persons; writes `contributions.json` |
| `contributions.json` | Cached bibliometric data (first/last author counts, top papers) |
| `render_tree.py` | Reads YAML + JSON → writes `lineage_tree.html` |
| `cache/` | Per-author cache files from OpenAlex (gitignored) |

## Adding a new person

1. Add an entry under `persons:` in `mentorship.yaml`:
   ```yaml
   - slug: yourname          # unique key, lowercase, no spaces
     name: Full Name
     orcid: null             # add ORCID string for precise resolution
     affiliation: Institution
     role: pi                # founding_pi | pi | pi_company_head | company_head
   ```

2. Add mentorship edges under `mentorship_edges:`:
   ```yaml
   - advisor: advisorslug
     advisee: yourname
     type: phd               # phd | phd_co | postdoc
     verified: false
   ```

3. Re-run the pipeline:
   ```bash
   python build_contributions.py
   python render_tree.py
   ```

## Adding a company / spinout

1. Add under `companies:`:
   ```yaml
   - slug: mycompany
     name: My Company
     type: company           # company | lab
   ```

2. Link a founder under `affiliation_edges:`:
   ```yaml
   - person: foundername
     company: mycompany
     type: founded           # founded | co-founded | leads | faculty
     verified: false
   ```

3. Re-run `python render_tree.py` (no API calls needed for company nodes).

## Data quality flags

- `verified: false` on an edge — relationship is known but not cross-checked against a primary source (CV, PhD thesis record).
- `flagged: true` on a person — OpenAlex author was matched by name only, not ORCID. Bibliometric counts may include other authors with the same name. Add an `orcid:` to fix.

## Edge types

| Type | Description | Visual |
|------|-------------|--------|
| `phd` | Primary PhD advisor | Solid blue arrow |
| `phd_co` | PhD co-advisor | Dashed blue arrow |
| `postdoc` | Postdoc host | Dashed green arrow |
| `founded` | Company founder | Solid amber arrow |
| `co-founded` | Company co-founder | Dashed amber arrow |
| `leads` | Current head (non-founder) | Dotted amber arrow |
| `faculty` | Faculty member (shared lab) | Faint dotted arrow |

## Refreshing cached data

Delete the relevant cache files and re-run:
```bash
rm cache/<slug>_id.json cache/<slug>_works.json
python build_contributions.py
```

Or delete `cache/` entirely to refresh all authors.
