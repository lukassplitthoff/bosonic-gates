"""
Generate lineage_tree.html from mentorship.yaml + contributions.json.

Usage:
    python render_tree.py

Outputs: lineage_tree.html  (self-contained; requires internet for D3 CDN)
"""

import json
import os
import yaml


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    with open("mentorship.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    contribs = {}
    if os.path.exists("contributions.json"):
        with open("contributions.json", encoding="utf-8") as f:
            contribs = json.load(f)

    return data, contribs


# ── Tree hierarchy builder ────────────────────────────────────────────────────

def build_tree(data):
    persons = data.get("persons", [])
    companies = data.get("companies", [])
    mentorship_edges = data.get("mentorship_edges", [])
    affiliation_edges = data.get("affiliation_edges", [])

    # Map slug → node info
    node_map = {}
    for p in persons:
        node_map[p["slug"]] = {
            "id": p["slug"],
            "name": p["name"],
            "type": "person",
            "role": p.get("role", "pi"),
            "affiliation": p.get("affiliation", ""),
            "flagged": p.get("flagged", False),
            "google_scholar": p.get("google_scholar"),
        }
    for c in companies:
        node_map[c["slug"]] = {
            "id": c["slug"],
            "name": c["name"],
            "type": c.get("type", "company"),   # "company" or "lab"
            "role": c.get("type", "company"),
            "affiliation": "",
            "flagged": False,
        }

    # Primary parent priority: phd > phd_co > postdoc
    # Postdoc is used as tree parent only when no PhD relationship exists (avoids orphan roots)
    phd_parents: dict[str, str] = {}
    phd_co_parents: dict[str, str] = {}
    postdoc_parents: dict[str, str] = {}
    for e in mentorship_edges:
        t = e["advisee"]
        if e["type"] == "phd" and t not in phd_parents:
            phd_parents[t] = e["advisor"]
        elif e["type"] == "phd_co" and t not in phd_co_parents:
            phd_co_parents[t] = e["advisor"]
        elif e["type"] == "postdoc" and t not in postdoc_parents:
            postdoc_parents[t] = e["advisor"]

    primary_parent = {**postdoc_parents, **phd_co_parents, **phd_parents}   # phd wins

    # Primary company → person link: founded > leads > co-founded
    type_priority = {"founded": 0, "leads": 1, "co-founded": 2, "faculty": 3}
    company_primary_person: dict[str, tuple[str, int]] = {}
    for e in affiliation_edges:
        prio = type_priority.get(e.get("type", "faculty"), 99)
        slug = e["company"]
        if slug not in company_primary_person or prio < company_primary_person[slug][1]:
            company_primary_person[slug] = (e["person"], prio)

    # Adjacency: parent → [children]
    children: dict[str, list[str]] = {slug: [] for slug in node_map}

    for child, parent in primary_parent.items():
        if parent in children and child in children:
            children[parent].append(child)

    # Companies are rendered in a separate bottom row — not added to tree hierarchy

    # Roots: persons with no primary parent
    roots = [p for p in persons if p["slug"] not in primary_parent]

    def build_subtree(slug):
        node = dict(node_map[slug])
        kids = [build_subtree(c) for c in children.get(slug, [])]
        if kids:
            node["children"] = kids
        return node

    tree_data = {
        "id": "__root__",
        "name": "",
        "type": "__root__",
        "children": [build_subtree(r["slug"]) for r in roots],
    }

    return tree_data, node_map


def build_links(data):
    """Return mentorship edges for overlay rendering (affiliation edges handled separately)."""
    links = []
    for e in data.get("mentorship_edges", []):
        links.append({
            "source": e["advisor"],
            "target": e["advisee"],
            "type": e["type"],
            "verified": e.get("verified", True),
        })
    return links


def build_affil_links(data):
    """Return person→company edges (founded / leads / co-founded only, no faculty/lab)."""
    links = []
    for e in data.get("affiliation_edges", []):
        if e.get("type") not in ("faculty",):
            links.append({
                "source": e["person"],
                "target": e["company"],
                "type": e.get("type", "leads"),
            })
    return links


def build_companies(data):
    """Return company nodes (type==company only, not labs) for bottom-row rendering."""
    return [
        {"id": c["slug"], "name": c["name"]}
        for c in data.get("companies", [])
        if c.get("type") == "company"
    ]


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bosonic QC Mentorship Lineage</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #ffffff;
      color: #111111;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif;
      overflow: hidden;
    }
    #canvas { display: block; }

    .link { fill: none; }
    .node { cursor: pointer; }
    .node-label { font-size: 11px; fill: #111; pointer-events: none; font-weight: 500; }
    .node-sublabel { font-size: 9px; fill: #888; pointer-events: none; }

    /* Side panel */
    #panel {
      position: fixed; right: 0; top: 0;
      width: 320px; height: 100vh;
      background: #fafafa; border-left: 1px solid #e5e7eb;
      padding: 24px 20px 20px; overflow-y: auto;
      transform: translateX(100%); transition: transform 0.2s ease;
      z-index: 10;
    }
    #panel.open { transform: translateX(0); }
    #panel-close {
      position: absolute; top: 14px; right: 14px;
      background: none; border: none; color: #888;
      cursor: pointer; font-size: 20px; line-height: 1;
    }
    #panel-close:hover { color: #111; }
    #panel h2 { font-size: 16px; color: #111; margin-bottom: 3px; line-height: 1.3; font-weight: 600; }
    .panel-affil { font-size: 12px; color: #666; margin-bottom: 10px; }
    .scholar-link {
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 11px; color: #1a56db; text-decoration: none;
      border: 1px solid #c7d7f9; border-radius: 5px;
      padding: 4px 10px; margin-bottom: 12px;
      transition: background 0.15s;
    }
    .scholar-link:hover { background: #eef2ff; }
    .panel-badge {
      display: inline-block; font-size: 10px; padding: 2px 8px;
      border-radius: 12px; margin-bottom: 14px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.4px;
    }
    .stat-row { display: flex; justify-content: space-between; padding: 6px 0;
      border-bottom: 1px solid #e5e7eb; font-size: 13px; }
    .stat-label { color: #666; }
    .stat-val { color: #111; font-weight: 600; }
    .papers-header {
      font-size: 10px; color: #888; text-transform: uppercase;
      letter-spacing: 0.5px; margin: 18px 0 10px;
    }
    .paper { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb; }
    .paper-title { font-size: 12px; color: #222; margin-bottom: 3px; line-height: 1.4; }
    .paper-meta { font-size: 11px; color: #888; }
    .flagged-warning {
      background: #fffbeb; border: 1px solid #fcd34d;
      border-radius: 6px; padding: 8px 10px;
      font-size: 11px; color: #92400e; margin-bottom: 14px;
    }

    /* Legend */
    #legend {
      position: fixed; bottom: 20px; left: 20px;
      background: #ffffffee; backdrop-filter: blur(4px);
      border: 1px solid #e5e7eb; border-radius: 8px;
      padding: 12px 14px; z-index: 5;
    }
    .leg-section { font-size: 9px; color: #999; text-transform: uppercase;
      letter-spacing: 0.6px; margin-bottom: 6px; margin-top: 10px; }
    .leg-section:first-child { margin-top: 0; }
    .leg-row { display: flex; align-items: center; gap: 8px;
      margin-bottom: 5px; font-size: 11px; color: #333; }

    /* Title */
    #title-box {
      position: fixed; top: 16px; left: 20px; z-index: 5;
    }
    #title-box h1 { font-size: 16px; font-weight: 700; color: #111; }
    #title-box p { font-size: 11px; color: #888; margin-top: 2px; }

    /* Tooltip */
    #tooltip {
      position: fixed; pointer-events: none;
      background: #fff; border: 1px solid #e5e7eb;
      border-radius: 6px; padding: 8px 12px;
      font-size: 12px; color: #111;
      opacity: 0; transition: opacity 0.1s;
      max-width: 220px; z-index: 20;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Zoom hint */
    #zoom-hint {
      position: fixed; bottom: 20px; right: 20px;
      font-size: 10px; color: #bbb; z-index: 5;
    }
  </style>
</head>
<body>
  <svg id="canvas"></svg>

  <div id="panel">
    <button id="panel-close" onclick="closePanel()">&#215;</button>
    <h2 id="panel-name"></h2>
    <div class="panel-affil" id="panel-affil"></div>
    <div id="panel-badge-wrap"></div>
    <div id="panel-scholar" style="display:none"></div>
    <div id="panel-warning" class="flagged-warning" style="display:none">
      &#9888; Author matched by name only — not ORCID-verified.
    </div>
    <div id="panel-stats"></div>
    <div class="papers-header" id="panel-papers-header" style="display:none">Selected papers</div>
    <div id="panel-papers"></div>
  </div>

  <div id="legend">
    <div class="leg-section">Mentorship</div>
    <div class="leg-row">
      <svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="#222" stroke-width="1.5"/></svg>
      PhD / co-advisor
    </div>
    <div class="leg-row">
      <svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="#222" stroke-width="1.5" stroke-dasharray="5 3"/></svg>
      Postdoc mentor (dashed)
    </div>
    <div class="leg-section">People</div>
    <div class="leg-row">
      <svg width="36" height="16"><rect x="1" y="1" width="34" height="14" rx="4" fill="#fff" stroke="#1a56db" stroke-width="2"/></svg>
      Founding PI
    </div>
    <div class="leg-row">
      <svg width="36" height="16"><rect x="1" y="1" width="34" height="14" rx="4" fill="#fff" stroke="#0e7c3a" stroke-width="2"/></svg>
      PI / researcher
    </div>
    <div class="leg-row">
      <svg width="36" height="16"><rect x="1" y="1" width="34" height="14" rx="4" fill="#fff" stroke="#b45309" stroke-width="2"/></svg>
      Company head
    </div>
    <div class="leg-row">
      <svg width="36" height="16"><rect x="1" y="1" width="34" height="14" rx="4" fill="#f8f8f8" stroke="#bbb" stroke-width="1.5"/></svg>
      Spinout company
    </div>
  </div>

  <div id="title-box">
    <h1>Bosonic QC Mentorship Lineage</h1>
    <p>PhD advisor (solid) · postdoc mentor (dashed) · spinouts below</p>
  </div>

  <div id="tooltip"></div>
  <div id="zoom-hint">scroll to zoom &nbsp;·&nbsp; drag to pan &nbsp;·&nbsp; click for details</div>

  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
  // ── Injected data ────────────────────────────────────────────────────────────
  const TREE_DATA   = __TREE_DATA__;
  const LINKS_DATA  = __LINKS_DATA__;
  const CONTRIBS    = __CONTRIBS__;
  const AFFIL_LINKS = __AFFIL_LINKS__;
  const COMPANIES   = __COMPANIES__;

  // ── Constants ─────────────────────────────────────────────────────────────────
  const ROLE_COLOR = {
    founding_pi:     '#1a56db',
    pi:              '#0e7c3a',
    pi_company_head: '#b45309',
    company_head:    '#b45309',
    company:         '#555',
    lab:             '#555',
  };

  const ROLE_LABEL = {
    founding_pi:     'Founding PI',
    pi:              'PI / researcher',
    pi_company_head: 'PI & Company head',
    company_head:    'Company head',
    company:         'Company',
    lab:             'Lab',
  };

  const BADGE_BG = {
    founding_pi:     '#eff6ff',
    pi:              '#f0fdf4',
    pi_company_head: '#fffbeb',
    company_head:    '#fffbeb',
    company:         '#f3f4f6',
    lab:             '#f3f4f6',
  };

  const NODE_DX = 120;  // horizontal node spacing
  const NODE_DY = 130;  // vertical (depth) spacing
  const PERS_W = 90;  const PERS_H = 26; // person box
  const COMP_W = 110; const COMP_H = 26; // company/lab box

  function halfH(data) {
    return data.type === 'person' ? PERS_H / 2 : COMP_H / 2;
  }

  // ── SVG setup ─────────────────────────────────────────────────────────────────
  const svgEl = document.getElementById('canvas');
  svgEl.setAttribute('width',  window.innerWidth);
  svgEl.setAttribute('height', window.innerHeight);
  const svg = d3.select('#canvas');

  // Single black arrowhead for all edges
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arr')
    .attr('viewBox', '0 -4 8 8')
    .attr('refX', 8).attr('refY', 0)
    .attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-3.5L8,0L0,3.5Z')
    .attr('fill', '#444');

  const g = svg.append('g');

  // ── Zoom / pan ────────────────────────────────────────────────────────────────
  const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoom);

  // ── Tree layout ───────────────────────────────────────────────────────────────
  const hierarchy = d3.hierarchy(TREE_DATA);
  const treeLayout = d3.tree().nodeSize([NODE_DX, NODE_DY]);
  treeLayout(hierarchy);

  // Filter out the virtual root from rendered nodes
  const allNodes = hierarchy.descendants().filter(d => d.data.id !== '__root__');
  const allLinks = hierarchy.links().filter(l => l.source.data.id !== '__root__');

  // Build position index for overlay edges
  const posMap = {};
  hierarchy.descendants().forEach(d => { posMap[d.data.id] = { x: d.x, y: d.y }; });

  // Center the tree in the viewport initially
  const xs = allNodes.map(d => d.x);
  const ys = allNodes.map(d => d.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const ty = 80 - Math.min(...ys);
  svg.call(zoom.transform, d3.zoomIdentity.translate(window.innerWidth / 2 - cx, ty));

  // ── Edge style helpers ────────────────────────────────────────────────────────
  // Build a lookup from node id → type for edge offset calculations
  const nodeTypeMap = {};
  hierarchy.descendants().forEach(d => { nodeTypeMap[d.data.id] = d.data.type; });

  function edgeDash(type) {
    if (type === 'postdoc') return '6 4';
    if (type === 'faculty') return '2 4';
    return null;
  }

  // Custom cubic bezier connecting bottom of source to top of target box
  function boxLink(sx, sy, sType, tx, ty, tType) {
    const srcY = sy + (sType === 'person' ? PERS_H / 2 : COMP_H / 2);
    const tgtY = ty - (tType === 'person' ? PERS_H / 2 : COMP_H / 2) - 1;
    const ym = (srcY + tgtY) / 2;
    return `M${sx},${srcY} C${sx},${ym} ${tx},${ym} ${tx},${tgtY}`;
  }

  // ── Draw tree links ───────────────────────────────────────────────────────────
  const treeLinksG = g.append('g').attr('class', 'tree-links');
  treeLinksG.selectAll('path')
    .data(allLinks)
    .join('path')
    .attr('class', 'link')
    .attr('d', d => boxLink(
      d.source.x, d.source.y, d.source.data.type,
      d.target.x, d.target.y, d.target.data.type
    ))
    .each(function(d) {
      const src = d.source.data.id, tgt = d.target.data.id;
      const entry = LINKS_DATA.find(l => l.source === src && l.target === tgt);
      const type = entry ? entry.type : 'phd';
      d3.select(this)
        .attr('stroke', '#333')
        .attr('stroke-width', type === 'faculty' ? 1 : 1.5)
        .attr('stroke-dasharray', edgeDash(type))
        .attr('opacity', type === 'faculty' ? 0.25 : 1)
        .attr('marker-end', type !== 'faculty' ? 'url(#arr)' : null);
    });

  // ── Overlay links (non-primary edges) ────────────────────────────────────────
  const treeEdgeSet = new Set(allLinks.map(l => `${l.source.data.id}|${l.target.data.id}`));

  const overlayLinks = LINKS_DATA.filter(l => {
    const key = `${l.source}|${l.target}`;
    if (treeEdgeSet.has(key)) return false;
    return posMap[l.source] && posMap[l.target];
  });

  const overlayG = g.append('g').attr('class', 'overlay-links');
  overlayG.selectAll('path')
    .data(overlayLinks)
    .join('path')
    .attr('class', 'link')
    .attr('d', d => {
      const s = posMap[d.source], t = posMap[d.target];
      const sType = nodeTypeMap[d.source] || 'person';
      const tType = nodeTypeMap[d.target] || 'person';
      const srcY = s.y + (sType === 'person' ? PERS_H / 2 : COMP_H / 2);
      const tgtY = t.y - (tType === 'person' ? PERS_H / 2 : COMP_H / 2) - 1;
      const mx = Math.min(s.x, t.x) - 55;
      const my = (srcY + tgtY) / 2;
      return `M${s.x},${srcY} C${mx},${my} ${mx},${my} ${t.x},${tgtY}`;
    })
    .each(function(d) {
      d3.select(this)
        .attr('stroke', '#333')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', edgeDash(d.type))
        .attr('opacity', 0.6)
        .attr('marker-end', 'url(#arr)');
    });

  // ── Nodes ─────────────────────────────────────────────────────────────────────
  const nodeG = g.append('g').attr('class', 'nodes');
  const nodeGrp = nodeG.selectAll('g.node')
    .data(allNodes)
    .join('g')
    .attr('class', 'node')
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .on('click', (e, d) => openPanel(d.data))
    .on('mousemove', (e, d) => showTooltip(e, d.data))
    .on('mouseleave', hideTooltip);

  // Person boxes with colored border
  nodeGrp.filter(d => d.data.type === 'person')
    .append('rect')
    .attr('x', -PERS_W / 2).attr('y', -PERS_H / 2)
    .attr('width', PERS_W).attr('height', PERS_H)
    .attr('rx', 6)
    .attr('fill', '#fff')
    .attr('stroke', d => ROLE_COLOR[d.data.role] || '#888')
    .attr('stroke-width', 2);

  // All nodes in hierarchy are persons — no company nodes in tree
  // Person labels: last name inside box, vertically centered
  nodeGrp.append('text')
    .attr('class', 'node-label')
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')
    .text(d => d.data.name.split(' ').slice(-1)[0]);

  // ── Company bottom row ────────────────────────────────────────────────────────
  const COMP_Y_GAP  = 150;
  const MIN_COMP_SEP = 150;
  const companyY = Math.max(...allNodes.map(d => d.y)) + COMP_Y_GAP;

  // Primary x for each company: x of its 'founded' person, else first linked person
  const compPrimaryX = {};
  COMPANIES.forEach(c => {
    const links = AFFIL_LINKS.filter(l => l.target === c.id);
    const primary = links.find(l => l.type === 'founded')
                 || links.find(l => l.type === 'leads')
                 || links[0];
    compPrimaryX[c.id] = primary && posMap[primary.source]
      ? posMap[primary.source].x : 0;
  });

  // Sort by x, then enforce minimum separation
  const sortedComps = [...COMPANIES].sort((a, b) => compPrimaryX[a.id] - compPrimaryX[b.id]);
  const compPos = {};
  sortedComps.forEach((c, i) => {
    let x = compPrimaryX[c.id];
    if (i > 0) {
      const prevX = compPos[sortedComps[i - 1].id].x;
      if (x < prevX + MIN_COMP_SEP) x = prevX + MIN_COMP_SEP;
    }
    compPos[c.id] = { x, y: companyY };
  });

  // Draw affiliation edges (person → company) before company nodes
  const affilG = g.append('g').attr('class', 'affil-links');
  affilG.selectAll('path')
    .data(AFFIL_LINKS.filter(l => compPos[l.target] && posMap[l.source]))
    .join('path')
    .attr('fill', 'none')
    .attr('d', d => {
      const s = posMap[d.source];
      const t = compPos[d.target];
      const srcY = s.y + PERS_H / 2;
      const tgtY = t.y - COMP_H / 2 - 1;
      const ym = (srcY + tgtY) / 2;
      return `M${s.x},${srcY} C${s.x},${ym} ${t.x},${ym} ${t.x},${tgtY}`;
    })
    .attr('stroke', '#aaa')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', d => d.type === 'co-founded' ? '4 3' : null)
    .attr('opacity', 0.7)
    .attr('marker-end', 'url(#arr)');

  // Draw company rects
  const compRowG = g.append('g').attr('class', 'company-nodes');
  const compGrpEl = compRowG.selectAll('g')
    .data(COMPANIES)
    .join('g')
    .attr('transform', d => `translate(${compPos[d.id].x},${compPos[d.id].y})`)
    .on('mousemove', (e, d) => showTooltip(e, d))
    .on('mouseleave', hideTooltip);

  compGrpEl.append('rect')
    .attr('x', -COMP_W / 2).attr('y', -COMP_H / 2)
    .attr('width', COMP_W).attr('height', COMP_H)
    .attr('rx', 6)
    .attr('fill', '#f8f8f8')
    .attr('stroke', '#bbb')
    .attr('stroke-width', 1.5);

  compGrpEl.append('text')
    .attr('class', 'node-label')
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')
    .attr('font-size', '10px')
    .attr('fill', '#555')
    .text(d => d.name);

  // Auto-size company rects to text width
  compGrpEl.each(function() {
    const el = d3.select(this);
    const tw = el.select('text').node().getBBox().width;
    const pw = Math.max(tw + 18, 60);
    el.select('rect').attr('x', -pw / 2).attr('width', pw);
  });

  // ── Side panel ────────────────────────────────────────────────────────────────
  function openPanel(node) {
    document.getElementById('panel-name').textContent = node.name;
    document.getElementById('panel-affil').textContent = node.affiliation || '';

    const badge = document.getElementById('panel-badge-wrap');
    const role = node.role || node.type;
    badge.innerHTML = `<span class="panel-badge" style="background:${BADGE_BG[role] || '#f3f4f6'};color:${ROLE_COLOR[role] || '#333'}">${ROLE_LABEL[role] || role}</span>`;

    document.getElementById('panel-warning').style.display = node.flagged ? 'block' : 'none';

    const scholarEl = document.getElementById('panel-scholar');
    if (node.google_scholar) {
      scholarEl.innerHTML = `<a class="scholar-link" href="${node.google_scholar}" target="_blank" rel="noopener">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 24a7 7 0 110-14 7 7 0 010 14zm0-24L0 9.5l4.838.95L12 8.5l7.162 1.95L24 9.5 12 0zm-1 9.5h2v11h-2V9.5z"/></svg>
        Google Scholar &#8599;
      </a>`;
      scholarEl.style.display = 'block';
    } else {
      scholarEl.style.display = 'none';
    }

    const c = CONTRIBS[node.id];
    const stats = document.getElementById('panel-stats');
    if (c) {
      stats.innerHTML = `
        <div class="stat-row"><span class="stat-label">First-author papers</span><span class="stat-val">${c.n_first}</span></div>
        <div class="stat-row"><span class="stat-label">Last-author papers</span><span class="stat-val">${c.n_last}</span></div>
        <div class="stat-row"><span class="stat-label">Total lead papers</span><span class="stat-val">${c.n_first + c.n_last}</span></div>
      `;
      const ph = document.getElementById('panel-papers-header');
      const pp = document.getElementById('panel-papers');
      if (c.lead_papers && c.lead_papers.length) {
        ph.style.display = 'block';
        pp.innerHTML = c.lead_papers.slice(0, 8).map(p => `
          <div class="paper">
            <div class="paper-title">${p.title || '(untitled)'}</div>
            <div class="paper-meta">${p.year || '&mdash;'} &nbsp;&middot;&nbsp; ${p.citations ?? 0} citations</div>
          </div>`).join('');
      } else {
        ph.style.display = 'none';
        pp.innerHTML = '';
      }
    } else {
      stats.innerHTML = '<div class="stat-row"><span class="stat-label">No bibliometric data yet</span></div>';
      document.getElementById('panel-papers-header').style.display = 'none';
      document.getElementById('panel-papers').innerHTML = '';
    }

    document.getElementById('panel').classList.add('open');
  }

  function closePanel() { document.getElementById('panel').classList.remove('open'); }
  svg.on('click', e => { if (e.target === svgEl) closePanel(); });

  // ── Tooltip ───────────────────────────────────────────────────────────────────
  function showTooltip(event, node) {
    const c = CONTRIBS[node.id];
    const tt = document.getElementById('tooltip');
    tt.innerHTML = `<strong>${node.name}</strong>` +
      (node.affiliation ? `<br><span style="color:#666;font-size:11px">${node.affiliation}</span>` : '') +
      (c ? `<br><span style="color:#999;font-size:11px">${c.n_first} first &middot; ${c.n_last} last author</span>` : '');
    tt.style.opacity = '1';
    tt.style.left  = (event.clientX + 14) + 'px';
    tt.style.top   = (event.clientY - 10) + 'px';
  }
  function hideTooltip() { document.getElementById('tooltip').style.opacity = '0'; }

  // ── Responsive resize ─────────────────────────────────────────────────────────
  window.addEventListener('resize', () => {
    svgEl.setAttribute('width',  window.innerWidth);
    svgEl.setAttribute('height', window.innerHeight);
  });
  </script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    data, contribs = load_data()
    tree_data, _node_map = build_tree(data)
    links_data    = build_links(data)
    affil_links   = build_affil_links(data)
    companies     = build_companies(data)

    html = HTML_TEMPLATE
    html = html.replace("__TREE_DATA__",    json.dumps(tree_data,   ensure_ascii=False))
    html = html.replace("__LINKS_DATA__",   json.dumps(links_data,  ensure_ascii=False))
    html = html.replace("__CONTRIBS__",     json.dumps(contribs,    ensure_ascii=False))
    html = html.replace("__AFFIL_LINKS__",  json.dumps(affil_links, ensure_ascii=False))
    html = html.replace("__COMPANIES__",    json.dumps(companies,   ensure_ascii=False))

    out = "lineage_tree.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written {out}  ({len(html):,} bytes)")


if __name__ == "__main__":
    build()
