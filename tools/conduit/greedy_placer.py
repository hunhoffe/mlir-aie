#!/usr/bin/env python3
"""greedy_placer.py — Minimal greedy tile placement prototype for Conduit conduits.

Algorithm:
  1. Parse conduit.create ops (or conduit.objectfifo_link memtile hints) from a
     .conduit.mlir file or a JSON manifest.
  2. Maintain a list of MemTiles with capacity budgets.
  3. Assign each conduit to the tile with the most remaining capacity that is
     also compatible with the conduit's annotation (lower_to=objectfifo → AIE tile,
     lower_to=channel → AIR/host tile).
  4. Detect conflicts (capacity overflow, link cycle stubs).
  5. Output placement.assign JSON and a conflict report.

Usage:
  python3 tools/greedy_placer.py <input.conduit.mlir> --output <placement.json>
                                  [--tiles <N>] [--tile-capacity <bytes>]

Output JSON schema:
  {
    "input": "<path>",
    "tiles": [{"id": "tile_0_1", "capacity": 65536, "used": 2048, "conduits": ["c1","c2"]}],
    "placement_assign": {"c1": "tile_0_1", ...},
    "conflicts": [{"type": "overflow", "tile": "tile_0_1", "over_by": 512}],
    "link_utilization": {"tile_0_1": 0.03},
    "summary": "..."
  }
"""
import argparse, json, re, sys
from pathlib import Path
from collections import defaultdict

# Defaults
DEFAULT_TILES    = 4
DEFAULT_CAPACITY = 65536   # 64 KB per MemTile data memory segment

CREATE_RE  = re.compile(r'conduit\.objectfifo_link\s+.*?memtile="([^"]+)"', re.DOTALL)
# COND_RE matches DSL form (CREATE/conduit.create) and AIR form (air.channel @name [dims]).
# For air.channel, group(4) captures all bracket dimension groups, e.g. "[2][4]".
COND_RE    = re.compile(
    r'(?:'
    r'(?:CREATE|conduit\.create)\s+@?(\w+)\s+(?:capacity=)?(\d+)'  # DSL/conduit form
    r'|'
    r'air\.channel\s+@(\w+)((?:\s*\[\d+\])*)'                      # AIR channel decl
    r')'
)
LINK_RE    = re.compile(
    r'conduit\.objectfifo_link\s+srcs?=\[([^\]]*)\].*?dsts=\[([^\]]*)\]'
    r'.*?memtile="([^"]*)"', re.DOTALL)
# ANNOT_RE matches DSL form and MLIR attr-dict form.
ANNOT_RE   = re.compile(
    r'(?:'
    r'ANNOTATE\s+(\w+)\s+lower_to=(\w+)'                             # DSL form
    r'|'
    r'conduit\.annotate\s*\{[^}]*name\s*=\s*"(\w+)"[^}]*key\s*=\s*"lower_to"[^}]*value\s*=\s*"(\w+)"'  # MLIR attr-dict
    r')'
)


def parse_conduit_file(path):
    text = Path(path).read_text(errors='replace')
    conduits = {}  # name -> {"capacity": N, "lower_to": "objectfifo"|"channel"}
    for m in COND_RE.finditer(text):
        if m.group(1) is not None:
            # DSL/conduit.create form: groups 1 (name) and 2 (capacity)
            conduits[m.group(1)] = {"capacity": int(m.group(2)), "lower_to": "objectfifo"}
        elif m.group(3) is not None:
            # air.channel @name [D0][D1]... form: group 3 (name), group 4 (bracket dims).
            # Compute capacity as the product of all dimension values.
            dims_str = m.group(4) or ""
            dims = [int(d) for d in re.findall(r'\[(\d+)\]', dims_str)]
            capacity = 1
            for d in dims:
                capacity *= d
            conduits[m.group(3)] = {"capacity": capacity, "lower_to": "channel"}
    for m in ANNOT_RE.finditer(text):
        if m.group(1) is not None:
            # DSL form: groups 1 (name) and 2 (lower_to value)
            name, val = m.group(1), m.group(2)
        else:
            # MLIR attr-dict form: groups 3 (name) and 4 (lower_to value)
            name, val = m.group(3), m.group(4)
        if name in conduits:
            conduits[name]["lower_to"] = val
    links = []
    for m in LINK_RE.finditer(text):
        srcs = [x.strip().lstrip('%') for x in m.group(1).split(',') if x.strip()]
        dsts = [x.strip().lstrip('%') for x in m.group(2).split(',') if x.strip()]
        links.append({"srcs": srcs, "dsts": dsts, "memtile": m.group(3)})
    return conduits, links


def make_tiles(n, capacity):
    return [{"id": f"tile_0_{i+1}", "capacity": capacity, "used": 0, "conduits": []}
            for i in range(n)]


def greedy_assign(conduits, links, tiles):
    """Assign conduits to tiles greedily by remaining capacity."""
    assignment = {}

    # Pre-assign conduits mentioned in links to the link's memtile
    hint = {}
    for link in links:
        mt = link["memtile"]
        for c in link["srcs"] + link["dsts"]:
            hint[c] = mt

    def tile_by_id(tid):
        for t in tiles:
            if t["id"] == tid:
                return t
        return None

    def best_tile(cname, lower_to):
        # Respect memtile hint if given
        if cname in hint:
            t = tile_by_id(hint[cname])
            if t:
                return t
        # Prefer AIE tiles (odd index) for objectfifo, AIR tiles (even) for channel
        candidates = sorted(tiles, key=lambda t: -( t["capacity"] - t["used"]))
        for t in candidates:
            if lower_to == "objectfifo" and int(t["id"].split('_')[-1]) % 2 == 1:
                return t
            if lower_to == "channel" and int(t["id"].split('_')[-1]) % 2 == 0:
                return t
        return candidates[0] if candidates else None

    for cname, info in conduits.items():
        cap = info["capacity"] * 4  # bytes (assume i32 elements)
        t = best_tile(cname, info.get("lower_to", "objectfifo"))
        if t is None:
            assignment[cname] = "unassigned"
            continue
        t["used"] += cap
        t["conduits"].append(cname)
        assignment[cname] = t["id"]

    return assignment


def detect_conflicts(tiles):
    conflicts = []
    for t in tiles:
        if t["used"] > t["capacity"]:
            conflicts.append({
                "type": "overflow",
                "tile": t["id"],
                "capacity": t["capacity"],
                "used": t["used"],
                "over_by": t["used"] - t["capacity"],
            })
    return conflicts


def compute_utilization(tiles):
    return {t["id"]: round(t["used"] / t["capacity"], 4) if t["capacity"] else 0 for t in tiles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=None, help="Input .conduit.mlir file")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tiles", type=int, default=DEFAULT_TILES)
    ap.add_argument("--tile-capacity", type=int, default=DEFAULT_CAPACITY)
    ap.add_argument("--input-json", default=None,
                    help="JSON manifest with conduits list instead of parsing a file")
    args = ap.parse_args()

    if args.input_json:
        data = json.loads(Path(args.input_json).read_text())
        conduits = {c["name"]: {"capacity": c["capacity"], "lower_to": c.get("lower_to","objectfifo")}
                    for c in data.get("conduits", [])}
        links = data.get("links", [])
        input_label = args.input_json
    elif args.input:
        conduits, links = parse_conduit_file(args.input)
        input_label = args.input
    else:
        # No input: synthesize minimal demo conduits
        conduits = {
            "c_in":   {"capacity": 2048, "lower_to": "objectfifo"},
            "c_mid":  {"capacity": 1024, "lower_to": "objectfifo"},
            "c_out0": {"capacity": 1024, "lower_to": "channel"},
            "c_out1": {"capacity": 1024, "lower_to": "channel"},
        }
        links = [{"srcs": ["c_in"], "dsts": ["c_mid"], "memtile": "tile_0_1"}]
        input_label = "<synthetic>"

    tiles = make_tiles(args.tiles, args.tile_capacity)
    assignment = greedy_assign(conduits, links, tiles)
    conflicts  = detect_conflicts(tiles)
    utilization = compute_utilization(tiles)

    valid_assignments = sum(1 for v in assignment.values() if v != "unassigned")
    result = {
        "input": input_label,
        "num_tiles": args.tiles,
        "tile_capacity_bytes": args.tile_capacity,
        "tiles": tiles,
        "placement_assign": assignment,
        "conflicts": conflicts,
        "link_utilization": utilization,
        "conduits_placed": valid_assignments,
        "conduits_total": len(conduits),
        "summary": (
            f"Placed {valid_assignments}/{len(conduits)} conduits across {args.tiles} tiles. "
            f"Conflicts: {len(conflicts)}. "
            f"Peak tile utilization: {max(utilization.values(), default=0):.1%}."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if len(conflicts) == 0 else 2)


if __name__ == "__main__":
    main()
