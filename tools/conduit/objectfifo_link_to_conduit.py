#!/usr/bin/env python3
"""objectfifo_link_to_conduit.py
Translate AIE objectfifo.link lines into conduit.objectfifo_link lines.

Input : path to an AIE MLIR file
Output: <same-dir>/<stem>.conduit.mlir
        <same-dir>/<stem>.conduit.mlir.manifest.json

Attributes preserved: src, dsts, memtile (heuristic from surrounding
aie.objectfifo declarations), offsets (from bracket arguments), lock_id.
Unknown attributes (e.g. repeat_count, via_DMA, disable_synchronization)
are emitted verbatim as conduit.annotate entries on the following line.
Non-matching lines are copied unchanged; a warning is recorded.
"""
import json, re, sys
from pathlib import Path

# Pattern: aie.objectfifo.link [@src0,...] -> [@dst0,...] ([join_offs] [dist_offs])
LINK_RE = re.compile(
    r'aie\.objectfifo\.link\s+'
    r'\[(?P<srcs>[^\]]*)\]\s*->\s*\[(?P<dsts>[^\]]*)\]'
    r'\s*\(\[(?P<join>[^\]]*)\]\s*\[(?P<dist>[^\]]*)\]\)'
    r'(?P<tail>.*)'
)

# Pattern: aie.objectfifo @name(...) — used to extract memtile hints
FIFO_RE = re.compile(
    r'aie\.objectfifo\s+@(?P<name>\w+)\s*\('
    r'(?P<producer>[^,]+),\s*\{(?P<consumers>[^}]*)\}'
)


def parse_names(s):
    return [x.strip().lstrip('@') for x in s.split(',') if x.strip()]


def detect_mode(srcs, dsts):
    if len(srcs) == 1 and len(dsts) >= 1:
        return "distribute"
    if len(srcs) > 1 and len(dsts) == 1:
        return "join"
    return "distribute"  # default


def translate_link(line, fifo_tiles, warnings):
    m = LINK_RE.search(line)
    if not m:
        return None, None
    srcs = parse_names(m.group('srcs'))
    dsts = parse_names(m.group('dsts'))
    join_offs = [x.strip() for x in m.group('join').split(',') if x.strip()]
    dist_offs = [x.strip() for x in m.group('dist').split(',') if x.strip()]
    tail = m.group('tail').strip()

    mode = detect_mode(srcs, dsts)
    offsets = dist_offs if mode == "distribute" else join_offs

    # Heuristic: memtile = producer tile of first dst (relay tile)
    memtile = "unknown"
    if dsts and dsts[0] in fifo_tiles:
        memtile = fifo_tiles[dsts[0]].get('producer', 'unknown')
    elif srcs and srcs[0] in fifo_tiles:
        memtile = fifo_tiles[srcs[0]].get('consumers', ['unknown'])[0] \
            if fifo_tiles[srcs[0]].get('consumers') else 'unknown'

    # Extract lock_id from tail if present
    lock_id_attr = ""
    lock_m = re.search(r'lock_id\s*=\s*(\d+)', tail)
    if lock_m:
        lock_id_attr = f" lock_id={lock_m.group(1)}"

    # Collect unknown attributes from tail (anything that isn't lock_id)
    known_attr_re = re.compile(r'lock_id\s*=\s*\d+')
    unknown_tail = known_attr_re.sub('', tail).strip().strip('{}').strip()
    # Also capture {attr = val} blocks from the original line not in tail
    brace_attrs = re.findall(r'\{([^}]+)\}', line)
    extra_attrs = []
    for ba in brace_attrs:
        for kv in ba.split(','):
            kv = kv.strip()
            if kv and not re.match(r'lock_id\s*=', kv):
                extra_attrs.append(kv.strip())
    if unknown_tail:
        extra_attrs.append(unknown_tail)

    src_list = ', '.join(f'%{s}' for s in srcs)
    dst_list = ', '.join(f'%{d}' for d in dsts)
    offset_str = f'[{", ".join(offsets)}]' if offsets else '[]'

    indent = len(line) - len(line.lstrip())
    sp = ' ' * indent
    conduit_line = (
        f'{sp}conduit.objectfifo_link '
        f'src=[{src_list}] '
        f'dsts=[{dst_list}] '
        f'mode="{mode}" '
        f'memtile="{memtile}" '
        f'offsets={offset_str}'
        f'{lock_id_attr}\n'
    )
    # Emit unknown attributes as conduit.annotate lines
    annotate_lines = ''
    for attr in extra_attrs:
        annotate_lines += f'{sp}conduit.annotate unknown_attr="{attr}"\n'
    return conduit_line + annotate_lines, (srcs, dsts, mode, offsets, memtile)


def main():
    if len(sys.argv) < 2:
        print("Usage: objectfifo_link_to_conduit.py <input.mlir>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.parent / (input_path.stem + '.conduit.mlir')
    manifest_path = output_path.with_suffix('.mlir.manifest.json')

    with open(input_path) as f:
        lines = f.readlines()

    def is_test_harness(line):
        """Return True for FileCheck / test-harness lines that must be skipped."""
        s = line.lstrip()
        if s.startswith('// CHECK') or '// CHECK:' in line:
            return True
        if '{{' in line and '}}' in line:   # FileCheck wildcard syntax
            return True
        return False

    # First pass: collect fifo tile mappings (skip test harness lines)
    fifo_tiles = {}
    for line in lines:
        if is_test_harness(line):
            continue
        m = FIFO_RE.search(line)
        if m:
            name = m.group('name')
            producer = m.group('producer').strip()
            consumers = [c.strip().lstrip('%') for c in m.group('consumers').split(',')]
            fifo_tiles[name] = {'producer': producer, 'consumers': consumers}

    # Second pass: translate
    out_lines = []
    warnings = []
    translated = 0
    for line in lines:
        # Skip FileCheck / test-harness lines silently
        if is_test_harness(line):
            out_lines.append(line)
            continue
        if 'aie.objectfifo.link' in line:
            conduit_line, meta = translate_link(line, fifo_tiles, warnings)
            if conduit_line:
                out_lines.append(conduit_line)
                translated += 1
            else:
                warnings.append(f"Could not parse objectfifo.link: {line.rstrip()}")
                out_lines.append(line)
        else:
            out_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "links_translated": translated,
        "warnings": warnings,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
