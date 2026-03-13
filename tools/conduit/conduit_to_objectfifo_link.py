#!/usr/bin/env python3
"""conduit_to_objectfifo_link.py
Translate conduit.objectfifo_link lines back to aie.objectfifo.link.

Input : path to a .conduit.mlir file
Output: <same-dir>/backend_roundtrip.mlir
        <same-dir>/backend_roundtrip.mlir.manifest.json
"""
import json, re, sys
from pathlib import Path

# Pattern: conduit.objectfifo_link src=[...] dsts=[...] mode="..." memtile="..." offsets=[...] [lock_id=N]
CONDUIT_LINK_RE = re.compile(
    r'conduit\.objectfifo_link\s+'
    r'src=\[(?P<srcs>[^\]]*)\]\s+'
    r'dsts=\[(?P<dsts>[^\]]*)\]\s+'
    r'mode="(?P<mode>[^"]+)"\s+'
    r'memtile="(?P<memtile>[^"]+)"\s+'
    r'offsets=\[(?P<offsets>[^\]]*)\]'
    r'(?P<tail>.*)'
)


def parse_names(s):
    return [x.strip().lstrip('%') for x in s.split(',') if x.strip()]


def translate_conduit_link(line, warnings):
    m = CONDUIT_LINK_RE.search(line)
    if not m:
        return None
    srcs = parse_names(m.group('srcs'))
    dsts = parse_names(m.group('dsts'))
    mode = m.group('mode')
    offsets = [x.strip() for x in m.group('offsets').split(',') if x.strip()]
    tail = m.group('tail').strip()

    # Reconstruct lock_id attribute if present
    lock_attr = ""
    lock_m = re.search(r'lock_id=(\d+)', tail)
    if lock_m:
        lock_attr = f" {{lock_id = {lock_m.group(1)} : i32}}"

    src_str = ', '.join(f'@{s}' for s in srcs)
    dst_str = ', '.join(f'@{d}' for d in dsts)
    offset_str = ', '.join(offsets)

    if mode == "distribute":
        join_part = ''
        dist_part = offset_str
    else:  # join
        join_part = offset_str
        dist_part = ''

    indent = len(line) - len(line.lstrip())
    aie_line = (
        f'{" " * indent}aie.objectfifo.link '
        f'[{src_str}] -> [{dst_str}] '
        f'([{join_part}] [{dist_part}])'
        f'{lock_attr}\n'
    )
    return aie_line


def main():
    if len(sys.argv) < 2:
        print("Usage: conduit_to_objectfifo_link.py <input.conduit.mlir>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.parent / 'backend_roundtrip.mlir'
    manifest_path = output_path.with_suffix('.mlir.manifest.json')

    with open(input_path) as f:
        lines = f.readlines()

    out_lines = []
    warnings = []
    translated = 0
    annotate_consumed = 0
    for line in lines:
        if 'conduit.objectfifo_link' in line:
            aie_line = translate_conduit_link(line, warnings)
            if aie_line:
                out_lines.append(aie_line)
                translated += 1
            else:
                warnings.append(f"Could not parse conduit.objectfifo_link: {line.rstrip()}")
                out_lines.append(line)
        elif 'conduit.annotate' in line:
            # Consume conduit.annotate lines — they were emitted by the forward
            # translator for unknown attrs and have no aie.objectfifo.link equivalent.
            annotate_consumed += 1
        else:
            out_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "links_translated": translated,
        "annotate_lines_consumed": annotate_consumed,
        "warnings": warnings,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
