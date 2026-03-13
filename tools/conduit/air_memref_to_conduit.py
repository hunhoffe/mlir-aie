#!/usr/bin/env python3
"""air_memref_to_conduit.py
Translate air.channel.put/get calls with memref descriptors into
conduit.put_memref / conduit.get_memref forms.

Input : path to an AIR MLIR file
Output: <same-dir>/<stem>.conduit.mlir
        <same-dir>/<stem>.conduit.mlir.manifest.json

Descriptor form recognized:
  [%tok =] air.channel.put [async [deps]] @chan[idx...] (%buf[off...] [sz...] [st...]) {attrs} : (memref<...>)
  [%tok =] air.channel.get [async [deps]] @chan[idx...] (%buf[off...] [sz...] [st...]) {attrs} : (memref<...>)

Unknown attributes not in the known set (id, async, offsets/sizes/strides) are
preserved verbatim as conduit.annotate entries on the line immediately following
the translated op. This ensures no metadata is silently dropped.
"""
import json, re, sys
from pathlib import Path

# Matches optional result, async deps, channel name+indices, and memref descriptor
PUT_RE = re.compile(
    r'(?P<result>%\S+\s*=\s*)?'
    r'air\.channel\.(?P<op>put|get)\s+'
    r'(?P<async_kw>async\s*)?'
    r'(?:\[(?P<deps>[^\]]*)\]\s*)?'
    r'@(?P<chan>\w+)\[(?P<idx>[^\]]*)\]\s+'
    r'\((?P<buf>[^[]+)\[(?P<offsets>[^\]]*)\]\s*\[(?P<sizes>[^\]]*)\]\s*\[(?P<strides>[^\]]*)\]\)'
    r'(?P<tail>[^:]*)'
    r'(?::\s*\((?P<memref>[^)]*)\))?'
    r'(?P<trailing>.*)'
)


def parse_list(s):
    return [x.strip() for x in s.split(',') if x.strip()]


def translate_put(line, warnings):
    m = PUT_RE.search(line)
    if not m:
        return None

    # Strip trailing '=' and whitespace so result holds just the SSA name (e.g. '%tok')
    result_raw = (m.group('result') or '').strip()
    result = result_raw.rstrip('= ').strip()  # '%tok = ' -> '%tok'
    op = m.group('op')        # put or get
    is_async = bool(m.group('async_kw'))
    deps = parse_list(m.group('deps') or '')
    chan = m.group('chan')
    idx = parse_list(m.group('idx') or '')
    buf = m.group('buf').strip()
    offsets = parse_list(m.group('offsets') or '')
    sizes = parse_list(m.group('sizes') or '')
    strides = parse_list(m.group('strides') or '')
    tail = (m.group('tail') or '').strip()
    memref_type = (m.group('memref') or '').strip()
    trailing = (m.group('trailing') or '').strip()

    # Compute num_elems from sizes.
    # If any value is symbolic (e.g. %c64_new, %arg0) preserve '?' verbatim
    # without emitting a warning — symbolic values are expected and valid.
    def is_numeric(v):
        try:
            int(v)
            return True
        except ValueError:
            return False

    def extract_const(v):
        """Try %cN or %cN_suffix -> N, else raise."""
        m = re.match(r'%c(\d+)(?:_\w+)?$', v)
        if m:
            return int(m.group(1))
        return int(v)

    try:
        num_elems = 1
        for s in sizes:
            num_elems *= extract_const(s)
    except Exception:
        num_elems = '?'  # symbolic — preserved verbatim, no warning needed

    # Build conduit op name
    conduit_op = f"conduit.{'put' if op == 'put' else 'get'}_memref"
    if is_async:
        conduit_op += "_async"

    # Build attribute dict
    attr_parts = [
        f"offsets=[{', '.join(offsets)}]",
        f"sizes=[{', '.join(sizes)}]",
        f"strides=[{', '.join(strides)}]",
        f"num_elems={num_elems}",
        f"chan_indices=[{', '.join(idx)}]",
    ]
    if deps:
        attr_parts.append(f"deps=[{', '.join(deps)}]")
    if memref_type:
        attr_parts.append(f"memref_type=\"{memref_type}\"")
    if tail:
        # pass through any {id = N} attrs
        id_m = re.search(r'id\s*=\s*(\d+)', tail)
        if id_m:
            attr_parts.append(f"id={id_m.group(1)}")

    # Collect unknown attributes from {attrs} block (anything not id=, not already captured)
    known_keys = {'id', 'async', 'offsets', 'sizes', 'strides', 'num_elems',
                  'chan_indices', 'deps', 'memref_type'}
    unknown_attrs = []
    brace_m = re.search(r'\{([^}]*)\}', tail)
    if brace_m:
        for kv in brace_m.group(1).split(','):
            kv = kv.strip()
            key = kv.split('=')[0].strip().split()[-1] if '=' in kv else kv.split()[0] if kv else ''
            if key and key not in known_keys:
                unknown_attrs.append(kv)

    indent = len(line) - len(line.lstrip())
    sp = ' ' * indent
    result_prefix = f"{result} = " if result else ""
    conduit_line = (
        f'{sp}{result_prefix}{conduit_op} %{chan}, {buf} '
        f'{{{", ".join(attr_parts)}}}\n'
    )
    # Emit unknown attributes as conduit.annotate lines
    for ua in unknown_attrs:
        conduit_line += f'{sp}conduit.annotate unknown_attr="{ua}"\n'
    return conduit_line


def main():
    if len(sys.argv) < 2:
        print("Usage: air_memref_to_conduit.py <input.mlir>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.parent / (input_path.stem + '.conduit.mlir')
    manifest_path = output_path.with_suffix('.mlir.manifest.json')

    with open(input_path) as f:
        lines = f.readlines()

    def is_test_harness(line):
        s = line.lstrip()
        if s.startswith('// CHECK') or '// CHECK:' in line:
            return True
        if '{{' in line and '}}' in line:
            return True
        return False

    out_lines = []
    warnings = []
    translated = 0
    for line in lines:
        # Skip FileCheck / test-harness lines silently
        if is_test_harness(line):
            out_lines.append(line)
            continue
        if re.search(r'air\.channel\.(put|get)', line):
            conduit_line = translate_put(line, warnings)
            if conduit_line:
                out_lines.append(conduit_line)
                translated += 1
            else:
                warnings.append(f"Could not parse air.channel op: {line.rstrip()}")
                out_lines.append(line)
        else:
            out_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "ops_translated": translated,
        "warnings": warnings,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
