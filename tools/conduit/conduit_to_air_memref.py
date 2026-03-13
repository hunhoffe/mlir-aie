#!/usr/bin/env python3
"""conduit_to_air_memref.py
Translate conduit.put_memref / conduit.get_memref forms back to
air.channel.put / air.channel.get with full memref descriptor syntax.

Input : path to a .conduit.mlir file
Output: <same-dir>/backend_roundtrip.mlir
        <same-dir>/backend_roundtrip.mlir.manifest.json
"""
import json, re, sys
from pathlib import Path

# Matches: [%result =] conduit.{put|get}_memref[_async] %chan, %buf {attrs}
CONDUIT_MEMREF_RE = re.compile(
    r'(?P<result>%\S+\s*=\s*)?'
    r'conduit\.(?P<op>put|get)_memref(?P<async_sfx>_async)?\s+'
    r'%(?P<chan>\w+),\s*(?P<buf>%\S+)\s+'
    r'\{(?P<attrs>[^}]*)\}'
    r'(?P<trailing>.*)'
)

def parse_attr(attrs_str, key):
    """Extract value of key=[...] or key=N from an attribute string."""
    list_m = re.search(rf'{key}=\[([^\]]*)\]', attrs_str)
    if list_m:
        vals = [x.strip() for x in list_m.group(1).split(',') if x.strip()]
        return vals
    scalar_m = re.search(rf'{key}=([^\s,}}]+)', attrs_str)
    if scalar_m:
        return scalar_m.group(1)
    return None


def translate_conduit_memref(line, warnings):
    m = CONDUIT_MEMREF_RE.search(line)
    if not m:
        return None

    result_raw = (m.group('result') or '').strip()
    result = result_raw.rstrip('= ').strip()  # '%tok = ' -> '%tok'
    op = m.group('op')
    is_async = bool(m.group('async_sfx'))
    chan = m.group('chan')
    buf = m.group('buf')
    attrs = m.group('attrs')
    trailing = (m.group('trailing') or '').strip()

    offsets = parse_attr(attrs, 'offsets') or []
    sizes   = parse_attr(attrs, 'sizes')   or []
    strides = parse_attr(attrs, 'strides') or []
    deps_raw = parse_attr(attrs, 'deps')
    deps = deps_raw if isinstance(deps_raw, list) else []
    idx_raw = parse_attr(attrs, 'chan_indices')
    idx = idx_raw if isinstance(idx_raw, list) else []
    memref_type_m = re.search(r'memref_type="([^"]*)"', attrs)
    memref_type = memref_type_m.group(1) if memref_type_m else 'memref<?xi32>'
    id_m = re.search(r'\bid=(\d+)', attrs)
    id_attr = f' {{id = {id_m.group(1)} : i32}}' if id_m else ''

    off_str = ', '.join(offsets)
    sz_str  = ', '.join(sizes)
    st_str  = ', '.join(strides)
    idx_str = ', '.join(idx)
    deps_str = ', '.join(deps)

    async_kw = 'async ' if is_async else ''
    deps_bracket = f'[{deps_str}] ' if is_async else ''
    result_prefix = f"{result} = " if result else ""

    indent = len(line) - len(line.lstrip())
    air_line = (
        f'{" " * indent}{result_prefix}'
        f'air.channel.{op} {async_kw}{deps_bracket}'
        f'@{chan}[{idx_str}] '
        f'({buf}[{off_str}] [{sz_str}] [{st_str}])'
        f'{id_attr} : ({memref_type})\n'
    )
    return air_line


def main():
    if len(sys.argv) < 2:
        print("Usage: conduit_to_air_memref.py <input.conduit.mlir>", file=sys.stderr)
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
        if re.search(r'conduit\.(put|get)_memref', line):
            air_line = translate_conduit_memref(line, warnings)
            if air_line:
                out_lines.append(air_line)
                translated += 1
            else:
                warnings.append(f"Could not parse conduit memref op: {line.rstrip()}")
                out_lines.append(line)
        elif 'conduit.annotate' in line:
            # Consume conduit.annotate lines — they were emitted by the forward
            # translator for unknown attrs and have no air.channel equivalent.
            annotate_consumed += 1
        else:
            out_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "ops_translated": translated,
        "annotate_lines_consumed": annotate_consumed,
        "warnings": warnings,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
