#!/usr/bin/env python3
"""conduit_interpreter.py — Conduit DSL interpreter.

Supported commands:
  CREATE <name> <capacity>         — create a named FIFO (capacity ignored for overflow)
  CREATE <name> capacity=<N>       — same, key=value form
  PUT    <name> <val> [<val> ...]  — enqueue one or more integer tokens
  GET    <name>                    — dequeue and print the front token
  PREFILL <name> <val> [...]       — enqueue tokens before streaming begins
  PEEK   <name> <offset>           — non-destructively inspect token at offset (0=front)
  ADVANCE <name> <count>           — consume (discard) count tokens from the front
  ACQUIRE <name> <count>           — print a window of count tokens without consuming
  RELEASE <name> <count>           — consume count tokens (pair with ACQUIRE)
  ANNOTATE <name> <key>=<val>      — no-op hint (lowering metadata, ignored at runtime)

Lines beginning with '//' are comments and are ignored.
"""
import sys
from collections import deque


class Conduit:
    def __init__(self, capacity):
        self.capacity = capacity
        self.q = deque()
        # Tracks tokens currently held under an ACQUIRE (not yet released)
        self._acquired = 0

    def put(self, v):
        self.q.append(v)

    def get(self):
        if not self.q:
            raise RuntimeError("GET on empty conduit")
        return self.q.popleft()

    def peek(self, offset):
        if offset >= len(self.q):
            raise RuntimeError(
                f"PEEK offset={offset} out of range (queue length={len(self.q)})"
            )
        return self.q[offset]

    def advance(self, count):
        for _ in range(count):
            if not self.q:
                raise RuntimeError("ADVANCE on empty conduit")
            self.q.popleft()

    def acquire(self, count):
        """Return a view of the front `count` tokens without consuming them."""
        if count > len(self.q):
            raise RuntimeError(
                f"ACQUIRE {count} tokens but only {len(self.q)} available"
            )
        self._acquired = count
        return list(self.q)[: count]

    def release(self, count):
        """Consume `count` tokens (the previously acquired window)."""
        if count > len(self.q):
            raise RuntimeError(
                f"RELEASE {count} tokens but only {len(self.q)} available"
            )
        for _ in range(count):
            self.q.popleft()
        self._acquired = max(0, self._acquired - count)


conduits = {}

for filepath in sys.argv[1:]:
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw or raw.startswith("//"):
                continue
            parts = raw.split()
            cmd = parts[0].upper()

            try:
                if cmd == "CREATE":
                    name = parts[1]
                    raw_cap = parts[2] if len(parts) > 2 else "16"
                    cap = int(raw_cap.split("=")[-1])
                    conduits[name] = Conduit(cap)

                elif cmd == "ANNOTATE":
                    pass  # lowering hint; no runtime effect

                elif cmd == "PREFILL":
                    name = parts[1]
                    for v in parts[2:]:
                        conduits[name].put(int(v))

                elif cmd == "PUT":
                    name = parts[1]
                    for v in parts[2:]:
                        conduits[name].put(int(v))

                elif cmd == "GET":
                    name = parts[1]
                    val = conduits[name].get()
                    print(f"GET {name} {val}")

                elif cmd == "PEEK":
                    name = parts[1]
                    offset = int(parts[2]) if len(parts) > 2 else 0
                    val = conduits[name].peek(offset)
                    print(f"PEEK {name}[{offset}] = {val}")

                elif cmd == "ADVANCE":
                    name = parts[1]
                    count = int(parts[2]) if len(parts) > 2 else 1
                    conduits[name].advance(count)

                elif cmd == "ACQUIRE":
                    name = parts[1]
                    count = int(parts[2]) if len(parts) > 2 else 1
                    window = conduits[name].acquire(count)
                    print(f"ACQUIRE {name} window={window}")

                elif cmd == "RELEASE":
                    name = parts[1]
                    count = int(parts[2]) if len(parts) > 2 else 1
                    conduits[name].release(count)

                else:
                    # Unknown command: warn but continue
                    print(
                        f"WARNING {filepath}:{lineno}: unknown command {cmd!r}",
                        file=sys.stderr,
                    )

            except (KeyError, RuntimeError, ValueError, IndexError) as e:
                print(
                    f"ERROR {filepath}:{lineno}: {cmd} — {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
