#!/bin/bash
# compare_ir_outputs.sh — Compare Conduit lowering pipeline against the
# stateful objectfifo transform for a given MLIR input file.
#
# Usage:
#   ./tools/compare_ir_outputs.sh <aie-opt-path> <mlir-file>
#
# Output:
#   A table showing op counts for each path, with MATCH/DIFF indicators.
#   Exits 0 if all counts match, exits 1 if any differ.
#
# Op types compared:
#   aie.buffer    aie.lock    aie.dma_bd    aie.flow    aie.use_lock
#
# Notes:
# - Comment lines (starting with //) are excluded from counts.
# - Both paths are run with --mlir-print-debuginfo suppressed to normalize output.
# - If either path crashes/errors, the failure is reported as CRASH with count -1.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <aie-opt-path> <mlir-file>" >&2
    exit 2
fi

AIE_OPT="$1"
MLIR_FILE="$2"

if [ ! -f "$AIE_OPT" ]; then
    echo "ERROR: aie-opt not found at: $AIE_OPT" >&2
    exit 2
fi

if [ ! -f "$MLIR_FILE" ]; then
    echo "ERROR: MLIR file not found: $MLIR_FILE" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Run both lowering paths and capture output
# ---------------------------------------------------------------------------

CONDUIT_OUT=$(mktemp /tmp/conduit_out.XXXXXX)
STATEFUL_OUT=$(mktemp /tmp/stateful_out.XXXXXX)
CONDUIT_ERR=$(mktemp /tmp/conduit_err.XXXXXX)
STATEFUL_ERR=$(mktemp /tmp/stateful_err.XXXXXX)

trap "rm -f '$CONDUIT_OUT' '$STATEFUL_OUT' '$CONDUIT_ERR' '$STATEFUL_ERR'" EXIT

CONDUIT_EXIT=0
STATEFUL_EXIT=0

"$AIE_OPT" --objectfifo-to-conduit --conduit-to-dma "$MLIR_FILE" \
    > "$CONDUIT_OUT" 2>"$CONDUIT_ERR" || CONDUIT_EXIT=$?

"$AIE_OPT" --aie-objectFifo-stateful-transform "$MLIR_FILE" \
    > "$STATEFUL_OUT" 2>"$STATEFUL_ERR" || STATEFUL_EXIT=$?

# ---------------------------------------------------------------------------
# Count op occurrences (excluding comment lines)
# ---------------------------------------------------------------------------

count_op() {
    local file="$1"
    local op="$2"
    local exit_code="$3"
    if [ "${exit_code}" != "0" ]; then
        echo "-1"
        return
    fi
    # Filter out comment lines (//) before counting; grep -c exits 1 on no
    # match so we catch that and return 0.
    local n
    n=$(grep -v '^\s*//' "$file" | grep -c "${op}" 2>/dev/null) || n=0
    echo "${n}"
}

OP_TYPES=("aie.buffer" "aie.lock" "aie.dma_bd" "aie.next_bd" "aie.flow" "aie.use_lock")

declare -A STATEFUL_COUNTS
declare -A CONDUIT_COUNTS

for op in "${OP_TYPES[@]}"; do
    STATEFUL_COUNTS[$op]=$(count_op "$STATEFUL_OUT" "$op" "$STATEFUL_EXIT")
    CONDUIT_COUNTS[$op]=$(count_op "$CONDUIT_OUT" "$op" "$CONDUIT_EXIT")
done

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

BASENAME=$(basename "$MLIR_FILE")
echo "================================================================"
echo "IR comparison: $BASENAME"
echo "================================================================"
printf "%-20s  %12s  %12s  %s\n" "op_type" "stateful" "conduit" "result"
echo "----------------------------------------------------------------"

ALL_MATCH=true

for op in "${OP_TYPES[@]}"; do
    sc="${STATEFUL_COUNTS[$op]}"
    cc="${CONDUIT_COUNTS[$op]}"

    # Use arithmetic context to compare; guard crash sentinels first.
    if [ "${sc}" = "-1" ] && [ "${cc}" = "-1" ]; then
        result="BOTH_CRASH"
        ALL_MATCH=false
    elif [ "${sc}" = "-1" ]; then
        result="STATEFUL_CRASH"
        ALL_MATCH=false
    elif [ "${cc}" = "-1" ]; then
        result="CONDUIT_CRASH"
        ALL_MATCH=false
    elif [ "${sc}" -eq "${cc}" ]; then
        result="MATCH"
    else
        delta=$(( cc - sc ))
        result="DIFF (delta=${delta})"
        ALL_MATCH=false
    fi

    printf "%-20s  %12s  %12s  %s\n" "$op" "$sc" "$cc" "$result"
done

echo "----------------------------------------------------------------"

# Print error summaries if either path failed
if [ "$CONDUIT_EXIT" -ne 0 ]; then
    echo ""
    echo "CONDUIT PATH ERROR (exit $CONDUIT_EXIT):"
    head -5 "$CONDUIT_ERR" | sed 's/^/  /'
fi

if [ "$STATEFUL_EXIT" -ne 0 ]; then
    echo ""
    echo "STATEFUL PATH ERROR (exit $STATEFUL_EXIT):"
    head -5 "$STATEFUL_ERR" | sed 's/^/  /'
fi

echo ""
if $ALL_MATCH; then
    echo "RESULT: PASS — all op counts match"
    exit 0
else
    echo "RESULT: FAIL — one or more op counts differ"
    exit 1
fi
