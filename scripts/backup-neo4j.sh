#!/usr/bin/env bash
# Backup the Serenade Neo4j world-state graph.
#
# Solo-operator disaster recovery (Phase M1). Prefers `neo4j-admin
# database dump` (offline-style dump that restores fastest), falling
# back to a `cypher-shell`-driven APOC export when neo4j-admin isn't on
# PATH (community-edition containers).
#
# Usage:  scripts/backup-neo4j.sh [output-dir]
#         (default output dir is ./backups/neo4j/)

set -euo pipefail

OUT_DIR="${1:-./backups/neo4j}"
mkdir -p "$OUT_DIR"

if [ -f .env ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASS="${NEO4J_PASSWORD:-}"
NEO4J_DB="${NEO4J_DATABASE:-serenade}"
NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"

TS="$(date -u +%Y-%m-%d-%H%M%S)"
OUT_FILE="${OUT_DIR%/}/serenade-neo4j-${TS}.dump.gz"

START_TS="$(date +%s)"

ERR_LOG="$(mktemp)"
trap 'rm -f "$ERR_LOG"' EXIT

if command -v neo4j-admin >/dev/null 2>&1; then
    # Offline dump tier. neo4j-admin writes to a directory; we pipe the
    # resulting file through gzip and into our timestamped name.
    TMP_DIR="$(mktemp -d)"
    if ! neo4j-admin database dump "$NEO4J_DB" --to-path="$TMP_DIR" 2>"$ERR_LOG"; then
        echo "neo4j-admin dump failed:" >&2
        cat "$ERR_LOG" >&2
        rm -rf "$TMP_DIR"
        exit 1
    fi
    DUMP_FILE="$(find "$TMP_DIR" -type f -name '*.dump' | head -n1)"
    if [ -z "$DUMP_FILE" ]; then
        echo "neo4j-admin produced no .dump file in $TMP_DIR" >&2
        rm -rf "$TMP_DIR"
        exit 1
    fi
    gzip -c "$DUMP_FILE" > "$OUT_FILE"
    rm -rf "$TMP_DIR"
else
    # APOC fallback. Requires the running Neo4j container to have APOC
    # installed (the bundled Serenade compose does). Output is a
    # cypher-shell replayable script — restore is `cypher-shell < file`.
    OUT_FILE="${OUT_DIR%/}/serenade-neo4j-${TS}.cypher.gz"
    if ! command -v cypher-shell >/dev/null 2>&1; then
        echo "neither neo4j-admin nor cypher-shell on PATH; cannot dump Neo4j" >&2
        exit 1
    fi
    if ! cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASS" \
        "CALL apoc.export.cypher.all(null, {format:'cypher-shell', stream:true}) YIELD cypherStatements RETURN cypherStatements" \
        2>"$ERR_LOG" | gzip -9 > "$OUT_FILE"; then
        echo "cypher-shell APOC export failed:" >&2
        cat "$ERR_LOG" >&2
        rm -f "$OUT_FILE"
        exit 1
    fi
fi

END_TS="$(date +%s)"
DURATION=$((END_TS - START_TS))

if SIZE="$(du -h "$OUT_FILE" 2>/dev/null | cut -f1)" && [ -n "$SIZE" ]; then
    :
else
    SIZE="$(stat -c%s "$OUT_FILE" 2>/dev/null || echo "?")B"
fi

# Daily rotation: drop non-Sunday backups older than 7d.
find "$OUT_DIR" -maxdepth 1 -type f \( -name 'serenade-neo4j-*.dump.gz' -o -name 'serenade-neo4j-*.cypher.gz' \) -mtime +7 -print0 \
    | while IFS= read -r -d '' f; do
        BASE="$(basename "$f")"
        DATE_PART="${BASE#serenade-neo4j-}"
        DATE_PART="${DATE_PART%%-[0-9][0-9][0-9][0-9][0-9][0-9]*}"
        DOW="$(date -d "$DATE_PART" +%u 2>/dev/null || echo 0)"
        if [ "$DOW" = "7" ]; then
            continue
        fi
        rm -f "$f"
    done

# Weekly rotation: drop Sunday backups older than 28d.
find "$OUT_DIR" -maxdepth 1 -type f \( -name 'serenade-neo4j-*.dump.gz' -o -name 'serenade-neo4j-*.cypher.gz' \) -mtime +28 -print0 \
    | while IFS= read -r -d '' f; do
        BASE="$(basename "$f")"
        DATE_PART="${BASE#serenade-neo4j-}"
        DATE_PART="${DATE_PART%%-[0-9][0-9][0-9][0-9][0-9][0-9]*}"
        DOW="$(date -d "$DATE_PART" +%u 2>/dev/null || echo 0)"
        if [ "$DOW" = "7" ]; then
            rm -f "$f"
        fi
    done

RETAINED="$(find "$OUT_DIR" -maxdepth 1 -type f \( -name 'serenade-neo4j-*.dump.gz' -o -name 'serenade-neo4j-*.cypher.gz' \) | wc -l | tr -d ' ')"
echo "Backed up to ${OUT_FILE}, ${SIZE}, ${DURATION}s. Older backups: ${RETAINED} retained."
