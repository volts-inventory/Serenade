#!/usr/bin/env bash
# Backup the Serenade Postgres database to a gzipped dump.
#
# Solo-operator disaster recovery (Phase M1). Reads connection info from
# `SERENADE_DATABASE_URL` (preferred) or the legacy POSTGRES_* vars from
# .env. Rotates the output dir down to "last 7 daily + last 4 weekly".
#
# Usage:  scripts/backup-postgres.sh [output-dir]
#         (default output dir is ./backups/postgres/)

set -euo pipefail

OUT_DIR="${1:-./backups/postgres}"
mkdir -p "$OUT_DIR"

# Load .env if present so POSTGRES_* are available. Run in a subshell so
# we don't pollute the caller's environment with everything in .env.
if [ -f .env ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# Resolve the pg_dump connection string. `SERENADE_DATABASE_URL` wins.
# Otherwise compose from the legacy POSTGRES_* split. We deliberately
# default the host to 127.0.0.1 (not the docker-internal "postgres") so
# the script works from the host shell against a compose-published port.
DB_URL="${SERENADE_DATABASE_URL:-}"
if [ -z "$DB_URL" ]; then
    PGHOST="${POSTGRES_HOST:-127.0.0.1}"
    # When .env names the docker-network host, swap to localhost so the
    # script works from the operator's shell (the laptop-recovery path).
    if [ "$PGHOST" = "postgres" ]; then
        PGHOST="127.0.0.1"
    fi
    PGPORT="${POSTGRES_PORT:-5432}"
    PGDB="${POSTGRES_DB:-serenade}"
    PGUSER="${POSTGRES_USER:-serenade}"
    PGPASS="${POSTGRES_PASSWORD:-}"
    DB_URL="postgresql://${PGUSER}:${PGPASS}@${PGHOST}:${PGPORT}/${PGDB}"
fi

TS="$(date -u +%Y-%m-%d-%H%M%S)"
OUT_FILE="${OUT_DIR%/}/serenade-postgres-${TS}.sql.gz"

START_TS="$(date +%s)"

# Capture pg_dump stderr so a failure can surface the underlying error.
# `set -o pipefail` (above) makes the pipeline return pg_dump's exit code
# if it fails, even though gzip succeeds on a partial stream.
ERR_LOG="$(mktemp)"
trap 'rm -f "$ERR_LOG"' EXIT

if ! pg_dump --no-owner --no-privileges --format=plain "$DB_URL" 2>"$ERR_LOG" | gzip -9 > "$OUT_FILE"; then
    echo "pg_dump failed:" >&2
    cat "$ERR_LOG" >&2
    rm -f "$OUT_FILE"
    exit 1
fi

END_TS="$(date +%s)"
DURATION=$((END_TS - START_TS))

# Size in human-readable form. Falls back to raw bytes if `du -h` isn't
# available (some minimal busybox shells).
if SIZE="$(du -h "$OUT_FILE" 2>/dev/null | cut -f1)" && [ -n "$SIZE" ]; then
    :
else
    SIZE="$(stat -c%s "$OUT_FILE" 2>/dev/null || echo "?")B"
fi

# Rotation:
#   1. Delete dailies older than 7d EXCEPT the Sunday weekly snapshots.
#   2. Delete Sunday weeklies older than 28d.
# Filenames embed an ISO date so we can shell-match the Sunday set.
# `find -name "*-Sun-*"` won't match (we use ISO), so we compute the
# weekday from the filename via a small loop instead.
find "$OUT_DIR" -maxdepth 1 -type f -name 'serenade-postgres-*.sql.gz' -mtime +7 -print0 \
    | while IFS= read -r -d '' f; do
        # Extract the YYYY-MM-DD portion from the filename.
        BASE="$(basename "$f")"
        DATE_PART="${BASE#serenade-postgres-}"
        DATE_PART="${DATE_PART%%-[0-9][0-9][0-9][0-9][0-9][0-9]*}"
        # Skip Sunday snapshots (the weekly tier). `date -d` returns the
        # weekday number; 0/7 = Sunday on GNU date.
        DOW="$(date -d "$DATE_PART" +%u 2>/dev/null || echo 0)"
        if [ "$DOW" = "7" ]; then
            continue
        fi
        rm -f "$f"
    done

# Sunday weeklies older than 28 days are pruned.
find "$OUT_DIR" -maxdepth 1 -type f -name 'serenade-postgres-*.sql.gz' -mtime +28 -print0 \
    | while IFS= read -r -d '' f; do
        BASE="$(basename "$f")"
        DATE_PART="${BASE#serenade-postgres-}"
        DATE_PART="${DATE_PART%%-[0-9][0-9][0-9][0-9][0-9][0-9]*}"
        DOW="$(date -d "$DATE_PART" +%u 2>/dev/null || echo 0)"
        if [ "$DOW" = "7" ]; then
            rm -f "$f"
        fi
    done

RETAINED="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'serenade-postgres-*.sql.gz' | wc -l | tr -d ' ')"
echo "Backed up to ${OUT_FILE}, ${SIZE}, ${DURATION}s. Older backups: ${RETAINED} retained."
