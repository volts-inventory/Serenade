#!/usr/bin/env bash
# Snapshot the three Serenade stores into a single directory.
# Dev reference. See docs/BACKUP_RESTORE.md for the production guidance.

set -euo pipefail
OUT="${1:?usage: snapshot.sh <output-dir>}"
mkdir -p "$OUT"

# Env from .env
set -o allexport
if [ -f .env ]; then source .env; fi
set +o allexport

POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-serenade}"
POSTGRES_USER="${POSTGRES_USER:-serenade}"

NEO4J_USER="${NEO4J_USER:-neo4j}"
MINIO_BUCKET="${MINIO_BUCKET:-serenade-evidence}"

echo "==> Postgres dump"
docker compose exec -T postgres pg_dump \
    -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    --format=plain --compress=9 \
    > "$OUT/postgres-dump.sql.gz"

echo "==> Neo4j Cypher export (via APOC)"
docker compose exec -T neo4j cypher-shell -u "$NEO4J_USER" -p "${NEO4J_PASSWORD}" \
    "CALL apoc.export.cypher.all('/tmp/neo4j-dump.cypher', {format:'cypher-shell'})" || true
docker compose exec -T neo4j cat /tmp/neo4j-dump.cypher | gzip -9 > "$OUT/neo4j-cypher-dump.cypher.gz"

echo "==> MinIO bucket mirror"
docker compose exec -T orchestrator mc alias set local "http://$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" >/dev/null
docker compose exec -T orchestrator sh -c "rm -rf /tmp/minio-mirror && mc mirror --quiet local/$MINIO_BUCKET /tmp/minio-mirror && tar czf - -C /tmp minio-mirror" \
    > "$OUT/minio-bucket.tar.gz"

echo "==> Manifest"
{
    echo "snapshot: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "files:"
    for f in postgres-dump.sql.gz neo4j-cypher-dump.cypher.gz minio-bucket.tar.gz; do
        sha=$(sha256sum "$OUT/$f" | cut -d' ' -f1)
        sz=$(stat -c%s "$OUT/$f")
        echo "  - name: $f"
        echo "    sha256: $sha"
        echo "    size_bytes: $sz"
    done
} > "$OUT/snapshot.manifest"

echo "==> Done. Snapshot at $OUT"
