#!/usr/bin/env bash
# Ingest the full test corpus into test.db
set -euo pipefail

DB="test.db"
DOCS_DIR="./docs"

# Remove existing DB for a clean run
if [ -f "$DB" ]; then
    echo "Removing existing $DB..."
    rm -f "$DB" "${DB}-wal" "${DB}-shm"
fi

echo "Ingesting all documents from $DOCS_DIR into $DB..."
echo
shoal --db "$DB" ingest-dir "$DOCS_DIR"
echo
shoal --db "$DB" status
