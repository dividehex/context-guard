#!/usr/bin/env sh
# Create or update the Context Guard filter through the Open WebUI API.
# Usage: OPENWEBUI_URL=http://localhost:3000 OPENWEBUI_API_KEY=sk-... ./install-filter.sh
# The manual path (Admin → Functions → Import) is equivalent; this only saves clicks on redeploys.
set -eu
: "${OPENWEBUI_URL:?set OPENWEBUI_URL}"
: "${OPENWEBUI_API_KEY:?set OPENWEBUI_API_KEY (an admin API key)}"
ID="context_guard"
DIR=$(cd "$(dirname "$0")" && pwd)
CONTENT=$(python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read()))' "$DIR/context_guard_filter.py")
PAYLOAD=$(printf '{"id":"%s","name":"Context Guard","content":%s,"meta":{"description":"Shows the Context Guard health score under each reply (UI only).","manifest":{}}}' "$ID" "$CONTENT")
if curl -fsS -H "Authorization: Bearer $OPENWEBUI_API_KEY" "$OPENWEBUI_URL/api/v1/functions/id/$ID" >/dev/null 2>&1; then
  curl -fsS -X POST -H "Authorization: Bearer $OPENWEBUI_API_KEY" -H "Content-Type: application/json" \
    -d "$PAYLOAD" "$OPENWEBUI_URL/api/v1/functions/id/$ID/update" >/dev/null && echo "updated function $ID"
else
  curl -fsS -X POST -H "Authorization: Bearer $OPENWEBUI_API_KEY" -H "Content-Type: application/json" \
    -d "$PAYLOAD" "$OPENWEBUI_URL/api/v1/functions/create" >/dev/null && echo "created function $ID"
fi
echo "Now enable it and toggle Global in Admin → Functions."
