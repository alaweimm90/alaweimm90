#!/bin/bash
# Enhanced Secrets Manager v2.3
# AI Tools - Secure API key management

SECRETS_FILE="$HOME/.ai_tools/.secrets"
touch "$SECRETS_FILE" && chmod 600 "$SECRETS_FILE"

set_secret() { grep -q "^$1=" "$SECRETS_FILE" && sed -i "s|^$1=.*|$1=$2|" "$SECRETS_FILE" || echo "$1=$2" >> "$SECRETS_FILE"; echo "✓ Set: $1"; }
get_secret() { grep "^$1=" "$SECRETS_FILE" | cut -d= -f2-; }
list_secrets() { echo "Secrets:"; cut -d= -f1 "$SECRETS_FILE" | while read k; do echo "  $k=***"; done; }
delete_secret() { sed -i "/^$1=/d" "$SECRETS_FILE"; echo "✓ Deleted: $1"; }
load_secrets() { set -a; source "$SECRETS_FILE" 2>/dev/null; set +a; echo "✓ Loaded"; }

case "$1" in
    set) set_secret "$2" "$3" ;;
    get) get_secret "$2" ;;
    list) list_secrets ;;
    delete) delete_secret "$2" ;;
    load) load_secrets ;;
    *) echo "Usage: ai-secrets <set|get|list|delete|load>" ;;
esac
