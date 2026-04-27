#!/bin/bash
#
# Stop Validator Nitro Enclave
# ============================
#

echo "=========================================="
echo "🛑 Stopping Validator Nitro Enclave"
echo "=========================================="

# Get running enclaves
ENCLAVES=$(nitro-cli describe-enclaves 2>/dev/null)

if [ -z "$ENCLAVES" ] || [ "$ENCLAVES" = "[]" ]; then
    echo "ℹ️  No enclaves running"
    exit 0
fi

echo "Current enclaves:"
echo "$ENCLAVES"
echo ""

# Extract enclave IDs and terminate each
echo "$ENCLAVES" | grep -o '"EnclaveID": "[^"]*"' | cut -d'"' -f4 | while read ENCLAVE_ID; do
    if [ -n "$ENCLAVE_ID" ]; then
        echo "🛑 Terminating enclave: $ENCLAVE_ID"
        nitro-cli terminate-enclave --enclave-id "$ENCLAVE_ID" || true
    fi
done

echo ""
echo "✅ All enclaves terminated"
echo ""
nitro-cli describe-enclaves
