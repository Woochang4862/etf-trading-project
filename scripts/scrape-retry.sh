#!/bin/bash
# 단일 종목 리트라이 스크립트
# Usage: ./scripts/scrape-retry.sh AAPL
# Usage: ./scripts/scrape-retry.sh AAPL NVDA MSFT  (multiple symbols)

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 SYMBOL [SYMBOL2 SYMBOL3 ...]"
    echo "Example: $0 AAPL"
    echo "Example: $0 AAPL NVDA MSFT"
    exit 1
fi

API_BASE="http://localhost/api/scraper"

echo "🔄 Starting retry for symbols: $@"

# Check if job running
HEALTH=$(curl -s "$API_BASE/health" 2>/dev/null || echo '{"error": "connection failed"}')
CURRENT_JOB=$(echo "$HEALTH" | jq -r '.current_job // empty')

if [ -n "$CURRENT_JOB" ] && [ "$CURRENT_JOB" != "null" ]; then
    echo "❌ Job already running: $CURRENT_JOB"
    echo "Wait for completion or cancel with: curl -X POST $API_BASE/jobs/cancel"
    exit 1
fi

# Build symbols array
SYMBOLS_JSON=$(printf '%s\n' "$@" | jq -R . | jq -s .)

# Start retry job
echo "📤 Sending retry request..."
RESPONSE=$(curl -s -X POST "$API_BASE/jobs/retry" \
    -H "Content-Type: application/json" \
    -d "{\"symbols\": $SYMBOLS_JSON}")

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

if [ -z "$JOB_ID" ] || [ "$JOB_ID" == "null" ]; then
    echo "❌ Failed to start job: $RESPONSE"
    exit 1
fi

echo "✅ Retry job started: $JOB_ID"
echo ""
echo "📊 Monitor progress:"
echo "   curl $API_BASE/jobs/status"
echo ""
echo "📜 View logs:"
echo "   curl $API_BASE/jobs/logs"
echo ""
echo "❌ Cancel job:"
echo "   curl -X POST $API_BASE/jobs/cancel"
