#!/bin/bash
# 매일 TradingView 데이터 스크래핑 실행 스크립트 (API 호출 방식)
# 미국 정규장 마감 후 실행 (5 PM ET = 22:00 UTC, 월~금)
# cron: 0 22 * * 1-5 /home/ahnbi2/etf-trading-project/scripts/scrape-daily.sh

set -e

# PATH 설정 (cron 환경용)
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:$PATH"

API_BASE="http://localhost/api/scraper"
LOG_DIR="/home/ahnbi2/etf-trading-project/logs"
LOG_FILE="$LOG_DIR/cron-scraper-$(date +%Y%m%d).log"
POLL_INTERVAL=60  # seconds
MAX_WAIT_HOURS=4  # Maximum wait time

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "📊 일일 스크래핑 시작"
log "========================================"

# 1. SSH 터널 확인 및 시작
if ! pgrep -f "ssh.*3306:127.0.0.1:5100" > /dev/null; then
    log "📡 SSH 터널 시작..."
    ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 2>> "$LOG_FILE"

    if [ $? -ne 0 ]; then
        log "❌ SSH 터널 시작 실패"
        exit 1
    fi
    sleep 3
    log "✅ SSH 터널 시작 완료"
else
    log "✅ SSH 터널 이미 실행 중"
fi

# 2. Check if job already running
HEALTH=$(curl -s "$API_BASE/health" 2>/dev/null || echo '{"error": "connection failed"}')
if echo "$HEALTH" | grep -q "error"; then
    log "❌ Scraper service not available: $HEALTH"
    exit 1
fi

CURRENT_JOB=$(echo "$HEALTH" | jq -r '.current_job // empty')
if [ -n "$CURRENT_JOB" ] && [ "$CURRENT_JOB" != "null" ]; then
    log "⚠️ Job already running: $CURRENT_JOB"
    log "Waiting for completion..."
else
    # 3. Start full scraping job
    log "🚀 Starting full scraping job..."
    RESPONSE=$(curl -s -X POST "$API_BASE/jobs/full" -H "Content-Type: application/json")
    JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

    if [ -z "$JOB_ID" ] || [ "$JOB_ID" == "null" ]; then
        log "❌ Failed to start job: $RESPONSE"
        exit 1
    fi

    log "✅ Job started: $JOB_ID"
fi

# 4. Poll for completion
START_TIME=$(date +%s)
MAX_WAIT=$((MAX_WAIT_HOURS * 3600))

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $MAX_WAIT ]; then
        log "❌ Timeout: Job exceeded ${MAX_WAIT_HOURS} hours"
        exit 1
    fi

    STATUS_RESPONSE=$(curl -s "$API_BASE/jobs/status" 2>/dev/null || echo '{"status": "error"}')
    STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status // "unknown"')
    PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress.current // 0')
    TOTAL=$(echo "$STATUS_RESPONSE" | jq -r '.progress.total // 0')
    CURRENT_SYMBOL=$(echo "$STATUS_RESPONSE" | jq -r '.progress.current_symbol // "N/A"')

    log "Status: $STATUS | Progress: $PROGRESS/$TOTAL | Current: $CURRENT_SYMBOL"

    case "$STATUS" in
        "completed")
            log "✅ 스크래핑 성공!"
            SUMMARY=$(echo "$STATUS_RESPONSE" | jq -c '.progress // {}')
            log "Summary: $SUMMARY"
            exit 0
            ;;
        "failed")
            log "❌ 스크래핑 실패!"
            ERRORS=$(echo "$STATUS_RESPONSE" | jq -r '.progress.errors[]' 2>/dev/null || echo "Unknown error")
            log "Errors: $ERRORS"
            exit 1
            ;;
        "cancelled")
            log "⚠️ Job was cancelled"
            exit 1
            ;;
        "idle"|"null"|"")
            log "✅ No job running (completed or idle)"
            exit 0
            ;;
        "running"|"pending")
            sleep "$POLL_INTERVAL"
            ;;
        *)
            log "⚠️ Unknown status: $STATUS - continuing to poll"
            sleep "$POLL_INTERVAL"
            ;;
    esac
done
