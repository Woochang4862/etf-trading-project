#!/bin/bash
# 매일 전체 종목 예측 실행 스크립트
# 미국 정규장 마감 후 실행 (5 PM ET = 22:00 UTC, 월~금)
# cron: 0 22 * * 1-5 /path/to/scripts/predict-daily.sh

# PATH 설정 (cron 환경용)
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:$PATH"
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

LOG_DIR="/home/ahnbi2/etf-trading-project/logs"
LOG_FILE="$LOG_DIR/predict-$(date +%Y%m%d).log"
PROJECT_DIR="/home/ahnbi2/etf-trading-project"

mkdir -p "$LOG_DIR"

echo "========================================" >> "$LOG_FILE"
echo "🔮 일일 예측 시작: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 1. 서비스 상태 확인 및 시작
cd "$PROJECT_DIR"

# SSH 터널 확인
if ! pgrep -f "ssh.*3306:127.0.0.1:5100" > /dev/null; then
    echo "📡 SSH 터널 시작..." >> "$LOG_FILE"
    ssh -f -N -L 3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3
    sleep 3
fi

# Docker 컨테이너 확인
if ! docker ps | grep -q "etf-ml-service"; then
    echo "🐳 Docker 컨테이너 시작..." >> "$LOG_FILE"
    docker-compose up -d
    sleep 10
fi

# 2. 헬스체크
for i in {1..30}; do
    if wget -q -O- http://localhost:8000/health | grep -q "healthy"; then
        echo "✅ 서비스 정상" >> "$LOG_FILE"
        break
    fi
    sleep 2
done

# 3. 전체 종목 랭킹 예측 실행
echo "🚀 랭킹 예측 실행 중..." >> "$LOG_FILE"
RESULT=$(wget -q -O- --post-data='' \
    "http://localhost:8000/api/predictions/ranking")

# 결과 파싱
TOTAL=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_symbols', 0))" 2>/dev/null)
TOP=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin)['rankings'][0]; print(f\"{r['symbol']} (score={r['score']})\")" 2>/dev/null)

echo "📊 결과: $TOTAL 종목 랭킹 예측 완료" >> "$LOG_FILE"
echo "🏆 1위: $TOP" >> "$LOG_FILE"
echo "완료 시간: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 4. 요약 출력
echo "✅ 일일 랭킹 예측 완료: $TOTAL 종목, 1위: $TOP"
