#!/bin/bash
# cron 작업 설정 스크립트
# 전체 자동화 파이프라인: 데이터 수집 → 예측 → 매매 실행

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📅 Cron 작업 설정"
echo "================="
echo ""
echo "다음 작업을 crontab에 추가합니다:"
echo ""
echo "1. [데이터 파이프라인] 매일 22:00 UTC (7:00 KST, 월~금) - 스크래핑 → 피처 처리 → 예측"
echo "2. [서비스 헬스체크]   매 6시간마다 - Docker 서비스 상태 확인"
echo "3. [수익률 업데이트]   매주 일요일 02:00 UTC - 예측 실적 평가"
echo "4. [모델 재학습]       매년 1월 1일 03:00 UTC - ML 모델 재학습"
echo ""
echo "※ 매매 실행은 trading-service 내부 APScheduler가 처리 (기본: 23:30 KST)"
echo "  APScheduler가 자동으로 ML 랭킹 기반 매수/매도를 실행합니다."
echo ""

# 현재 crontab 백업
crontab -l > /tmp/crontab_backup 2>/dev/null
echo "📋 기존 crontab 백업: /tmp/crontab_backup"

# 기존 ETF 관련 작업 제거 후 새로 추가
(crontab -l 2>/dev/null | grep -v "etf-trading-project") | crontab -

# --- 로그 디렉토리 생성 ---
mkdir -p "$PROJECT_DIR/logs"

# =============================================================================
# 1. 전체 데이터 파이프라인 (스크래핑 → 피처 처리 → 예측)
#    - 22:00 UTC (5 PM ET / 7:00 KST 다음날) 월~금
#    - run-pipeline.sh가 5단계 순차 실행
#    - 예측 완료 후 trading-service의 APScheduler가 23:30 KST에 자동매매
# =============================================================================
(crontab -l 2>/dev/null; echo "# ETF Pipeline - 데이터 파이프라인 (스크래핑→피처→예측, 22:00 UTC 월~금)") | crontab -
(crontab -l 2>/dev/null; echo "0 22 * * 1-5 $PROJECT_DIR/scripts/run-pipeline.sh --skip-validation >> $PROJECT_DIR/logs/cron.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "") | crontab -

# =============================================================================
# 2. 서비스 헬스체크 (6시간마다)
#    - Docker 컨테이너 상태 확인
#    - 비정상 시 로그에 기록 (Slack 알림 가능)
# =============================================================================
(crontab -l 2>/dev/null; echo "# ETF Pipeline - 서비스 헬스체크 (6시간마다)") | crontab -
(crontab -l 2>/dev/null; echo "0 */6 * * * $PROJECT_DIR/scripts/check-services.sh >> $PROJECT_DIR/logs/cron.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "") | crontab -

# =============================================================================
# 3. 주간 수익률 업데이트 (매주 일요일 02:00 UTC)
#    - 과거 예측 실적 계산 및 DB 업데이트
# =============================================================================
(crontab -l 2>/dev/null; echo "# ETF Pipeline - 주간 수익률 업데이트 (일요일 02:00 UTC)") | crontab -
(crontab -l 2>/dev/null; echo "0 2 * * 0 $PROJECT_DIR/scripts/update-returns.sh >> $PROJECT_DIR/logs/cron.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "") | crontab -

# =============================================================================
# 4. 연간 모델 재학습 (매년 1월 1일 03:00 UTC)
#    - 전년도 데이터로 ML 모델 재학습
# =============================================================================
(crontab -l 2>/dev/null; echo "# ETF Pipeline - 연간 모델 학습 (1월 1일 03:00 UTC)") | crontab -
(crontab -l 2>/dev/null; echo "0 3 1 1 * $PROJECT_DIR/scripts/train-yearly.sh >> $PROJECT_DIR/logs/cron.log 2>&1") | crontab -

echo ""
echo "✅ Cron 작업 설정 완료!"
echo ""
echo "현재 설정된 cron 작업:"
echo "---------------------"
crontab -l | grep -v "^$" | grep -v "^#$"
echo ""
echo "📝 로그 위치: $PROJECT_DIR/logs/"
echo ""
echo "========================================="
echo "📌 자동화 파이프라인 타임라인 (KST 기준)"
echo "========================================="
echo ""
echo "  07:00  데이터 파이프라인 시작 (cron)"
echo "  │  Step 1: TradingView 스크래핑"
echo "  │  Step 3: 피처 처리 (85개 피처 생성)"
echo "  │  Step 4: ML 랭킹 예측 (LightGBM)"
echo "  │  Step 5: trading-service 상태 확인"
echo "  ~12:00  파이프라인 완료"
echo "  │"
echo "  23:30  APScheduler 매매 실행 (trading-service)"
echo "         │  잔고 조회 → ML 랭킹 소비"
echo "         │  FIFO 매도 (Day >= 64)"
echo "         │  고정 ETF 매수 (30%)"
echo "         │  전략 ETF 매수 (70%, 100종목)"
echo "         └  거래 기록 저장 (SQLite)"
echo ""
echo "========================================="
echo ""
echo "수동 매매 실행: ./scripts/execute-trading.sh"
echo "매매 상태 확인: ./scripts/execute-trading.sh --status"
echo "포트폴리오:     ./scripts/execute-trading.sh --portfolio"
