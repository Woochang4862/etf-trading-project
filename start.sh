#!/bin/bash
# ETF Trading Pipeline - 원클릭 시작 스크립트 (Nginx 포함)
# macOS (Docker Desktop) 및 Linux 모두 지원
#
# 사용법:
#   ./start.sh                    # 전체 서비스 빌드 및 시작 (기존 동작)
#   ./start.sh web-dashboard      # web-dashboard만 빌드 및 재시작
#   ./start.sh ml-service         # ml-service만 빌드 및 재시작
#   ./start.sh nginx              # nginx만 재시작
#   ./start.sh web-dashboard ml-service  # 여러 서비스 빌드 및 재시작
#
# 예시:
#   # 코드 변경 후 Next.js만 빠르게 재빌드
#   ./start.sh web-dashboard
#
#   # ml-service 코드만 변경 후 재빌드
#   ./start.sh ml-service

cd "$(dirname "$0")"

echo "🚀 ETF Trading Pipeline 시작..."

# OS 감지
OS_NAME=$(uname)

# Docker 실행 확인 및 권한 체크
check_docker() {
    # Docker 명령어 존재 확인
    if ! command -v docker >/dev/null 2>&1; then
        echo "❌ Docker가 설치되어 있지 않습니다."
        if [ "$OS_NAME" = "Darwin" ]; then
            echo "💡 macOS: Docker Desktop을 설치하세요"
            echo "   https://www.docker.com/products/docker-desktop/"
        else
            echo "💡 Linux: Docker를 설치하세요"
            echo "   curl -fsSL https://get.docker.com | sh"
        fi
        exit 1
    fi

    # Docker 데몬 실행 확인 (docker ps로 실제 연결 테스트)
    echo "🔍 Docker 연결 확인 중..."

    if ! docker ps >/dev/null 2>&1; then
        if [ "$OS_NAME" = "Darwin" ]; then
            # macOS: Docker Desktop이 실행 중인지 확인
            echo "⚠️  Docker Desktop에 연결할 수 없습니다."

            # Docker Desktop 자동 시작 시도
            if [ -d "/Applications/Docker.app" ]; then
                echo "🔄 Docker Desktop 시작 중..."
                open -a Docker

                # Docker가 준비될 때까지 대기 (최대 60초)
                echo -n "   Docker Desktop 준비 대기 중"
                for i in $(seq 1 60); do
                    if docker ps >/dev/null 2>&1; then
                        echo ""
                        echo "   ✅ Docker Desktop 준비 완료"
                        return 0
                    fi
                    echo -n "."
                    sleep 1
                done
                echo ""
                echo "❌ Docker Desktop 시작 시간 초과"
                echo ""
                echo "💡 해결 방법:"
                echo "   1. Docker Desktop이 완전히 시작될 때까지 기다리세요"
                echo "   2. 메뉴바에서 Docker 아이콘이 'Docker Desktop is running' 상태인지 확인"
                echo "   3. 다시 ./start.sh 실행"
            else
                echo "❌ Docker Desktop이 설치되어 있지 않습니다."
                echo "💡 https://www.docker.com/products/docker-desktop/ 에서 설치하세요"
            fi
        else
            # Linux: Docker 데몬 또는 권한 문제
            echo "❌ Docker 데몬에 연결할 수 없습니다."
            echo ""
            echo "💡 해결 방법:"
            echo ""

            # systemd로 docker 서비스 확인
            if command -v systemctl >/dev/null 2>&1; then
                if ! systemctl is-active --quiet docker 2>/dev/null; then
                    echo "   Docker 서비스가 실행되지 않습니다. 시작하세요:"
                    echo "   sudo systemctl start docker"
                    echo ""
                fi
            fi

            # docker 그룹 확인
            if getent group docker >/dev/null 2>&1; then
                if groups | grep -q docker; then
                    echo "   ✅ 사용자가 docker 그룹에 포함되어 있습니다."
                    echo "   🔄 그룹 변경사항이 아직 적용되지 않았을 수 있습니다."
                    echo ""
                    echo "   다음 중 하나를 실행하세요:"
                    echo "   1. newgrp docker  # 현재 세션에 즉시 적용"
                    echo "   2. 재로그인       # 새 세션에서 적용"
                    echo ""
                    echo "   그 다음 다시: ./start.sh"
                else
                    echo "   ❌ 사용자가 docker 그룹에 포함되어 있지 않습니다."
                    echo ""
                    echo "   다음 명령을 실행하여 docker 그룹에 추가하세요:"
                    echo "   sudo usermod -aG docker $USER"
                    echo ""
                    echo "   그 다음:"
                    echo "   newgrp docker  # 또는 재로그인"
                    echo "   ./start.sh"
                fi
            else
                echo "   Docker 그룹이 존재하지 않습니다."
                echo "   Docker가 올바르게 설치되었는지 확인하세요."
            fi
            echo ""
            echo "   또는 임시로 sudo 사용:"
            echo "   sudo ./start.sh"
        fi
        echo ""
        exit 1
    fi

    echo "✅ Docker 준비 완료"
}

# Docker 확인 실행
check_docker

# 1. SSH 터널 시작 (이미 있으면 스킵)
# OS에 따라 바인딩 주소 결정 (OS_NAME은 위에서 이미 설정됨)
if [ "$OS_NAME" = "Darwin" ]; then
    # macOS: Docker Desktop이 호스트의 localhost 포워딩을 처리하므로 127.0.0.1 사용
    BIND_ADDRESS="127.0.0.1"
    echo "🍎 macOS: SSH 터널을 localhost($BIND_ADDRESS)에 바인딩"
else
    # Linux: Docker 컨테이너가 host.docker.internal로 접근하려면 호스트 IP(혹은 0.0.0.0)에 바인딩 필요
    BIND_ADDRESS="0.0.0.0"
    echo "🐧 Linux: SSH 터널을 모든 인터페이스($BIND_ADDRESS)에 바인딩"
fi

if ! pgrep -f "ssh.*3306:127.0.0.1:5100" > /dev/null; then
    echo "📡 SSH 터널 시작 중..."
    ssh -f -N -L ${BIND_ADDRESS}:3306:127.0.0.1:5100 ahnbi2@ahnbi2.suwon.ac.kr \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3
    sleep 2
    echo "✅ SSH 터널 시작됨"
else
    echo "✅ SSH 터널 이미 실행 중"
fi

# 2. 기존 프로세스 정리 (포트 3000, 8000, 80 충돌 방지)
echo "🔍 기존 프로세스 확인 중..."
for port in 3000 8000 80; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$pid" ]; then
        echo "  포트 $port 사용 중인 프로세스 (PID: $pid) 발견"
        # Docker 컨테이너가 아닌 경우에만 종료 시도
        if ! docker ps --format "{{.ID}}" 2>/dev/null | grep -q "$pid" 2>/dev/null; then
            # 일반 권한으로 종료 시도, 실패하면 sudo 사용
            if kill -9 $pid 2>/dev/null; then
                echo "  포트 $port 프로세스 종료됨"
            elif sudo kill -9 $pid 2>/dev/null; then
                echo "  포트 $port 프로세스 종료됨 (sudo 사용)"
            else
                echo "  ⚠️  포트 $port 프로세스 종료 실패 (권한 필요할 수 있음)"
            fi
        else
            echo "  포트 $port는 Docker 컨테이너에서 사용 중 - 무시"
        fi
    fi
done

# 3. Docker Compose로 서비스 시작
echo "🐳 Docker 컨테이너 시작 중..."

# Docker 연결 재확인 (compose 실행 전)
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker 연결이 끊어졌습니다. 다시 시도하세요."
    exit 1
fi

# Docker Compose v2 찾기 (PATH 또는 직접 경로)
DOCKER_COMPOSE_CMD=""
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif [ -f "$HOME/.docker/cli-plugins/docker-compose" ] && "$HOME/.docker/cli-plugins/docker-compose" version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="$HOME/.docker/cli-plugins/docker-compose"
else
    echo "❌ Docker Compose v2를 찾을 수 없습니다"
    if [ "$OS_NAME" = "Darwin" ]; then
        echo "💡 Docker Desktop에 포함되어 있어야 합니다. Docker Desktop을 재설치해보세요."
    else
        echo "💡 Docker Compose v2 설치 필요:"
        echo "   mkdir -p ~/.docker/cli-plugins"
        echo "   wget -O ~/.docker/cli-plugins/docker-compose https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-x86_64"
        echo "   chmod +x ~/.docker/cli-plugins/docker-compose"
    fi
    exit 1
fi

echo "   사용할 명령: $DOCKER_COMPOSE_CMD"

if [ $# -gt 0 ]; then
    echo ""
    echo "🎯 대상 서비스만 빌드: $@"
    if ! $DOCKER_COMPOSE_CMD up -d --build "$@"; then
        REBUILD_ERROR=true
    fi
else
    if ! $DOCKER_COMPOSE_CMD up -d --build; then
        REBUILD_ERROR=true
    fi
fi

if [ "$REBUILD_ERROR" = true ]; then
    echo ""
    echo "❌ Docker Compose 실행 실패"
    echo ""
    echo "💡 문제 해결:"
    if [ "$OS_NAME" = "Darwin" ]; then
        echo "   1. Docker Desktop이 완전히 시작되었는지 확인"
        echo "   2. Docker Desktop > Settings > Resources 에서 메모리/CPU 확인"
    fi
    echo "   3. 로그 확인: $DOCKER_COMPOSE_CMD logs"
    exit 1
fi

# 4. 헬스체크 대기 (Nginx를 통해 포트 80으로)
echo "⏳ 서비스 준비 대기 중..."
MAX_WAIT=90
PORT_READY=0
HTTP_READY=0

for i in $(seq 1 $MAX_WAIT); do
    # 1단계: 포트 80이 열려있는지 확인
    if [ $PORT_READY -eq 0 ]; then
        if nc -zv localhost 80 >/dev/null 2>&1 || ss -tlnp 2>/dev/null | grep -q ':80 '; then
            PORT_READY=1
            echo ""
            echo "   ✅ 포트 80 열림"
        fi
    fi
    
    # 2단계: HTTP 응답 확인 (포트가 열린 후에만)
    if [ $PORT_READY -eq 1 ]; then
        response=""
        if command -v wget >/dev/null 2>&1; then
            response=$(wget -q -O- --timeout=2 http://localhost/health 2>/dev/null || echo "")
        elif command -v curl >/dev/null 2>&1; then
            response=$(curl -s --max-time 2 http://localhost/health 2>/dev/null || echo "")
        fi
        
        if [ ! -z "$response" ]; then
            if echo "$response" | grep -qE "(healthy|status)"; then
                HTTP_READY=1
            fi
        fi
    fi
    
    # 두 단계 모두 완료되면 성공
    if [ $PORT_READY -eq 1 ] && [ $HTTP_READY -eq 1 ]; then
        echo ""
        echo "✅ 서비스 시작 완료!"
        echo ""
        echo "🌐 외부 접근 URL:"
        echo "   📊 웹 대시보드: http://ahnbi2.suwon.ac.kr/"
        echo "   📖 API 문서: http://ahnbi2.suwon.ac.kr/docs"
        echo "   💚 헬스체크: http://ahnbi2.suwon.ac.kr/health"
        echo "   🔌 API 엔드포인트: http://ahnbi2.suwon.ac.kr/api/predictions"
        echo ""
        echo "🏠 로컬 접근 URL:"
        echo "   📊 웹 대시보드: http://localhost/"
        echo "   📖 API 문서: http://localhost/docs"
        echo "   💚 헬스체크: http://localhost/health"
        echo ""
        
        # 컨테이너 상태 확인
        if [ ! -z "$DOCKER_COMPOSE_CMD" ]; then
            echo "📦 컨테이너 상태:"
            $DOCKER_COMPOSE_CMD ps 2>/dev/null || true
        fi
        
        exit 0
    fi
    
    echo -n "."
    sleep 1
done

# 타임아웃 시 상세 정보 제공
echo ""
echo "⚠️  서비스 시작 타임아웃"
echo ""
echo "📋 진단 정보:"
echo "   포트 80 상태:"
if ss -tlnp 2>/dev/null | grep -q ':80 '; then
    echo "   ✅ 포트 80 열림"
    ss -tlnp 2>/dev/null | grep ':80 ' | head -2
else
    echo "   ❌ 포트 80 닫힘"
fi

echo ""
echo "   컨테이너 상태:"
if [ ! -z "$DOCKER_COMPOSE_CMD" ]; then
    $DOCKER_COMPOSE_CMD ps 2>/dev/null || echo "   확인 불가"
else
    echo "   확인 불가 (Docker Compose 명령 없음)"
fi

echo ""
echo "   로그 확인:"
echo "   $DOCKER_COMPOSE_CMD logs nginx --tail 30"
echo "   $DOCKER_COMPOSE_CMD logs web-dashboard --tail 30"
echo "   $DOCKER_COMPOSE_CMD logs ml-service --tail 30"

echo ""
echo "❌ 서비스 시작 실패 또는 타임아웃"
echo "📋 로그 확인:"
if [ ! -z "$DOCKER_COMPOSE_CMD" ]; then
    echo "   $DOCKER_COMPOSE_CMD logs"
    echo "   $DOCKER_COMPOSE_CMD ps"
else
    echo "   docker logs <container-name>"
    echo "   docker ps"
fi
exit 1
