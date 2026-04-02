#!/bin/bash
cd "$(dirname "$0")/../.."

echo "=================================================="
echo "  ReviewDong 초기 설정"
echo "=================================================="

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3이 설치되어 있지 않습니다."
    echo "   https://www.python.org/downloads/ 에서 설치해주세요."
    read -p "아무 키나 누르면 종료..."
    exit 1
fi

echo "✓ Python3: $(python3 --version)"

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo "→ 가상환경 생성 중..."
    python3 -m venv venv
    echo "✓ 가상환경 생성 완료"
else
    echo "✓ 가상환경 이미 존재"
fi

# 의존성 설치
echo "→ 의존성 설치 중... (최초 실행 시 수 분 소요)"
source venv/bin/activate
pip install -r requirements.txt --quiet

echo ""
echo "=================================================="
echo "  ✓ 설정 완료!"
echo "  scripts/mac/run.command를 더블클릭하여 서버를 시작하세요."
echo "=================================================="
read -p "아무 키나 누르면 종료..."
