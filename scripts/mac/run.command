#!/bin/bash
cd "$(dirname "$0")/../.."

# 기존 서버 종료
lsof -ti:5001 | xargs kill -9 2>/dev/null

# 가상환경 활성화
source venv/bin/activate

# 서버 실행
echo "=================================================="
echo "  ReviewDong 서버를 시작합니다..."
echo "=================================================="
python app.py
