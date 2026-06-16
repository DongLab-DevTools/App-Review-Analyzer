#!/bin/bash
# 스크립트 폴더를 절대경로로 먼저 고정 (cd 이후엔 $0 상대경로가 깨지므로)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# 기존 서버 종료
lsof -ti:5001 | xargs kill -9 2>/dev/null

# 가상환경 활성화
source venv/bin/activate

# 로컬 LLM(Ollama) 준비 — 없으면 설치/데몬/모델 (실패해도 Gemini로 진행)
source "$SCRIPT_DIR/ensure_ollama.sh"

# 서버 실행
echo "=================================================="
echo "  ReviewDong 서버를 시작합니다..."
echo "=================================================="
python app.py
