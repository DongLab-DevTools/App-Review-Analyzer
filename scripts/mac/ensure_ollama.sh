#!/bin/bash
# 로컬 LLM(Ollama) 준비: 설치 확인 → 없으면 설치 → 데몬 기동 → 모델 다운로드.
# run.command / setup.command 에서 `source` 됨. 실패해도 서버는 그대로 진행(Gemini 사용).
#
#  - 모델 변경:  OLLAMA_MODEL 환경변수 또는 아래 기본값 수정
#                기본 2.4b(빠름·폴백용). 고품질 원하면 exaone3.5:7.8b (단 느림)
#  - 끄기:       USE_OLLAMA=0 으로 두면 이 단계를 통째로 건너뜀
: "${OLLAMA_MODEL:=exaone3.5:2.4b}"

_ensure_ollama() {
  if [ "${USE_OLLAMA:-1}" != "1" ]; then
    echo "ℹ USE_OLLAMA=0 → 로컬 LLM 준비 건너뜀 (Gemini만 사용)"
    return 0
  fi

  echo "──────────────────────────────────────────────"
  echo "  로컬 LLM(Ollama) 준비 — 모델: $OLLAMA_MODEL"

  # 1) 설치 확인 → 없으면 설치
  if ! command -v ollama >/dev/null 2>&1; then
    echo "→ Ollama 미설치. 설치를 시도합니다..."
    if command -v brew >/dev/null 2>&1; then
      brew install ollama
    else
      echo "  Homebrew 없음 → 공식 설치 스크립트 시도"
      curl -fsSL https://ollama.com/install.sh | sh
    fi
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    echo "⚠ Ollama 설치 실패 — https://ollama.com/download 에서 수동 설치하세요."
    echo "  (지금은 Gemini만으로 계속 진행합니다)"
    echo "──────────────────────────────────────────────"
    return 0
  fi
  echo "✓ Ollama: $(ollama --version 2>/dev/null | head -1)"

  # 2) 데몬 기동 (이미 떠 있으면 skip)
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "→ Ollama 데몬 시작..."
    ollama serve >/tmp/ollama_serve.log 2>&1 &
    for _ in $(seq 1 30); do
      curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi

  # 3) 모델 준비 (없으면 pull)
  if ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
    echo "✓ 모델 준비됨: $OLLAMA_MODEL"
  else
    # Ollama 레지스트리 인증 키 보장 (없으면 pull이 'id_ed25519: no such file'로 실패)
    mkdir -p "$HOME/.ollama" 2>/dev/null
    if [ ! -f "$HOME/.ollama/id_ed25519" ]; then
      echo "→ Ollama 인증 키 생성..."
      ssh-keygen -t ed25519 -f "$HOME/.ollama/id_ed25519" -N "" -q 2>/dev/null
    fi
    echo "→ 모델 다운로드: $OLLAMA_MODEL (최초 1회, 수 GB — 시간이 걸릴 수 있습니다)"
    ollama pull "$OLLAMA_MODEL" || echo "⚠ 모델 다운로드 실패 — Gemini로 계속 진행"
  fi
  echo "──────────────────────────────────────────────"
}
_ensure_ollama
