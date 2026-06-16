@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

:: 기존 서버 종료
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1

call venv\Scripts\activate.bat

:: 로컬 LLM(Ollama) 준비 — 없으면 설치/데몬/모델 (실패해도 Gemini로 진행)
call "%~dp0ensure_ollama.bat"

echo ==================================================
echo   ReviewDong 서버를 시작합니다...
echo ==================================================
python app.py
