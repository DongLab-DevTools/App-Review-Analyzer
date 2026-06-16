@echo off
chcp 65001 >nul
:: 로컬 LLM(Ollama) 준비: 설치 확인 → 없으면 winget 설치 → 데몬 기동 → 모델 다운로드.
:: run.bat / setup.bat 에서 call 됨. 실패해도 서버는 그대로 진행(Gemini 사용).
::   - 모델 변경:  OLLAMA_MODEL 환경변수 또는 아래 기본값 수정 (기본 2.4b 빠름, 고품질은 7.8b)
::   - 끄기:       set USE_OLLAMA=0

if "%USE_OLLAMA%"=="0" (
    echo ℹ USE_OLLAMA=0 - 로컬 LLM 준비 건너뜀 ^(Gemini만 사용^)
    goto :eof
)
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=exaone3.5:2.4b"

echo --------------------------------------------------
echo   로컬 LLM(Ollama) 준비 - 모델: %OLLAMA_MODEL%

:: 1) 설치 확인 → 없으면 winget 설치
where ollama >nul 2>&1
if errorlevel 1 (
    echo → Ollama 미설치. winget으로 설치 시도...
    winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
)

where ollama >nul 2>&1
if errorlevel 1 (
    echo ⚠ Ollama 설치 실패 또는 PATH 미반영 - 터미널을 다시 열거나 https://ollama.com/download 에서 수동 설치
    echo   지금은 Gemini만으로 계속 진행합니다.
    echo --------------------------------------------------
    goto :eof
)
echo ✓ Ollama 설치됨

:: 2) 데몬 확인 (Windows는 설치 시 자동 실행되는 경우가 많음)
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo → Ollama 데몬 시작...
    start "" /b ollama serve
    timeout /t 5 >nul
)

:: 3) 모델 준비 (없으면 pull)
ollama list 2>nul | findstr /i /c:"%OLLAMA_MODEL%" >nul 2>&1
if errorlevel 1 (
    :: Ollama 레지스트리 인증 키 보장 (없으면 pull이 id_ed25519 오류로 실패)
    if not exist "%USERPROFILE%\.ollama" mkdir "%USERPROFILE%\.ollama"
    if not exist "%USERPROFILE%\.ollama\id_ed25519" (
        echo → Ollama 인증 키 생성...
        ssh-keygen -t ed25519 -f "%USERPROFILE%\.ollama\id_ed25519" -N "" -q
    )
    echo → 모델 다운로드: %OLLAMA_MODEL% ^(최초 1회, 수 GB^)
    ollama pull %OLLAMA_MODEL%
) else (
    echo ✓ 모델 준비됨: %OLLAMA_MODEL%
)
echo --------------------------------------------------
goto :eof
