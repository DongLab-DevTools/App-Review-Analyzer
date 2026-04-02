@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

echo ==================================================
echo   ReviewDong 초기 설정
echo ==================================================

:: Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다.
    echo 자동으로 설치를 시도합니다...
    echo.

    :: winget으로 설치 시도
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo winget 설치 실패. 아래 링크에서 직접 설치해주세요.
        echo https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo.
    echo Python 설치 완료. 터미널을 닫고 Setup.bat를 다시 실행해주세요.
    pause
    exit /b 0
)

for /f "tokens=*" %%i in ('python --version') do echo ✓ %%i

:: 가상환경 생성
if not exist "venv" (
    echo → 가상환경 생성 중...
    python -m venv venv
    echo ✓ 가상환경 생성 완료
) else (
    echo ✓ 가상환경 이미 존재
)

:: 의존성 설치
echo → 의존성 설치 중... (최초 실행 시 수 분 소요)
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

echo.
echo ==================================================
echo   ✓ 설정 완료!
echo   scripts\windows\run.bat를 더블클릭하여 서버를 시작하세요.
echo ==================================================
pause
