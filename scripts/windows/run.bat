@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

:: 기존 서버 종료
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1

call venv\Scripts\activate.bat

echo ==================================================
echo   ReviewDong 서버를 시작합니다...
echo ==================================================
python app.py
