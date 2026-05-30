@echo off
REM Start the phone server for assistive navigation

cd /d %~dp0
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo   ASSISTIVE NAVIGATION - PHONE SERVER
echo ============================================================
echo.
echo Installing Flask if needed...
pip install flask >nul 2>&1

echo.
echo Starting server...
echo.
echo IMPORTANT: Note the IP address shown below!
echo You'll need it to connect from your phone.
echo.

python phone_server.py

pause
