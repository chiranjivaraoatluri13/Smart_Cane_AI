@echo off
REM Assistive Navigation - Walking Mode Launcher
REM Quick launch script for real-world walking navigation

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║    ASSISTIVE NAVIGATION - WALKING MODE LAUNCHER        ║
echo ╚════════════════════════════════════════════════════════╝
echo.

cd /d %~dp0

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo.
echo Select walking mode:
echo.
echo [1] Basic Walking Mode (camera only)
echo [2] Demo Mode (optimized, frequent updates)
echo [3] With Map Navigation (needs coordinates)
echo [4] Preview Mode (segmentation only, no voice)
echo [5] Custom command
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto basic
if "%choice%"=="2" goto demo
if "%choice%"=="3" goto map
if "%choice%"=="4" goto preview
if "%choice%"=="5" goto custom

echo Invalid choice!
pause
exit /b 1

:basic
echo.
echo Starting Basic Walking Mode...
echo - Camera: Index 0
echo - Mode: Fast (CPU optimized)
echo - Voice: Enabled
echo - Press Ctrl+C to stop
echo.
timeout /t 3
assistive-nav run --fast --camera 0
goto end

:demo
echo.
echo Starting Demo Mode...
echo - Camera: Index 0
echo - Mode: Demo (frequent updates)
echo - Voice: Enabled
echo - Press Ctrl+C to stop
echo.
timeout /t 3
assistive-nav run --demo --camera 0
goto end

:map
echo.
echo Map Navigation Mode
echo.
echo Enter your current coordinates (format: LAT,LON)
echo Example: 40.7484,-73.9857
set /p current="Current position: "

echo.
echo Enter destination coordinates (format: LAT,LON)
echo Example: 40.7510,-73.9830
set /p dest="Destination: "

echo.
echo Starting Map Navigation...
echo - Route will be fetched from OSRM
echo - Voice guidance enabled
echo - Obstacle detection active
echo - Press Ctrl+C to stop
echo.
timeout /t 3
assistive-nav run --fast --use-map --current "%current%" --dest "%dest%" --camera 0
goto end

:preview
echo.
echo Starting Preview Mode (segmentation only)...
echo - Shows visual overlay only
echo - No voice commands
echo - Press 'q' in window to stop
echo.
timeout /t 3
assistive-nav preview --fast --camera 0
goto end

:custom
echo.
echo Enter custom command (after 'assistive-nav'):
set /p custom_cmd="Command: "
echo.
echo Running: assistive-nav %custom_cmd%
timeout /t 2
assistive-nav %custom_cmd%
goto end

:end
echo.
echo Session ended.
pause
