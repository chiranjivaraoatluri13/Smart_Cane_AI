@echo off
REM Process your screen recording through assistive navigation

cd /d %~dp0
call .venv\Scripts\activate.bat

echo.
echo ================================================================
echo   ASSISTIVE NAVIGATION - VIDEO PROCESSOR
echo ================================================================
echo.
echo Processing your screen recording from today...
echo This will analyze the video and show what the system detected.
echo.
echo Options:
echo   [1] Process with visualization (recommended)
echo   [2] Process and save annotated frames
echo   [3] Quick analysis (no visualization)
echo   [Q] Quit
echo.
set /p choice="Select option (1-3 or Q): "

if /i "%choice%"=="Q" exit /b 0
if /i "%choice%"=="q" exit /b 0

set VIDEO_PATH=C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4

if "%choice%"=="1" (
    echo.
    echo Starting visualization...
    echo Press 'q' in the window to stop early.
    echo.
    python process_video.py "%VIDEO_PATH%" --show
)

if "%choice%"=="2" (
    echo.
    echo Processing and saving frames...
    echo Output will be in: output\video_analysis\
    echo.
    python process_video.py "%VIDEO_PATH%" --show --save-dir output\video_analysis
)

if "%choice%"=="3" (
    echo.
    echo Running quick analysis...
    echo.
    python process_video.py "%VIDEO_PATH%"
)

echo.
echo ================================================================
echo   DONE!
echo ================================================================
pause
