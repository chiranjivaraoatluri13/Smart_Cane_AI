@echo off
REM Quick video analysis - one click!

cd /d C:\Users\chira\Projects\assistive-navigation
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo   ANALYZING YOUR SCREEN RECORDING...
echo ============================================================
echo.

python process_video.py "C:\Users\chira\Videos\Screen Recordings\Screen Recording 2026-05-25 193552.mp4" --show --save-dir output\video_analysis

echo.
echo ============================================================
echo   Analysis complete!
echo   Check: output\video_analysis\analysis_summary.json
echo ============================================================
pause
