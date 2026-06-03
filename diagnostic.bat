@echo off
REM Diagnostic tool - See what the camera detects

cd /d %~dp0
call .venv\Scripts\activate.bat

echo.
echo 
echo   ASSISTIVE NAVIGATION - CAMERA DIAGNOSTIC
echo 
echo.
echo This will show you what the camera sees and detects.
echo.
echo 1. A window will open showing colored segmentation
echo 2. Colors mean:
echo    - Green shades = walkable (road, sidewalk)
echo    - Red/Orange = obstacles (buildings, people, cars)
echo    - Blue = sky
echo    - Purple/Brown = vegetation, terrain
echo.
echo 3. Press 'q' in the window to close
echo.
pause

assistive-nav preview --camera 0 --seg-save-dir output/diagnostic

echo.
echo Diagnostic complete! Check output/diagnostic/ for saved frames.
echo.
pause
