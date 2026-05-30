@echo off
cd /d %USERPROFILE%\Projects\assistive-navigation
call .venv\Scripts\activate.bat
.venv\Scripts\assistive-nav.exe run --demo --show-seg --camera 0
