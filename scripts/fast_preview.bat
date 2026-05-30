@echo off
cd /d %USERPROFILE%\Projects\assistive-navigation
call .venv\Scripts\activate.bat
.venv\Scripts\assistive-nav.exe preview --fast --camera 0
