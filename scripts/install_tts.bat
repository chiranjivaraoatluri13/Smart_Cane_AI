@echo off
cd /d %USERPROFILE%\Projects\assistive-navigation
call .venv\Scripts\activate.bat
pip install -e ".[tts]"
.venv\Scripts\python.exe -c "import pyttsx3; e=pyttsx3.init(); e.say('TTS is working'); e.runAndWait(); print('OK')"
