@echo off
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -e ".[dev]"
) else (
  call .venv\Scripts\activate.bat
)
if not exist tests\fixtures\sample.jpg (
  .venv\Scripts\python.exe scripts\create_sample_fixture.py
)
.venv\Scripts\assistive-nav.exe run --dry-run --image tests\fixtures\sample.jpg
