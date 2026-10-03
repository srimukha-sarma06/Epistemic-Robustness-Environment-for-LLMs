@echo off
REM Run LLM inference from the repo root. Set HF_TOKEN (and optionally
REM API_BASE_URL / MODEL_NAME / API_ENV_URL) before running, e.g.:
REM   set HF_TOKEN=hf_...
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if "%API_BASE_URL%"=="" set API_BASE_URL=https://router.huggingface.co/v1
if "%MODEL_NAME%"=="" set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py --task all --episodes 1
pause
