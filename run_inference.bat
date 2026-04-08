@echo off
cd /d D:\apps\HackScalar
call .\venv\Scripts\activate.bat
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o
set HF_TOKEN=your-api-key-here
python inference.py --task factual_resistance --episodes 1
pause