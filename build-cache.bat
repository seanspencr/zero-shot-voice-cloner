@echo off
setlocal enableextensions
REM ==========================================================================
REM  build-cache.bat  -  extract HuBERT + XTTS + F0 features ONCE, so training
REM                      steps no longer pay ~1 s of extraction each.
REM
REM  Measured on this machine: 0.90 utt/s, ~1.38 MB per utterance.
REM      1.000 utt ->  ~19 min,  1,4 GB
REM      5.000 utt ->  ~93 min,  6,9 GB
REM     12.000 utt ->  ~3,7 jam, 17 GB
REM  It is RESUMABLE: re-running skips what is already cached, so you can grow
REM  the cache in chunks.
REM
REM  Usage:
REM      build-cache.bat                        (default 5000 utterances)
REM      build-cache.bat --max-utterances 12000
REM      build-cache.bat --split dev-clean --max-utterances 1000
REM ==========================================================================
cd /d "%~dp0"

set "PY=C:\Users\NP25-1\AppData\Local\miniconda3\envs\speech_recog_aol\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found at: %PY%
    exit /b 1
)
if not exist "datasets\libritts\train-clean-100" (
    echo [ERROR] LibriTTS not found: datasets\libritts\train-clean-100
    exit /b 1
)
if not exist "logs" mkdir "logs"
set "LOG=logs\build-cache.log"
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

echo ==========================================================================
echo  FEATURE CACHE BUILD
echo    target : datasets\cache\hubert-L9   (resumable, re-run to extend)
echo    log    : %LOG%
echo  Extra args: %*
echo ==========================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u cache.py --split train-clean-100 --max-utterances 5000 --min-seconds 4.0 --progress-every 200 %* 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOG%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"
echo.
echo Exit code: %RC%   (0 = cache complete or extended)
echo Now train with:  train-medium.bat   (it picks up the cache automatically)
endlocal
exit /b %RC%
