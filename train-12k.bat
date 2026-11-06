@echo off
setlocal enableextensions
REM ==========================================================================
REM  train-12k.bat  -  THE REAL RUN (decision4.md section 5.1)
REM
REM  Step 1  cache frozen features for 12,000 utterances (~3.7 h, ~17 GB, resumable)
REM  Step 2  train 12,000 utterances x 4 epochs = 12,000 steps (~2.8 h)
REM
REM  Config is the champion locked in decision4.md section 4:
REM    LSGAN, d-every 1, mel 10 + cepstral 5 + detail 2, no dropout/weight-decay/augment,
REM    best.pt selected by validation MCD, early-stop patience 3.
REM
REM  Extra arguments are appended to step 2 and override the defaults, e.g.
REM      set N_UTTS=12000 ^& train-12k.bat --val-utterances 40
REM  Set SKIP_CONFIRM=1 to run unattended.
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

REM size comes from the environment: cmd's %* does NOT shrink after `shift`,
REM so a positional size argument would leak into the training command line.
if "%N_UTTS%"=="" (set "N=12000") else (set "N=%N_UTTS%")

if not exist "logs" mkdir "logs"
set "CACHE=datasets\cache\hubert-L9"
set "LOGC=logs\cache_12k.log"
set "LOGT=logs\train_12k.log"

set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

echo ==========================================================================
echo  FULL run - %N% utterances, 4 epochs
echo    step 1 : build feature cache        log: %LOGC%
echo    step 2 : train                      log: %LOGT%
echo    budget : ~3.7 h cache + ~2.8 h train (CPU, ~1.2 steps/s)
echo  Extra args: %*
echo ==========================================================================
echo.

if defined SKIP_CONFIRM goto :cache
set /p CONFIRM="Type YES to start the multi-hour run: "
if /i not "%CONFIRM%"=="YES" (
    echo Aborted.
    exit /b 0
)

:cache
echo === [1/2] feature cache: %N% utterances ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u cache.py --split train-clean-100 --libritts-root datasets\libritts --max-utterances %N% --content-source hubert --hubert-layer 9 --scan-jobs 16 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOGC%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [ERROR] cache build failed with %RC%. Re-run the same command; it resumes.
    exit /b %RC%
)

echo.
echo === [2/2] training: %N% utterances x 4 epochs ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u train.py --data libritts --content-source hubert --speaker-source xtts --segment-seconds 4.0 --batch-size 4 --d-every 1 --lambda-cep 5.0 --lambda-mel 10.0 --lambda-detail 2.0 --lambda-spk 0.0 --max-utterances %N% --epochs 4 --cache-dir %CACHE% --val-split dev-clean --val-every 1 --val-utterances 24 --early-stop-patience 3 --log-every 200 --render-every 800 --save-dir checkpoints_12k %* 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOGT%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"

echo.
echo --------------------------------------------------------------------------
echo Exit code: %RC%   (0 = the run completed normally)
echo   checkpoints_12k\best.pt     best epoch by validation MCD
echo   checkpoints_12k\latest.pt   final epoch
echo   %LOGC%   %LOGT%
echo.
echo Gates from decision4.md section 5.2 (val MCD ^< 34 dB, eval MCD ^< 38 dB):
echo   python evaluate.py --checkpoint checkpoints_12k\best.pt --split test-clean --max-utterances 20 --require-enrollment --segment-seconds 4.0 --out-dir eval_out
echo --------------------------------------------------------------------------
endlocal
exit /b %RC%
