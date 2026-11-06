@echo off
setlocal enableextensions
REM ==========================================================================
REM  train-medium.bat  -  a real training session on a slice of train-clean-100
REM                       (~1.3 h on this CPU)
REM
REM  data    : 2000 utterances, 2 epochs  (500 steps per epoch)
REM  features: real HuBERT (layer 9) content + XTTS v2's own 512-d speaker encoder
REM
REM  Extra arguments are appended and override the defaults, e.g.
REM      train-medium.bat --epochs 4
REM      train-medium.bat --segment-seconds 2 --epochs 6     (cheaper per step)
REM ==========================================================================
cd /d "%~dp0"

REM --- interpreter (edit if the conda env moves) ---
set "PY=C:\Users\NP25-1\AppData\Local\miniconda3\envs\speech_recog_aol\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found at: %PY%
    echo         Edit the PY variable at the top of this file.
    exit /b 1
)

if not exist "datasets\libritts\train-clean-100" (
    echo [ERROR] LibriTTS not found: datasets\libritts\train-clean-100
    exit /b 1
)

if not exist "logs" mkdir "logs"
set "LOG=logs\train-medium.log"

REM quieter libraries: no HF progress bars / warn spam in the log
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

echo ==========================================================================
echo  MEDIUM run - 5000 utterances, 6 epochs
echo    segments : 4.0 s     batch: 4
echo    val      : dev-clean, 24 utterances each epoch
echo    time     : about 1.3 hours on this CPU (~4.4 s per step)
echo    log      : %LOG%
echo  No time limit is set. Close the window or press Ctrl+C to stop;
echo  checkpoints from completed epochs stay on disk (there is no resume yet).
echo  ~1.7 h after build-cache.bat has cached 5000 utterances. Resumable with --resume.

echo  Extra args: %*
echo ==========================================================================
echo.

if defined SKIP_CONFIRM goto :start
set /p CONFIRM="Type YES to start the ~1.3 hour run: "
if /i not "%CONFIRM%"=="YES" (
    echo Aborted.
    exit /b 0
)
:start

REM python's stderr is merged and stringified so PowerShell does not wrap it in
REM NativeCommandError records; Tee keeps the console live and writes the log.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u train.py --data libritts --content-source hubert --speaker-source xtts --segment-seconds 4.0 --batch-size 4 --d-every 1 --lambda-cep 5.0 --lambda-mel 10.0 --lambda-detail 2.0 --max-utterances 5000 --epochs 6 --cache-dir datasets\cache\hubert-L9 --val-split dev-clean --val-every 2 --val-utterances 32 --render-every 1 --log-every 100 --save-dir checkpoints %* 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOG%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"

echo.
echo --------------------------------------------------------------------------
echo Exit code: %RC%   (0 = the run completed normally)
echo   samples\epoch_*.wav     converted audio
echo   checkpoints\latest.pt   final model     checkpoints\best.pt  best epoch
echo   %LOG%
echo.
echo Evaluation, e.g.:
echo   python evaluate.py --checkpoint checkpoints\best.pt --split test-clean --max-utterances 50 --require-enrollment --segment-seconds 4.0 --out-dir eval_out
echo --------------------------------------------------------------------------
endlocal
exit /b %RC%
