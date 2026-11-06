@echo off
setlocal enableextensions
REM ==========================================================================
REM  train-full.bat  -  the whole corpus: all 33,232 utterances of
REM                     train-clean-100, 1 epoch  (~10 h on this CPU)
REM
REM  data    : every utterance in train-clean-100 (no --max-utterances)
REM  features: real HuBERT (layer 9) content + XTTS v2's own 512-d speaker encoder
REM
REM  Extra arguments are appended and override the defaults, e.g.
REM      train-full.bat --max-minutes 600        (stop after 10 hours)
REM      train-full.bat --epochs 2              (two passes = ~20 h)
REM      train-full.bat --max-utterances 8000   (between medium and full)
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
set "LOG=logs\train-full.log"

REM quieter libraries: no HF progress bars / warn spam in the log
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

echo ==========================================================================
echo  FULL run - ALL 33232 utterances, 2 epochs
echo    segments : 4.0 s     batch: 4
echo    steps    : 8308 per epoch
echo    val      : dev-clean, 40 utterances at every epoch end
echo    time     : roughly 10 HOURS on this CPU (~4.4 s per step)
echo    log      : %LOG%
echo.
echo  WARNING: HuBERT currently runs on the fly, so this is CPU-bound.
echo  There is no resume: if you stop, completed-epoch checkpoints are kept
echo  but a new run starts from scratch. Consider train-medium.bat first, or
echo  asking for the feature cache to speed this up 2-3x.
echo  Whole-corpus cache is ~46 GB; with 51 GB free prefer build-cache.bat --max-utterances 12000.

echo  Extra args: %*
echo ==========================================================================
echo.

if defined SKIP_CONFIRM goto :start
set /p CONFIRM="Type YES to start the ~10 hour run: "
if /i not "%CONFIRM%"=="YES" (
    echo Aborted.
    exit /b 0
)
:start

REM python's stderr is merged and stringified so PowerShell does not wrap it in
REM NativeCommandError records; Tee keeps the console live and writes the log.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u train.py --data libritts --content-source hubert --speaker-source xtts --segment-seconds 4.0 --batch-size 4 --d-every 2 --epochs 2 --cache-dir datasets\cache\hubert-L9 --val-split dev-clean --val-every 1 --val-utterances 40 --render-every 1 --log-every 200 --save-dir checkpoints %* 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOG%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"

echo.
echo --------------------------------------------------------------------------
echo Exit code: %RC%   (0 = the run completed normally)
echo   samples\epoch_*.wav     converted audio
echo   checkpoints\latest.pt   final model     checkpoints\best.pt  best epoch
echo   %LOG%
echo.
echo Evaluation, e.g.:
echo   python evaluate.py --checkpoint checkpoints\best.pt --split test-clean --max-utterances 100 --require-enrollment --segment-seconds 4.0 --out-dir eval_out
echo --------------------------------------------------------------------------
endlocal
exit /b %RC%
