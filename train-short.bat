@echo off
setlocal enableextensions
REM ==========================================================================
REM  train-short.bat  -  quick real-data run, mainly to prove the pipeline
REM                       works end to end (~10-15 min on this CPU)
REM
REM  data    : 400 utterances from LibriTTS train-clean-100
REM  features: real HuBERT (layer 9) content + XTTS v2's own 512-d speaker encoder
REM  mel     : from the installed HiFi-GAN vocoder's own AudioProcessor
REM
REM  Extra arguments are appended and override the defaults, e.g.
REM      train-short.bat --epochs 4 --max-minutes 25
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
    echo         Extract the corpus first, or pass --libritts-root.
    exit /b 1
)

if not exist "logs" mkdir "logs"
set "LOG=logs\train-short.log"

REM quieter libraries: no HF progress bars / warn spam in the log
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

echo ==========================================================================
echo  SHORT run - 800 utterances, 4 epochs (cache-backed smoke test, ~10 min limit)
echo    segments : 4.0 s     batch: 4
echo    val      : dev-clean, 16 utterances each epoch
echo    stops    : after 10 minutes (--max-minutes 10)
echo    log      : %LOG%
echo  Expect ~200 steps; mel L1 should fall from ~2.0 to ~1.0.
echo  At ~1.2 steps/s once cached this is a quick sanity run, not a convergence run.

echo  Extra args: %*
echo ==========================================================================
echo.

REM python's stderr is merged and stringified so PowerShell does not wrap it in
REM NativeCommandError records; Tee keeps the console live and writes the log.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u train.py --data libritts --content-source hubert --speaker-source xtts --segment-seconds 4.0 --batch-size 4 --d-every 2 --max-utterances 800 --epochs 4 --cache-dir datasets\cache\hubert-L9 --val-split dev-clean --val-every 1 --val-utterances 16 --render-every 1 --log-every 25 --save-dir checkpoints --max-minutes 10 %* 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOG%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"

echo.
echo --------------------------------------------------------------------------
echo Exit code: %RC%   (0 = the run completed normally)
echo   samples\epoch_*.wav     converted audio (source sentence is printed above)
echo   checkpoints\latest.pt   model after the last epoch
echo   checkpoints\best.pt     model with the best mean G loss
echo   %LOG%
echo.
echo Evaluate on unseen speakers with evaluate.py, e.g.:
echo   python evaluate.py --checkpoint checkpoints\best.pt --split test-clean --max-utterances 50 --require-enrollment --segment-seconds 4.0 --out-dir eval_out
echo --------------------------------------------------------------------------
endlocal
exit /b %RC%
