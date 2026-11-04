@echo off
setlocal enableextensions
REM ==========================================================================
REM  eval.bat  -  convert 20 held-out utterances and report MCD + SECS
REM
REM  split      : test-clean (speakers the model never saw)
REM  enrolment  : a DIFFERENT clip of the same target speaker (true zero-shot)
REM  segments   : 4.0 s      features: HuBERT + XTTS speaker encoder
REM
REM  Outputs
REM    eval_out\<utt>_gen.wav    converted audio
REM    eval_out\<utt>_ref.wav    the original crop, so you can A/B by ear
REM    eval_out\report.csv/.json MCD, mel-L1, SECS, transcript per utterance
REM    logs\eval.log             full console output
REM
REM  Extra arguments are appended and override the defaults, e.g.
REM      eval.bat --max-utterances 50
REM      eval.bat --checkpoint checkpoints\latest.pt
REM      eval.bat --split dev-clean
REM      eval.bat --segment-seconds 2.0
REM ==========================================================================
cd /d "%~dp0"

REM --- interpreter (edit if the conda env moves) ---
set "PY=C:\Users\NP25-1\AppData\Local\miniconda3\envs\speech_recog_aol\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found at: %PY%
    echo         Edit the PY variable at the top of this file.
    exit /b 1
)

if not exist "datasets\libritts\test-clean" (
    echo [ERROR] LibriTTS not found: datasets\libritts\test-clean
    exit /b 1
)

if not exist "checkpoints\best.pt" (
    echo [ERROR] No checkpoint found at checkpoints\best.pt
    echo         Train first with train-short.bat, or pass --checkpoint ^<file^>.
    exit /b 1
)

if not exist "logs" mkdir "logs"
set "LOG=logs\eval.log"

REM quieter libraries: no HF progress bars / warn spam in the log
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

echo ==========================================================================
echo  EVAL - 20 utterances from test-clean (unseen speakers)
echo    enrolment: different clip of the same speaker (true zero-shot)
echo    segments : 4.0 s     checkpoint: checkpoints\best.pt
echo    time     : roughly 2 minutes on this CPU
echo    log      : %LOG%
echo  Extra args: %*
echo ==========================================================================
echo.

REM python's stderr is merged and stringified so PowerShell does not wrap it in
REM NativeCommandError records; Tee keeps the console live and writes the log.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PY%' -u evaluate.py --checkpoint checkpoints\best.pt --split test-clean --max-utterances 20 --require-enrollment --segment-seconds 4.0 --content-source hubert --speaker-source xtts --out-dir eval_out %* 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath '%LOG%'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"

echo.
echo --------------------------------------------------------------------------
echo Exit code: %RC%   (0 = evaluation completed normally)
echo   eval_out\*_gen.wav      converted audio
echo   eval_out\*_ref.wav      original crop for comparison
echo   eval_out\report.csv     per-utterance MCD / mel-L1 / SECS + sentence
echo   eval_out\report.json    same, machine readable
echo   %LOG%
echo.
echo Listen to a pair side by side, e.g.:
echo   start "" eval_out\^<utt_id^>_ref.wav
echo   start "" eval_out\^<utt_id^>_gen.wav
echo --------------------------------------------------------------------------
endlocal
exit /b %RC%
