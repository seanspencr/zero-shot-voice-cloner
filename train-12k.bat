@echo off
REM ---------------------------------------------------------------------------
REM  The real run: 12,000 utterances (~36%% of train-clean-100, ~220 speakers),
REM  4 epochs = 12,000 steps, with the configuration locked in decision4.md:
REM  LSGAN + d-every 1 + cepstral loss 5.0, no regularisation, best.pt by val MCD.
REM
REM  Step 1 caches the frozen features (resumable, ~3.7 h, ~17 GB).
REM  Step 2 trains (~2.8 h on 8 CPU cores at ~1.2 steps/s).
REM  Both write to logs\ so they can be tailed while they run.
REM ---------------------------------------------------------------------------
setlocal
set PY=C:\Users\NP25-1\AppData\Local\miniconda3\envs\speech_recog_aol\python.exe
set CACHE=datasets\cache\hubert-L9
set N=%1
if "%N%"=="" set N=12000

echo === [1/2] building the feature cache for %N% utterances ===
"%PY%" -u cache.py --data libritts --root datasets\libritts --split train-clean-100 ^
  --max-utterances %N% --content-source hubert --hubert-layer 9 --cache-dir %CACHE% ^
  --output "logs\cache_12k.log" 2>&1
if errorlevel 1 goto fail

echo === [2/2] training: %N% utterances x 4 epochs ===
"%PY%" -u train.py --data libritts --libritts-root datasets\libritts --split train-clean-100 ^
  --content-source hubert --speaker-source xtts --cache-dir %CACHE% ^
  --max-utterances %N% --epochs 4 --segment-seconds 4.0 --batch-size 4 ^
  --d-every 1 --lambda-cep 5.0 --lambda-mel 10.0 --lambda-detail 2.0 --lambda-spk 0.0 ^
  --val-split dev-clean --val-every 1 --val-utterances 24 --early-stop-patience 3 ^
  --log-every 200 --render-every 800 --save-dir checkpoints_12k ^
  --output "logs\train_12k.log" 2>&1
if errorlevel 1 goto fail

echo === done: checkpoints_12k\best.pt ===
echo next: eval.bat  (compare with the 42.705 dB / SECS 0.333 baseline)
goto :eof

:fail
echo FAILED with errorlevel %errorlevel%
exit /b 1
