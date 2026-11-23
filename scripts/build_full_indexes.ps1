# Builds the full-corpus indexes one after another (run from the repository root, laptop plugged in):
#   powershell -ExecutionPolicy Bypass -File scripts/build_full_indexes.ps1
# Each build is resumable: if it crashes or you stop it, run the script again and it continues with the first
# unfinished shard. A finished index is reported as "already complete" and skipped.
# Follow the progress of the current build in a second window:
#   Get-Content logs\v2_bgebase_bgesmall.log -Wait -Tail 5
$data = "data/processed/clean_movies.parquet"
$exclude = "evaluation/dev_queries.json evaluation/mood_queries.json"
New-Item -ItemType Directory -Force -Path logs | Out-Null

# name, field model, chunk model, extra flags. Run A is the main one; B and C are ablations (skip them if short of time).
$runs = @(
  @("v2_bgebase_bgesmall", "bge-base",   "bge-small",  ""),
  @("v2_bgebase_bgebase",  "bge-base",   "bge-base",   "--no-chunk-text"),
  @("v2_minilm_minilm",    "minilm-l6",  "minilm-l6",  "--no-chunk-text")
)
foreach ($r in $runs) {
    $name = $r[0]
    Write-Host "=== $name  started $(Get-Date)"
    $cmd = "python -m src.semantic_v2 train --data $data --output models/$name --field-model $($r[1]) --chunk-model $($r[2]) $($r[3]) --exclude-queries $exclude > logs\$name.log 2>&1"
    cmd /c $cmd
    Write-Host "=== $name  finished $(Get-Date)  (exit code $LASTEXITCODE)"
}
