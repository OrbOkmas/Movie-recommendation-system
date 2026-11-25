# Final run of semantic search v2 on the 100-movie sample + evaluation + results report.
# From the repository root:   powershell -ExecutionPolicy Bypass -File scripts/run_sample_eval.ps1
# Add -SkipFull to skip the (about 7 minute) re-run on the full-corpus index.
# Prerequisites: patch3 applied, evaluation/mood_judgments_v2.csv = the labeled sheet (all `relevance` filled),
# models/v2_bgebase_bgesmall (full index) present unless -SkipFull.
param([switch]$SkipFull)

function Step($title, [scriptblock]$cmd) {
    Write-Host ""; Write-Host "=== $title" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $title" -ForegroundColor Red; exit 1 }
}

$sampleIndex = "models/v2_sample_bge"
$judg = "evaluation/mood_judgments_v2.csv"
if (-not (Test-Path $judg)) { Write-Host "Missing $judg - copy the labeled sheet there first." -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path evaluation/results/v2 | Out-Null

Step "1/7 tests" { python -m pytest -q }

if (-not (Test-Path "$sampleIndex/manifest.json")) {
    Step "2/7 build the sample index (bge-base fields, bge-small chunks)" {
        python -m src.semantic_v2 train --data data/processed/sample_clean_movies.csv --output $sampleIndex `
            --field-model bge-base --chunk-model bge-small `
            --exclude-queries evaluation/dev_queries.json evaluation/mood_queries.json
    }
} else { Write-Host "=== 2/7 sample index exists - reusing it (delete $sampleIndex to rebuild)" -ForegroundColor Cyan }

Step "3/7 run the 28 mood queries (top-5); unjudged pairs are appended to the judgments sheet" {
    python -m src.semantic_v2 run --index $sampleIndex --queries evaluation/mood_queries.json `
        --name v2_A_sample --top-k 5 --config configs/v2_default.json --judgments $judg
}

$blank = python -c "import pandas as pd; j=pd.read_csv('$judg',encoding='utf-8-sig'); print(int(j.relevance.isna().sum()))"
if ([int]$blank -gt 0) {
    Write-Host "WARNING: $blank judged pairs have no relevance label yet (new pairs from this run). Label them, then run this script again." -ForegroundColor Yellow
}

Step "4/7 metrics: TF-IDF, semantic v1 variants and v2 on the same judgments" {
    python -m src.semantic_v2 metrics --judgments $judg --k 5 --out evaluation/results/v2/sample_metrics_raw.csv --rankings `
        tfidf=evaluation/results/tfidf_mood_rankings.csv sem_meta=evaluation/results/sem_meta_mood_rankings.csv `
        sem_subs_max=evaluation/results/sem_subs_max_mood_rankings.csv sem_subs_top3=evaluation/results/sem_subs_top3_mood_rankings.csv `
        sem_combined=evaluation/results/sem_combined_mood_rankings.csv v2=evaluation/results/v2/v2_A_sample_rankings.csv
}

Step "5/7 label-free diagnostics on the sample run" {
    python -m src.semantic_v2 diag --rankings evaluation/results/v2/v2_A_sample_rankings.csv --queries evaluation/mood_queries.json `
        --status evaluation/results/v2/v2_A_sample_query_status.csv | Out-File -Encoding utf8 evaluation/results/v2/v2_A_sample_diag.txt
}

Step "6/7 results report (docs/semantic_v2_results.md)" {
    python scripts/make_results_report.py --judgments $judg
}

if (-not $SkipFull) {
    if (Test-Path "evaluation/results/v2/v2_A_full_diag.txt") {
        Copy-Item evaluation/results/v2/v2_A_full_diag.txt evaluation/results/v2/v2_A_full_diag_before_patch3.txt -ErrorAction SilentlyContinue
    }
    Step "7/7 full-corpus run + diagnostics + calibration (about 7 min)" {
        python -m src.semantic_v2 run --index models/v2_bgebase_bgesmall --queries evaluation/dev_queries.json evaluation/mood_queries.json `
            --name v2_A_full --top-k 10 --config configs/v2_default.json
        python -m src.semantic_v2 diag --rankings evaluation/results/v2/v2_A_full_rankings.csv `
            --queries evaluation/dev_queries.json evaluation/mood_queries.json `
            --status evaluation/results/v2/v2_A_full_query_status.csv | Out-File -Encoding utf8 evaluation/results/v2/v2_A_full_diag.txt
    }
}
Write-Host ""; Write-Host "Done. Check docs/semantic_v2_results.md, then commit." -ForegroundColor Green
