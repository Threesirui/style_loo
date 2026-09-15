param(
    [string]$Python = "C:\Users\three\.conda\envs\TextWave\python.exe",
    [string]$OutputDir = ".\outputs\baselines_official"
)

$ErrorActionPreference = "Stop"

$runs = @(
    @("m4", "multilingual"),
    @("deepfake", "cross_domains_cross_models"),
    @("raid", "clean")
)

foreach ($run in $runs) {
    $dataset, $scenario = $run
    & $Python .\baselines\run.py `
        --dataset $dataset `
        --scenario $scenario `
        --method xlm_roberta_base_ft `
        --threshold-mode native `
        --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "XLM-R baseline failed for $dataset/$scenario with exit code $LASTEXITCODE"
    }
}
