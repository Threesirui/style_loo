$ErrorActionPreference = "Stop"

$upstreamRoot = Join-Path $PSScriptRoot "upstream"
$files = @(
    @("mbzuai-nlp/SemEval2024-task8", "d8350c840bc505eaba06b4baf69993c2d18fef5e", "LICENSE", "SemEval2024-task8-main/LICENSE"),
    @("mbzuai-nlp/SemEval2024-task8", "d8350c840bc505eaba06b4baf69993c2d18fef5e", "README.md", "SemEval2024-task8-main/README.md"),
    @("mbzuai-nlp/SemEval2024-task8", "d8350c840bc505eaba06b4baf69993c2d18fef5e", "subtaskA/baseline/transformer_baseline.py", "SemEval2024-task8-main/subtaskA/baseline/transformer_baseline.py"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "LICENSE", "fast-detect-gpt-main/LICENSE"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "README.md", "fast-detect-gpt-main/README.md"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "scripts/local_infer.py", "fast-detect-gpt-main/scripts/local_infer.py"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "scripts/fast_detect_gpt.py", "fast-detect-gpt-main/scripts/fast_detect_gpt.py"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "scripts/model.py", "fast-detect-gpt-main/scripts/model.py"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "scripts/data_builder.py", "fast-detect-gpt-main/scripts/data_builder.py"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "scripts/custom_datasets.py", "fast-detect-gpt-main/scripts/custom_datasets.py"),
    @("baoguangsheng/fast-detect-gpt", "971b05202bac2bb504d60c0ac0812fea7a8f7c82", "scripts/metrics.py", "fast-detect-gpt-main/scripts/metrics.py"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "LICENSE.md", "Binoculars-main/LICENSE.md"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "README.md", "Binoculars-main/README.md"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "requirements.txt", "Binoculars-main/requirements.txt"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "binoculars/__init__.py", "Binoculars-main/binoculars/__init__.py"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "binoculars/detector.py", "Binoculars-main/binoculars/detector.py"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "binoculars/metrics.py", "Binoculars-main/binoculars/metrics.py"),
    @("ahans30/Binoculars", "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8", "binoculars/utils.py", "Binoculars-main/binoculars/utils.py")
)

foreach ($item in $files) {
    $repo, $commit, $source, $relativeTarget = $item
    $target = Join-Path $upstreamRoot $relativeTarget
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    $url = "https://raw.githubusercontent.com/$repo/$commit/$source"
    Write-Host "Fetching $repo@$($commit.Substring(0, 12))/$source"
    Invoke-WebRequest -Uri $url -OutFile $target
}

$checks = @(
    @("SemEval2024-task8-main/subtaskA/baseline/transformer_baseline.py", "5c7574d7f43e6bb2f4c00cb1913d46e9c5b5cb4624e395766d5ed346c453c1bc"),
    @("fast-detect-gpt-main/scripts/local_infer.py", "4cbe06cdb456df729ce9eede14505757775bc388f4c17d762a6b28f57b169cb0"),
    @("fast-detect-gpt-main/scripts/fast_detect_gpt.py", "70fcdafca0182a4e5f9773d82f435cf06610f291bca60ff726ef0d2ff255a929"),
    @("fast-detect-gpt-main/scripts/custom_datasets.py", "163fc8d576251da5a5ee22ddd3c909c30feea0686f29a7ed25765b7d2b8ebb74"),
    @("Binoculars-main/binoculars/detector.py", "9724932e34a78e019c949ee986b34593f66e58f5975c9e76e359621ee0d1aade")
)

foreach ($check in $checks) {
    $relativePath, $expected = $check
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $upstreamRoot $relativePath)).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "Checksum mismatch for $relativePath (expected $expected, got $actual)"
    }
}

Write-Host "Official upstream source files fetched and verified."
