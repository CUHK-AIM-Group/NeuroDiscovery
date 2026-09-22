param(
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"

$python = "C:\Users\45846\anaconda3\envs\neuroclaw\python.exe"
$repo = "C:\Users\45846\Documents\Code\NeuroClaw"
$pilot = Join-Path $repo "core\scripts\case1_multimodel_pilot.py"
$case1Runner = Join-Path $repo "core\scripts\run_case1_multimodel_10x3.py"

$case3Dir = "\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v8_biomarker_independent\biomarker_discovery\model_robustness\cs3_5atlas_9disease_8model_10seed_siteaware"
$case3Performance = Join-Path $case3Dir "performance_folds.csv"
$case3Failures = Join-Path $case3Dir "failed_model_folds.csv"
$case3Manifest = Join-Path $case3Dir "manifest.json"
$case3RunLog = Join-Path $case3Dir "resume_10seed_handoff.log"

$case1Dir = "\\192.168.3.61\data\Public Dataset\case1_multimodel_10x3"
$case1Performance = Join-Path $case1Dir "performance_folds.csv"
$case1Failures = Join-Path $case1Dir "failed_model_folds.csv"
$case1Manifest = Join-Path $case1Dir "manifest.json"
$case1RunLog = Join-Path $case1Dir "orchestrator.log"
$monitorLog = Join-Path $case1Dir "handoff_monitor.log"

function Write-Monitor([string]$message) {
    Add-Content -LiteralPath $monitorLog -Value ("[{0:o}] {1}" -f (Get-Date), $message) -Encoding utf8
}

function Get-RowCount([string]$path) {
    if (!(Test-Path -LiteralPath $path)) { return 0 }
    return [Math]::Max(0, (Get-Content -LiteralPath $path | Measure-Object -Line).Lines - 1)
}

function Get-UnresolvedFailureCount([string]$path) {
    if (!(Test-Path -LiteralPath $path)) { return 0 }
    try {
        return @(Import-Csv -LiteralPath $path).Count
    } catch {
        return 1
    }
}

function Find-PythonProcess([string]$scriptName, [string]$runToken) {
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*$scriptName*" -and
        $_.CommandLine -like "*$runToken*"
    }) | Select-Object -First 1
}

function Test-Case3Complete {
    if (!(Test-Path -LiteralPath $case3Performance) -or !(Test-Path -LiteralPath $case3Manifest)) {
        return $false
    }
    $rows = @(Import-Csv -LiteralPath $case3Performance)
    if ($rows.Count -ne 10800) { return $false }
    $unique = @($rows | ForEach-Object {
        "$($_.atlas)|$($_.disease)|$($_.model)|$($_.seed)|$($_.fold)"
    } | Sort-Object -Unique).Count
    if ($unique -ne 10800 -or (Get-UnresolvedFailureCount $case3Failures) -ne 0) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $case3Manifest -Raw | ConvertFrom-Json
        return (
            [int]$manifest.n_model_folds -eq 10800 -and
            [int]$manifest.n_failed_model_folds -eq 0 -and
            [int]$manifest.folds -eq 3 -and
            @($manifest.seeds).Count -eq 10 -and
            @($manifest.atlases).Count -eq 5 -and
            @($manifest.diseases).Count -eq 9 -and
            @($manifest.models).Count -eq 8
        )
    } catch {
        return $false
    }
}

function Test-Case1Complete {
    if (!(Test-Path -LiteralPath $case1Performance) -or !(Test-Path -LiteralPath $case1Manifest)) {
        return $false
    }
    $rows = @(Import-Csv -LiteralPath $case1Performance)
    if ($rows.Count -ne 41040) { return $false }
    $unique = @($rows | ForEach-Object {
        "$($_.atlas)|$($_.disease)|$($_.model)|$($_.seed)|$($_.fold)"
    } | Sort-Object -Unique).Count
    if ($unique -ne 41040 -or (Get-UnresolvedFailureCount $case1Failures) -ne 0) {
        return $false
    }
    if (!(Test-Path -LiteralPath (Join-Path $case1Dir "heldout_attribution_summary.csv"))) {
        return $false
    }
    try {
        $manifest = Get-Content -LiteralPath $case1Manifest -Raw | ConvertFrom-Json
        return (
            $manifest.status -eq "complete" -and
            [int]$manifest.n_model_folds -eq 41040 -and
            [int]$manifest.n_attribution_model_folds -eq 41040 -and
            [int]$manifest.n_failed_model_folds -eq 0 -and
            [int]$manifest.folds -eq 3 -and
            @($manifest.seeds).Count -eq 10 -and
            @($manifest.atlases).Count -eq 19 -and
            @($manifest.diseases).Count -eq 9 -and
            @($manifest.models).Count -eq 8 -and
            [int]$manifest.n_attribution_rows -gt 0
        )
    } catch {
        return $false
    }
}

function Invoke-Case3Resume {
    & $python $pilot `
        --case-study-id biomarker_discovery `
        --transdiag-root "\\192.168.3.61\data\Public Dataset\transdiag_preprocessed" `
        --out-root "\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v8_biomarker_independent\biomarker_discovery\model_robustness" `
        --run-name cs3_5atlas_9disease_8model_10seed_siteaware `
        --atlases "aal_116,cc200,glasser_360,harvard_oxford_sub,schaefer_200_7net" `
        --diseases "ADHD,MDD_depression,OCD_OC_related,PTSD_trauma,anxiety,bipolar,eating_disorder,psychosis_SZ_SZA,substance_use" `
        --models "elasticnet,roi_mlp,brainnetcnn,braingnn,bnt,ibgnn,lggnn,combraintf" `
        --folds 3 `
        --seeds "20260816,20260817,20260818,20260819,20260820,20260821,20260822,20260823,20260824,20260825" `
        --epochs 24 `
        --patience 5 `
        --batch-size 8 `
        --lr 0.001 `
        --weight-decay 0.0001 `
        --graph-density 0.15 `
        --device cuda `
        --split-stratification diagnosis_x_site_if_feasible `
        --protocol-manifest "\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v8_biomarker_independent\biomarker_discovery\tables\table_manifest.json" `
        --resume *>> $case3RunLog
    return $LASTEXITCODE
}

function Invoke-Case1Resume {
    Set-Location -LiteralPath $repo
    & $python $case1Runner --output-root $case1Dir *>> $case1RunLog
    return $LASTEXITCODE
}

$createdNew = $false
$mutex = [Threading.Mutex]::new(
    $true,
    "Global\NeuroClaw_Case3_CS1_FullExperiment",
    [ref]$createdNew
)
if (!$createdNew) {
    Write-Monitor "A full-experiment monitor already owns the singleton lock; exiting pid=$PID"
    $mutex.Dispose()
    exit 0
}

try {
    while (!(Test-Case3Complete)) {
        $running = Find-PythonProcess "case1_multimodel_pilot.py" "cs3_5atlas_9disease_8model_10seed_siteaware"
        if ($running) {
            Write-Monitor "Case3 running pid=$($running.ProcessId) rows=$(Get-RowCount $case3Performance)/10800"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        Write-Monitor "Case3 incomplete with no active process; resuming"
        $exitCode = Invoke-Case3Resume
        Write-Monitor "Case3 resume exited code=$exitCode rows=$(Get-RowCount $case3Performance)/10800"
        Start-Sleep -Seconds 10
    }

    Write-Monitor "Case3 completion gate passed: 10800 unique folds, zero failures, manifest verified"
    while (!(Test-Case1Complete)) {
        $running = Find-PythonProcess "run_case1_multimodel_10x3.py" "case1_multimodel_10x3"
        if (!$running) {
            $running = Find-PythonProcess "case1_multimodel_pilot.py" "case1_multimodel_10x3"
        }
        if ($running) {
            Write-Monitor "Case1 running pid=$($running.ProcessId) aggregate_rows=$(Get-RowCount $case1Performance)/41040"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        Write-Monitor "Case1 incomplete with no active process; starting/resuming orchestrator"
        $exitCode = Invoke-Case1Resume
        Write-Monitor "Case1 orchestrator exited code=$exitCode aggregate_rows=$(Get-RowCount $case1Performance)/41040"
        Start-Sleep -Seconds 10
    }
    Write-Monitor "Case1 completion gate passed: 41040 unique folds, zero failures, attribution and manifest verified"
} finally {
    try { $mutex.ReleaseMutex() } catch {}
    $mutex.Dispose()
}
