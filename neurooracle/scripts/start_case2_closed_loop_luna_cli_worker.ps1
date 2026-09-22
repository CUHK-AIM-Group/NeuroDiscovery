[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [Parameter(Mandatory = $true)]
    [string]$ThreadId,
    [Parameter(Mandatory = $true)]
    [string]$ProviderWorkspace,
    [string]$HostId = "local",
    [string]$Python = "C:\Users\45846\Documents\Code\NeuroClaw\.venv-case2-benchmark\Scripts\python.exe",
    [string]$CodexExecutable = "",
    [string]$ResponseSchema = "",
    [int]$RequestTimeoutSeconds = 7200,
    [int]$MaxAttempts = 8
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python runtime is missing" }
if ([string]::IsNullOrWhiteSpace($CodexExecutable)) {
    $CodexExecutable = (Get-Command codex -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $CodexExecutable -PathType Leaf)) { throw "Codex CLI is missing" }
if (-not (Test-Path -LiteralPath $ProviderWorkspace -PathType Container)) {
    throw "Provider workspace is missing"
}

$resolvedDataDir = [System.IO.Path]::GetFullPath($DataDir)
$resolvedProviderWorkspace = [System.IO.Path]::GetFullPath($ProviderWorkspace)
New-Item -ItemType Directory -Path $resolvedDataDir -Force | Out-Null
$stdoutPath = Join-Path $resolvedDataDir "worker.stdout.log"
$stderrPath = Join-Path $resolvedDataDir "worker.stderr.log"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($ResponseSchema)) {
    $ResponseSchema = Join-Path $repoRoot "neurooracle\configs\case2_luna_response_envelope.schema.json"
}
if (-not (Test-Path -LiteralPath $ResponseSchema -PathType Leaf)) { throw "Response schema is missing" }
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "-m", "neurooracle.scripts.case2_closed_loop_luna_cli_worker",
        "--data-dir", ('"' + $resolvedDataDir.Replace('"', '\"') + '"'),
        "--thread-id", $ThreadId,
        "--host-id", $HostId,
        "--provider-workspace", ('"' + $resolvedProviderWorkspace.Replace('"', '\"') + '"'),
        "--codex-executable", ('"' + $CodexExecutable.Replace('"', '\"') + '"'),
        "--response-schema", ('"' + $ResponseSchema.Replace('"', '\"') + '"'),
        "--request-timeout-seconds", [string]$RequestTimeoutSeconds,
        "--max-attempts", [string]$MaxAttempts
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$deadline = (Get-Date).AddSeconds(15)
do {
    Start-Sleep -Milliseconds 250
    if ($process.HasExited) {
        throw "Luna CLI worker exited during startup; inspect $stderrPath"
    }
    $ready = Test-Path -LiteralPath $stdoutPath -PathType Leaf
    if ($ready) {
        $firstLine = Get-Content -LiteralPath $stdoutPath -TotalCount 1 -ErrorAction SilentlyContinue
        $ready = $firstLine -like '*"status": "ready"*'
    }
} until ($ready -or (Get-Date) -gt $deadline)
if (-not $ready) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "Luna CLI worker did not become ready"
}

$manifest = [ordered]@{
    schema_version = "neuroclaw.case2-closed-loop-luna-cli-worker-runtime.v1"
    benchmark_id = "case2_adni_closed_loop_benchmark_v3_luna_seed1"
    model = "gpt-5.6-luna"
    reasoning_effort = "max"
    response_capture_mode = "codex-cli output-last-message direct capture"
    data_dir = $resolvedDataDir
    provider_workspace = $resolvedProviderWorkspace
    codex_executable = $CodexExecutable
    response_schema = $ResponseSchema
    response_schema_sha256 = (Get-FileHash -LiteralPath $ResponseSchema -Algorithm SHA256).Hash.ToLowerInvariant()
    thread_id = $ThreadId
    host_id = $HostId
    pid = $process.Id
    max_attempts = $MaxAttempts
    request_timeout_seconds = $RequestTimeoutSeconds
    provider_api_calls = $false
    provider_keys_required = $false
    manual_response_content_edit = $false
    provider_boundary_enforcement = "per-turn prompt plus complete command-log audit"
    local_approval_and_sandbox_bypassed = $true
    created_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$manifestPath = Join-Path $resolvedDataDir "worker_runtime_manifest.json"
[System.IO.File]::WriteAllText(
    $manifestPath,
    (($manifest | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
$manifest | Add-Member -NotePropertyName runtime_manifest_path -NotePropertyValue $manifestPath
$manifest
