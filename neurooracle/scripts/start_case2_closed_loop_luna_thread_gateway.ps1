[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [Parameter(Mandatory = $true)]
    [string]$ThreadId,
    [string]$HostId = "local",
    [string]$Python = "C:\Users\45846\Documents\Code\NeuroClaw\.venv-case2-benchmark\Scripts\python.exe",
    [int]$Port = 18083,
    [string]$HostAddress = "127.0.0.1",
    [int]$ResponseTimeoutSeconds = 7200,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$benchmarkId = "case2_adni_closed_loop_benchmark_v3_luna_seed1"
$model = "gpt-5.6-luna"
$reasoningEffort = "max"

if ($HostAddress -notin @("127.0.0.1", "localhost", "::1")) {
    throw "The replay gateway may bind only to loopback"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python runtime is missing" }

$resolvedDataDir = [System.IO.Path]::GetFullPath($DataDir)
New-Item -ItemType Directory -Path $resolvedDataDir -Force | Out-Null

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    if (-not $Restart) { throw "Port $Port is already in use" }
    foreach ($listener in $listeners) {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
        $commandLine = [string]$processInfo.CommandLine
        if (
            $commandLine -notmatch "case2_closed_loop_luna_thread_gateway" -or
            $commandLine -notlike "*$resolvedDataDir*"
        ) {
            throw "Refusing to stop unrelated process $($listener.OwningProcess) on port $Port"
        }
        Stop-Process -Id ([int]$listener.OwningProcess) -Force
    }
    Start-Sleep -Seconds 1
}

$stdoutPath = Join-Path $resolvedDataDir "gateway.stdout.log"
$stderrPath = Join-Path $resolvedDataDir "gateway.stderr.log"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$quotedDataDir = '"' + $resolvedDataDir.Replace('"', '\"') + '"'
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "-m", "neurooracle.scripts.case2_closed_loop_luna_thread_gateway",
        "--host", $HostAddress, "--port", [string]$Port,
        "--data-dir", $quotedDataDir,
        "--thread-id", $ThreadId,
        "--host-id", $HostId,
        "--timeout-seconds", [string]$ResponseTimeoutSeconds
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$healthUrl = "http://$HostAddress`:$Port/health"
$deadline = (Get-Date).AddSeconds(30)
$health = $null
do {
    Start-Sleep -Milliseconds 500
    try { $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3 }
    catch { if ($process.HasExited) { break } }
} until ($null -ne $health -or (Get-Date) -gt $deadline)

$healthy = (
    $null -ne $health -and $health.status -eq "ok" -and
    $health.benchmark_id -eq $benchmarkId -and
    $health.model -eq $model -and
    $health.forced_reasoning_effort -eq $reasoningEffort -and
    $health.execution_channel -eq "codex_thread_replay" -and
    [int]$health.max_concurrent_upstream_requests -eq 1 -and
    -not [bool]$health.provider_api_calls -and
    -not [bool]$health.provider_keys_required -and
    [bool]$health.request_hash_lock -and
    [bool]$health.response_hash_lock -and
    $health.dispatch_delivery_mode -eq "message-specified single read-only dispatch file" -and
    $health.response_capture_mode -eq "codex-cli output-last-message direct capture" -and
    $health.thread_id -eq $ThreadId -and
    $health.host_id -eq $HostId -and
    -not [bool]$health.secrets_exposed
)
if (-not $healthy) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "Closed-loop Luna thread replay gateway failed its health check"
}

$manifest = [ordered]@{
    schema_version = "neuroclaw.case2-closed-loop-luna-thread-gateway-runtime.v1"
    benchmark_id = $benchmarkId
    model = $model
    reasoning_effort = $reasoningEffort
    base_url = "http://$HostAddress`:$Port/v1"
    health_url = $healthUrl
    data_dir = $resolvedDataDir
    pid = $process.Id
    execution_channel = "codex_thread_replay"
    thread_id = $ThreadId
    host_id = $HostId
    temperature_control = "not_exposed_by_codex_thread"
    max_concurrent_upstream_requests = 1
    request_hash_lock = $true
    response_hash_lock = $true
    dispatch_delivery_mode = "message-specified single read-only dispatch file"
    response_capture_mode = "codex-cli output-last-message direct capture"
    provider_api_calls = $false
    provider_keys_required = $false
    credentials_persisted = $false
    created_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$manifestPath = Join-Path $resolvedDataDir "runtime_manifest.json"
[System.IO.File]::WriteAllText(
    $manifestPath,
    (($manifest | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
$manifest | Add-Member -NotePropertyName runtime_manifest_path -NotePropertyValue $manifestPath
$manifest
