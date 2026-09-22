[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$Python = "C:\Users\45846\Documents\Code\NeuroClaw\.venv-case2-benchmark\Scripts\python.exe",
    [string]$KeysPath = "C:\Users\45846\Downloads\keys.txt",
    [int]$Port = 18082,
    [string]$HostAddress = "127.0.0.1",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($HostAddress -notin @("127.0.0.1", "localhost", "::1")) {
    throw "The credential-bearing gateway may bind only to loopback"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python runtime is missing: $Python"
}
if (-not (Test-Path -LiteralPath $KeysPath -PathType Leaf)) {
    throw "Keys file is missing"
}

$resolvedDataDir = [System.IO.Path]::GetFullPath($DataDir)
New-Item -ItemType Directory -Path $resolvedDataDir -Force | Out-Null
$checkpointDir = Join-Path $resolvedDataDir "checkpoints"
New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    if (-not $Restart) {
        throw "Port $Port is already in use"
    }
    foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
        $commandLine = [string]$process.CommandLine
        if ($commandLine -notmatch "case2_formal_deepseek_gateway" -or $commandLine -notlike "*$resolvedDataDir*") {
            throw "Refusing to stop unrelated process $($listener.OwningProcess) on port $Port"
        }
        Stop-Process -Id ([int]$listener.OwningProcess) -Force
    }
    Start-Sleep -Seconds 1
}

$stdoutPath = Join-Path $resolvedDataDir "gateway.stdout.log"
$stderrPath = Join-Path $resolvedDataDir "gateway.stderr.log"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$quotedKeysPath = '"' + $KeysPath.Replace('"', '\"') + '"'
$quotedCheckpointDir = '"' + $checkpointDir.Replace('"', '\"') + '"'
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "-m", "neurooracle.scripts.case2_formal_deepseek_gateway",
        "--host", $HostAddress,
        "--port", [string]$Port,
        "--keys", $quotedKeysPath,
        "--checkpoint-dir", $quotedCheckpointDir
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
    catch {
        if ($process.HasExited) { break }
    }
} until ($null -ne $health -or (Get-Date) -gt $deadline)

$healthy = (
    $null -ne $health -and
    $health.status -eq "ok" -and
    $health.benchmark_id -eq "case2_adni_formal_benchmark_v1" -and
    $health.model -eq "deepseek-v4-pro" -and
    $health.forced_reasoning_effort -eq "high" -and
    [double]$health.forced_temperature -eq 0.0 -and
    [int]$health.forced_max_output_tokens -eq 8192 -and
    [int]$health.max_concurrent_upstream_requests -eq 1 -and
    -not [bool]$health.official_deepseek_enabled
)
if (-not $healthy) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "Case 2 DeepSeek gateway failed its fixed-setting health check"
}

$manifest = [ordered]@{
    schema_version = "neuroclaw.case2-formal-deepseek-v4-pro-gateway-runtime.v1"
    benchmark_id = "case2_adni_formal_benchmark_v1"
    model = "deepseek-v4-pro"
    base_url = "http://$HostAddress`:$Port/v1"
    health_url = $healthUrl
    checkpoint_dir = $checkpointDir
    pid = $process.Id
    forced_reasoning_effort = "high"
    forced_temperature = 0.0
    forced_max_output_tokens = 8192
    max_concurrent_upstream_requests = 1
    automatic_route = @(
        "opencode_go_key_1",
        "opencode_go_key_2",
        "opencode_go_key_3",
        "ollama_cloud",
        "opencode_go_final_pass_once"
    )
    transient_retry_seconds = @(2, 8, 30)
    official_deepseek_automatic_fallback = $false
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
