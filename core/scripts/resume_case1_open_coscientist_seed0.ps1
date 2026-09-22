[CmdletBinding()]
param(
    [string]$ExperimentRoot = "\\192.168.3.61\data\Public Dataset\case1_autoresearch_comparison\20260826_kg2c0273_deepseekv4pro_true_closed_loop_3seeds_v1",
    [ValidatePattern("^[A-Z]$")]
    [string]$DriveLetter = "R",
    [int]$Port = 18081,
    [string]$KeysPath = "C:\Users\45846\Downloads\keys.txt",
    [string]$GatewayPython = "C:\Users\45846\Documents\Code\NeuroClaw\.venv-case2-benchmark\Scripts\python.exe",
    [string]$OpenCoPython = "C:\Users\45846\Documents\Code\autoresearch_baselines\open-coscientist\.venv\Scripts\python.exe",
    [string]$OpenCoRoot = "C:\Users\45846\Documents\Code\autoresearch_baselines\open-coscientist",
    [ValidateSet("automatic", "ollama_only")]
    [string]$RouteMode = "automatic",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ExpectedCacheIdentity = "daa5b0cb665ff4e1a3571a765d094b92abdfed5ce06eb5eacf07ff805407b3a4"
$ExpectedKgSha256 = "2C02732582DA9907C68300D791C3560B8E66CC268D987D453E7C971BD6CDEFF5"
$ExpectedClaimsSha256 = "705B079989FDEC3D7756CF737F12B41756EB8058F8ED66A009D6F832489F5BA4"
$ExpectedStateSha256 = "744B75718B2BEEBFDAF9055595CB2161ED841643ACD58F365F55717BC7B5360E"
$ExpectedRegistrySha256 = "5e7e422fc767e088fad46501048c60f2aa043bb6aa00904d3f2dc1e21ccd6502"
$TrialRelativePath = "baseline_generation\formal_official\open_coscientist\trial_00"
$LocalPath = "$DriveLetter`:"

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
}

function Stop-VerifiedGateway([int]$GatewayPort) {
    $listeners = @(Get-NetTCPConnection -LocalPort $GatewayPort -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
        $commandLine = [string]$process.CommandLine
        if ($commandLine -notmatch "deepseek_v4_pro_gateway") {
            throw "Refusing to stop unrelated process $($listener.OwningProcess) on port $GatewayPort"
        }
        Stop-Process -Id ([int]$listener.OwningProcess) -Force
    }
}

$ResolvedExperimentRoot = [System.IO.Path]::GetFullPath($ExperimentRoot).TrimEnd("\")
Require-File $KeysPath "Keys file"
Require-File $GatewayPython "Gateway Python"
Require-File $OpenCoPython "Open Co-Scientist Python"
Require-File (Join-Path $OpenCoRoot "src\open_coscientist\cache.py") "Open Co-Scientist source"

$mapping = Get-SmbMapping -LocalPath $LocalPath -ErrorAction SilentlyContinue
if ($null -ne $mapping) {
    $observedRemote = ([string]$mapping.RemotePath).TrimEnd("\")
    if (-not $observedRemote.Equals($ResolvedExperimentRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$LocalPath maps to another target: $observedRemote"
    }
}
else {
    if (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$LocalPath'" -ErrorAction SilentlyContinue) {
        throw "$LocalPath is already in use and is not the required SMB mapping"
    }
    New-SmbMapping -LocalPath $LocalPath -RemotePath $ResolvedExperimentRoot -Persistent $false | Out-Null
}

$MappedRoot = "$LocalPath\"
$UncTrial = Join-Path $ResolvedExperimentRoot $TrialRelativePath
$MappedTrial = Join-Path $MappedRoot $TrialRelativePath
$UncTask = Join-Path $UncTrial "task.json"
$MappedTask = Join-Path $MappedTrial "task.json"
$CacheManifestPath = Join-Path $MappedTrial "open_coscientist_cache_manifest.json"
$CacheDir = Join-Path $MappedTrial "open_coscientist_trial_cache"
$ReadinessPath = Join-Path $MappedRoot "runtime\formal_baseline_readiness_audit.json"

Require-File $UncTask "Preserved UNC task"
Require-File $MappedTask "Mapped task"
Require-File $CacheManifestPath "Open Co-Scientist cache manifest"
Require-File $ReadinessPath "Formal readiness audit"

$uncTaskSha = (Get-FileHash -LiteralPath $UncTask -Algorithm SHA256).Hash
$mappedTaskSha = (Get-FileHash -LiteralPath $MappedTask -Algorithm SHA256).Hash
if ($uncTaskSha -ne $mappedTaskSha) {
    throw "Mapped task differs from the preserved UNC task"
}

$cacheManifest = Get-Content -LiteralPath $CacheManifestPath -Raw | ConvertFrom-Json
if ([string]$cacheManifest.identity_sha256 -ne $ExpectedCacheIdentity) {
    throw "Open Co-Scientist cache identity changed"
}
$sealed = $cacheManifest.identity.sealed_inputs
$release = $sealed.canonical_kg_release
if ([string]$release.knowledge_graph_sha256 -ne $ExpectedKgSha256) {
    throw "Cache manifest pins another KG"
}
if ([string]$release.extracted_claims_sha256 -ne $ExpectedClaimsSha256) {
    throw "Cache manifest pins another claims file"
}
if ([string]$release.current_state_sha256 -ne $ExpectedStateSha256) {
    throw "Cache manifest pins another canonical state"
}
if ([string]$sealed.public_registry_sha256 -ne $ExpectedRegistrySha256) {
    throw "Cache manifest pins another public registry"
}

$readiness = Get-Content -LiteralPath $ReadinessPath -Raw | ConvertFrom-Json
if (-not [bool]$readiness.offline_ready) {
    throw "Formal readiness audit is not offline-ready"
}
if (-not [bool]$readiness.formal_progress.open_coscientist_seed0_cache.binding_ok) {
    throw "Formal readiness audit rejects the Open Co-Scientist cache binding"
}

$shortCacheJsonLength = $CacheDir.Length + 1 + 64 + 5
if ($shortCacheJsonLength -ge 260) {
    throw "Mapped cache path is still too long: $shortCacheJsonLength"
}

$checkpointDir = Join-Path $MappedRoot "runtime\deepseek_v4_pro_gateway\checkpoints"
$preflightOutput = & $GatewayPython -m neurooracle.scripts.deepseek_v4_pro_gateway `
    --host 127.0.0.1 `
    --port $Port `
    --keys $KeysPath `
    --checkpoint-dir $checkpointDir `
    --router-pause-status 422 `
    --route-mode $RouteMode `
    --preflight-only
if ($LASTEXITCODE -ne 0) {
    throw "DeepSeek V4 Pro gateway preflight failed"
}
$gatewayPreflight = ($preflightOutput | Out-String) | ConvertFrom-Json
if ([string]$gatewayPreflight.route_mode -ne $RouteMode) {
    throw "Gateway preflight returned another route mode"
}
$routeLabels = @($gatewayPreflight.automatic_route)
if ($RouteMode -eq "ollama_only") {
    if ($routeLabels.Count -lt 1 -or @($routeLabels | Where-Object { $_ -notlike "ollama_cloud_key_*" }).Count -gt 0) {
        throw "Ollama-only route contains a non-Ollama channel"
    }
}
elseif ([int]$gatewayPreflight.route_count -lt 7) {
    throw "Automatic finite route is incomplete"
}
if ([bool]$gatewayPreflight.official_deepseek_enabled) {
    throw "Official DeepSeek automatic fallback must remain disabled"
}
if ([int]$gatewayPreflight.router_pause_http_status -ne 422) {
    throw "Case 1 resume must use a non-retryable router pause status"
}

$cacheFiles = @(Get-ChildItem -LiteralPath $CacheDir -File -Filter "*.json")
$report = [ordered]@{
    schema_version = "case1.open-coscientist-seed0-resume-preflight.v1"
    experiment_root = $ResolvedExperimentRoot
    mapped_root = $MappedRoot
    task_sha256 = $mappedTaskSha
    cache_identity_sha256 = [string]$cacheManifest.identity_sha256
    cache_status = [string]$cacheManifest.status
    cache_open_count = [int]$cacheManifest.open_count
    cache_files = $cacheFiles.Count
    cache_bytes = ($cacheFiles | Measure-Object Length -Sum).Sum
    short_cache_json_path_length = $shortCacheJsonLength
    canonical_kg_sha256 = [string]$release.knowledge_graph_sha256
    public_registry_sha256 = [string]$sealed.public_registry_sha256
    offline_ready = [bool]$readiness.offline_ready
    formal_scoreable_method_seeds = [int]$readiness.formal_progress.scoreable_method_seeds
    gateway_route_count = [int]$gatewayPreflight.route_count
    gateway_route_mode = [string]$gatewayPreflight.route_mode
    gateway_route = $routeLabels
    router_pause_http_status = [int]$gatewayPreflight.router_pause_http_status
    official_deepseek_automatic_fallback = [bool]$gatewayPreflight.official_deepseek_enabled
    secrets_printed_or_persisted = $false
    preflight_only = [bool]$PreflightOnly
}

if ($PreflightOnly) {
    $report | ConvertTo-Json -Depth 6
    exit 0
}

$startScript = Join-Path $PSScriptRoot "start_deepseek_v4_pro_gateway.ps1"
$gatewayDataDir = Join-Path $MappedRoot "runtime\deepseek_v4_pro_gateway"
$gatewayManifest = & $startScript `
    -DataDir $gatewayDataDir `
    -Port $Port `
    -Python $GatewayPython `
    -KeysPath $KeysPath `
    -RouterPauseStatus 422 `
    -RouteMode $RouteMode

$adapterExit = 1
$managedEnvironment = @(
    "CS1_LOCAL_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "PYTHONUTF8",
    "PYTHONIOENCODING"
)
$priorEnvironment = @{}
foreach ($name in $managedEnvironment) {
    $priorEnvironment[$name] = if (Test-Path -LiteralPath "Env:$name") {
        [System.Environment]::GetEnvironmentVariable($name, "Process")
    }
    else {
        $null
    }
}
try {
    $env:CS1_LOCAL_API_KEY = "neuroclaw-local-router"
    $env:OPENAI_API_KEY = "neuroclaw-local-router"
    $env:OPENAI_BASE_URL = "http://127.0.0.1:$Port/v1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    Push-Location $OpenCoRoot
    try {
        & $OpenCoPython `
            (Join-Path $PSScriptRoot "case_study_official_adapter_client.py") `
            --method open_coscientist `
            --task $MappedTask `
            --out $MappedTrial `
            --repo $OpenCoRoot `
            --model deepseek-v4-pro `
            --base-url "http://127.0.0.1:$Port/v1" `
            --reasoning-effort high `
            --enable-native-retrieval
        $adapterExit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
finally {
    foreach ($name in $managedEnvironment) {
        if ($null -eq $priorEnvironment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            Set-Item -LiteralPath "Env:$name" -Value $priorEnvironment[$name]
        }
    }
    Stop-VerifiedGateway $Port
}

if ($adapterExit -ne 0) {
    throw "Open Co-Scientist seed 0 adapter exited with code $adapterExit"
}

$report["preflight_only"] = $false
$report["adapter_exit_code"] = $adapterExit
$report["native_result_exists"] = Test-Path -LiteralPath (Join-Path $MappedTrial "native_result.json")
$report["search_policy_exists"] = Test-Path -LiteralPath (Join-Path $MappedTrial "search_policy.json")
$report["gateway_stopped"] = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count -eq 0
$report | ConvertTo-Json -Depth 6
