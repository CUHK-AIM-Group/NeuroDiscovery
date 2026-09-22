[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [int]$Port = 18080,
    [string]$Model = "deepseek-v4-pro",
    [string]$BaseUrl = "http://127.0.0.1:18082/v1",
    [string]$BaselineRoot = "",
    [ValidateSet("minimal", "low", "medium", "high", "xhigh", "max")]
    [string]$ReasoningEffort = "high",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Stop-Case2BrainPilotTree {
    param([Parameter(Mandatory = $true)][int]$ProcessId)
    $children = @(
        Get-CimInstance Win32_Process |
            Where-Object { [int]$_.ParentProcessId -eq $ProcessId }
    )
    foreach ($child in $children) {
        Stop-Case2BrainPilotTree -ProcessId ([int]$child.ProcessId)
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

$neuroClawRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$baselineRoot = if ([string]::IsNullOrWhiteSpace($BaselineRoot)) {
    Join-Path (Split-Path $neuroClawRoot -Parent) "autoresearch_baselines"
}
else {
    (Resolve-Path -LiteralPath $BaselineRoot).Path
}
$brainPilotRoot = Join-Path $baselineRoot "BrainPilot"
$brainPilotCli = Join-Path $brainPilotRoot "packages\cli\dist\bin.js"
if (-not (Test-Path -LiteralPath $brainPilotCli -PathType Leaf)) {
    throw "BrainPilot CLI is missing: $brainPilotCli"
}

$resolvedDataDir = [System.IO.Path]::GetFullPath($DataDir)
New-Item -ItemType Directory -Path $resolvedDataDir -Force | Out-Null
$templateDir = Join-Path $resolvedDataDir "bp_template"
New-Item -ItemType Directory -Path $templateDir -Force | Out-Null

# This benchmark freezes the supplied evidence state.  No live MCP, local paper
# corpus, or retrieval skill is exposed to BrainPilot.
$mcpConfig = @{ mcpServers = @{} }
[System.IO.File]::WriteAllText(
    (Join-Path $templateDir "mcp_servers.json"),
    (($mcpConfig | ConvertTo-Json -Depth 4) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
$toolToggles = @{
    skill_search = $false
    get_domain_knowledge_local = $false
    search_papers_local = $false
}
[System.IO.File]::WriteAllText(
    (Join-Path $templateDir "tool_toggles.json"),
    (($toolToggles | ConvertTo-Json -Depth 3) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)

$modelConfig = Join-Path $resolvedDataDir "models.openai-responses.runtime.json"
$modelPayload = @{
    providers = @{
        "neuroclaw-local" = @{
            baseUrl = $BaseUrl.TrimEnd("/")
            api = "openai-responses"
            apiKey = '$ANTHROPIC_API_KEY'
            models = @(
                @{
                    id = $Model
                    reasoning = $true
                    input = @("text")
                    contextWindow = 200000
                    maxTokens = 8192
                }
            )
        }
    }
}
[System.IO.File]::WriteAllText(
    $modelConfig,
    (($modelPayload | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)

$runtimePort = $Port + 1
$listeners = @(
    foreach ($requiredPort in @($Port, $runtimePort)) {
        Get-NetTCPConnection -LocalPort $requiredPort -State Listen -ErrorAction SilentlyContinue
    }
)
if ($listeners.Count -gt 0) {
    if (-not $Restart) {
        throw "BrainPilot requires ports $Port and $runtimePort; at least one is already in use"
    }
    foreach ($owningProcessId in @($listeners.OwningProcess | Sort-Object -Unique)) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$owningProcessId"
        $commandLine = [string]$process.CommandLine
        if ($commandLine -notmatch "BrainPilot" -or $commandLine -notlike "*$resolvedDataDir*") {
            throw "Refusing to stop unrelated process $owningProcessId on a required port"
        }
        Stop-Case2BrainPilotTree -ProcessId ([int]$owningProcessId)
    }
    Start-Sleep -Seconds 2
}

# This is a non-secret loopback compatibility token.  Provider credentials are
# held only by the separately running DeepSeek gateway.
$env:ANTHROPIC_API_KEY = "neuroclaw-local-router"
$env:ANTHROPIC_MODEL = $Model
$env:BP_MODELS_JSON = $modelConfig
$env:BP_MODEL_PROVIDER = "neuroclaw-local"
$env:BP_THINKING_LEVEL = $ReasoningEffort

$stdoutPath = Join-Path $resolvedDataDir "server.stdout.log"
$stderrPath = Join-Path $resolvedDataDir "server.stderr.log"
$node = (Get-Command node).Source
$quotedDataDir = '"' + $resolvedDataDir.Replace('"', '\"') + '"'
$process = Start-Process `
    -FilePath $node `
    -ArgumentList @(
        $brainPilotCli,
        "up",
        "--dir", $quotedDataDir,
        "--port", [string]$Port,
        "--no-open",
        "--mode", "local"
    ) `
    -WorkingDirectory $brainPilotRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$healthUrl = "http://127.0.0.1:$Port/api/health"
$deadline = (Get-Date).AddSeconds(60)
$healthy = $false
do {
    Start-Sleep -Seconds 2
    try {
        $null = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
        $healthy = $true
    }
    catch {
        if ($process.HasExited) { break }
    }
} until ($healthy -or (Get-Date) -gt $deadline)

if (-not $healthy) {
    Stop-Case2BrainPilotTree -ProcessId $process.Id
    throw "BrainPilot failed its health check; inspect the service logs"
}

$manifest = [ordered]@{
    schema_version = "brainpilot.case2_formal_runtime.v1"
    model = $Model
    reasoning_effort = $ReasoningEffort
    wire_api = "openai-responses"
    provider = "neuroclaw-local"
    model_base_url = $BaseUrl.TrimEnd("/")
    native_retrieval_enabled = $false
    base_urls = @("http://127.0.0.1:$Port/api")
    runtime_port = $runtimePort
    data_dir = $resolvedDataDir
    baseline_repository_root = $baselineRoot
    backend_pid = $process.Id
    credentials_persisted = $false
    created_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
}
$manifestPath = Join-Path $resolvedDataDir "runtime_manifest.json"
[System.IO.File]::WriteAllText(
    $manifestPath,
    (($manifest | ConvertTo-Json -Depth 5) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
$manifest | Add-Member -NotePropertyName runtime_manifest_path -NotePropertyValue $manifestPath
$manifest
