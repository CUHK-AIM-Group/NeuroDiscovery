param(
  [string]$CondaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
  [string]$CondaEnv = "neuroclaw",
  [string]$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path,
  [string]$RuntimeRoot = (Join-Path (Resolve-Path "$PSScriptRoot\..").Path "runtime"),
  [string]$Requirements = (Join-Path (Resolve-Path "$PSScriptRoot\..").Path "runtime-requirements.txt"),
  [switch]$SkipPython,
  [switch]$SkipBackend
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Block
  )
  Write-Host ""
  Write-Host "==> $Label" -ForegroundColor Cyan
  & $Block
}

function Assert-File {
  param([string]$Path, [string]$Message)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw $Message
  }
}

function Assert-Directory {
  param([string]$Path, [string]$Message)
  if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    throw $Message
  }
}

function Remove-DirectoryFresh {
  param([string]$Path)
  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

function Invoke-Robocopy {
  param(
    [string]$Source,
    [string]$Target,
    [string[]]$ExcludeDirs,
    [string[]]$ExcludeFiles
  )
  Remove-DirectoryFresh -Path $Target
  New-Item -ItemType Directory -Path $Target -Force | Out-Null
  & robocopy $Source $Target /E /NFL /NDL /NJH /NJS /NP /XD $ExcludeDirs /XF $ExcludeFiles
  $code = $LASTEXITCODE
  if ($code -gt 7) {
    throw "robocopy failed with exit code $code"
  }
}

function Copy-RootFileIfExists {
  param(
    [string]$SourceRoot,
    [string]$TargetRoot,
    [string]$Name
  )
  $source = Join-Path $SourceRoot $Name
  if (Test-Path -LiteralPath $source -PathType Leaf) {
    Copy-Item -LiteralPath $source -Destination (Join-Path $TargetRoot $Name) -Force
  }
}

function Get-CondaEnvPrefix {
  param([string]$CondaExe, [string]$CondaEnv)
  $prefix = (& $CondaExe run -n $CondaEnv python -c "import sys; print(sys.prefix)") | Select-Object -Last 1
  $prefix = "$prefix".Trim()
  if ($prefix -and (Test-Path -LiteralPath (Join-Path $prefix "python.exe") -PathType Leaf)) {
    return [System.IO.Path]::GetFullPath($prefix)
  }
  $condaRoot = Split-Path -Parent (Split-Path -Parent $CondaExe)
  $fallback = Join-Path (Join-Path $condaRoot "envs") $CondaEnv
  if (Test-Path -LiteralPath (Join-Path $fallback "python.exe") -PathType Leaf) {
    return [System.IO.Path]::GetFullPath($fallback)
  }
  throw "Unable to locate conda env prefix for '$CondaEnv'"
}

function Write-Utf8NoBom {
  param(
    [string]$Path,
    [string]$Content
  )
  $encoding = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$Requirements = [System.IO.Path]::GetFullPath($Requirements)
$PythonTarget = Join-Path $RuntimeRoot "python"
$BackendTarget = Join-Path $RuntimeRoot "backend"
$PythonExe = Join-Path $PythonTarget "python.exe"

Assert-Directory $RepoRoot "Repo root not found: $RepoRoot"
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null

if (-not $SkipPython) {
  Assert-File $CondaExe "Conda executable not found: $CondaExe"
  Assert-File $Requirements "Runtime requirements file not found: $Requirements"
  $CondaEnvPrefix = Get-CondaEnvPrefix -CondaExe $CondaExe -CondaEnv $CondaEnv

  Invoke-Step "Create standalone bundled Python from conda env '$CondaEnv'" {
    Remove-DirectoryFresh -Path $PythonTarget
    New-Item -ItemType Directory -Path $PythonTarget -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $PythonTarget "Lib") -Force | Out-Null

    Get-ChildItem -LiteralPath $CondaEnvPrefix -File |
      Where-Object {
        $_.Name -match '^(pythonw?\.exe|python\d+\.dll|python\d*\.dll|vcruntime.*\.dll|msvcp.*\.dll|ucrtbase\.dll|api-ms-.*\.dll|concrt.*\.dll|zlib\.dll|LICENSE.*\.txt)$'
      } |
      Copy-Item -Destination $PythonTarget -Force

    Invoke-Robocopy `
      -Source (Join-Path $CondaEnvPrefix "DLLs") `
      -Target (Join-Path $PythonTarget "DLLs") `
      -ExcludeDirs @("__pycache__") `
      -ExcludeFiles @("*.pyc", "*.pyo", "*.pdb")

    Invoke-Robocopy `
      -Source (Join-Path $CondaEnvPrefix "Lib") `
      -Target (Join-Path $PythonTarget "Lib") `
      -ExcludeDirs @("site-packages", "test", "tkinter", "idlelib", "turtledemo", "__pycache__") `
      -ExcludeFiles @("*.pyc", "*.pyo", "*.pdb")

    Invoke-Robocopy `
      -Source (Join-Path $CondaEnvPrefix "Library\bin") `
      -Target (Join-Path $PythonTarget "Library\bin") `
      -ExcludeDirs @("__pycache__") `
      -ExcludeFiles @("*.pyc", "*.pyo", "*.pdb")

    Assert-File $PythonExe "Bundled Python executable was not created: $PythonExe"
  }

  Invoke-Step "Install desktop runtime dependencies" {
    & $PythonExe -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw "Failed to bootstrap pip" }
    & $PythonExe -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip tooling" }
    & $PythonExe -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Failed to install runtime requirements" }
  }
}

if (-not $SkipBackend) {
  Invoke-Step "Stage NeuroClaw backend source" {
    Remove-DirectoryFresh -Path $BackendTarget
    New-Item -ItemType Directory -Path $BackendTarget -Force | Out-Null

    $ExcludeDirs = @(
      ".pytest_cache",
      ".mypy_cache",
      "__pycache__",
      "dist",
      "build",
      "node_modules",
      "data",
      "models",
      "papers",
      "runs",
      "logs",
      "output",
      "materials",
      ".venv",
      "venv"
    )
    $ExcludeFiles = @(
      "*.pyc",
      "*.pyo",
      "*.log",
      ".env"
    )

    foreach ($dirName in @("core", "skills", "neurooracle")) {
      $source = Join-Path $RepoRoot $dirName
      if (Test-Path -LiteralPath $source -PathType Container) {
        Invoke-Robocopy -Source $source -Target (Join-Path $BackendTarget $dirName) -ExcludeDirs $ExcludeDirs -ExcludeFiles $ExcludeFiles
      }
    }

    foreach ($fileName in @("LICENSE", "README.md", "README_zh.md", "SOUL.md", "pyproject.toml")) {
      Copy-RootFileIfExists -SourceRoot $RepoRoot -TargetRoot $BackendTarget -Name $fileName
    }

    $defaultEnvironment = [ordered]@{
      setup_type = "bundled"
      python_path = "bundled"
      conda_env = ""
      llm_backend = [ordered]@{
        provider = "openai"
        model = "gpt-5.5"
        base_url = "https://api.openai.com/v1"
        api_key_env = "OPENAI_API_KEY"
        available_models = @(
          [ordered]@{
            provider = "openai"
            model = "gpt-5.5"
            label = "OpenAI / gpt-5.5"
          }
        )
      }
      cuda = [ordered]@{
        device = "cpu"
      }
      toolchain = [ordered]@{}
      compression_mode = "stub"
    }
    $defaultEnvironment |
      ConvertTo-Json -Depth 8 |
      ForEach-Object {
        Write-Utf8NoBom -Path (Join-Path $BackendTarget "neuroclaw_environment.json") -Content $_
      }
  }
}

$Manifest = [ordered]@{
  created_at = (Get-Date).ToUniversalTime().ToString("o")
  conda_env = $CondaEnv
  repo_root = $RepoRoot
  runtime_root = $RuntimeRoot
  python_target = $PythonTarget
  backend_target = $BackendTarget
  requirements = $Requirements
  strategy = "standalone-python-prefix"
}
$Manifest |
  ConvertTo-Json -Depth 4 |
  ForEach-Object {
    Write-Utf8NoBom -Path (Join-Path $RuntimeRoot "runtime-manifest.json") -Content $_
  }

Write-Host ""
Write-Host "Bundled runtime staged at $RuntimeRoot" -ForegroundColor Green
Write-Host "Next: npm run dist:win" -ForegroundColor Green
