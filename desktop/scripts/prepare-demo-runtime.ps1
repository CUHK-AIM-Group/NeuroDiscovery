param()
$ErrorActionPreference = "Stop"
$desktopRoot = (Resolve-Path "$PSScriptRoot\..").Path
$pythonSource = Join-Path $desktopRoot "runtime\python"
$runtimeTarget = Join-Path $desktopRoot "runtime-demo"
if (-not (Test-Path -LiteralPath (Join-Path $pythonSource "python.exe"))) {
  throw "Prepare the standard bundled Python runtime first: npm run prepare:runtime:win"
}
if (Test-Path -LiteralPath $runtimeTarget) {
  throw "Demo staging already exists; preserve it and choose a new staging directory before rebuilding."
}
New-Item -ItemType Directory -Path $runtimeTarget | Out-Null
& robocopy $pythonSource (Join-Path $runtimeTarget "python") /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc *.pyo
if ($LASTEXITCODE -gt 7) { throw "Bundled Python copy failed: $LASTEXITCODE" }
& "$PSScriptRoot\prepare-bundled-runtime.ps1" -RuntimeRoot $runtimeTarget -SkipPython -Demo
if (-not $?) { throw "Demo backend staging failed" }
exit 0
