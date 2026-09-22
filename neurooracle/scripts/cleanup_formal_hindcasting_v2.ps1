[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Execute')]
    [string]$Mode = 'Prepare',
    [string]$ManifestPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$HindcastingRoot = (Resolve-Path -LiteralPath (
    Join-Path $RepoRoot 'neurooracle\data\experiments\hindcasting'
)).Path
$ProtocolRoot = (Resolve-Path -LiteralPath (
    Join-Path $HindcastingRoot 'optimization_protocol_semantic_v4_20260814'
)).Path
$FrozenRoot = (Resolve-Path -LiteralPath (
    Join-Path $RepoRoot 'neurooracle\.frozen'
)).Path
$DesignPath = Join-Path $ProtocolRoot 'formal_hindcasting_design_v2_five_windows.json'
$LockPath = Join-Path $ProtocolRoot 'formal_hindcasting_design_v2_five_windows.lock.json'
$ExpectedDesignSha256 = '7FD12A2B3B309F90351C91FBB28C75219DCE367C056B8D137BCD61DACC15960D'

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $ProtocolRoot 'formal_hindcasting_v2_cleanup_provenance_20260825.json'
}
$ManifestPath = [IO.Path]::GetFullPath($ManifestPath)
$ManifestLockPath = [IO.Path]::ChangeExtension($ManifestPath, '.lock.json')

$ImmediateNames = @(
    'snapshots_full_v2',
    'snapshots_full_v2_endpoint_v3',
    'formal_static_endpoint_v7_20260813',
    'formal_static_endpoint_v8_20260814',
    'frozen_baselines_20260811_all_seed0_9',
    'snapshots_full_v2_endpoint_v2',
    'neurodiscovery_20260811_all_seed0_9',
    'frozen_baselines_20260811_all_seed0_9_eval',
    'kge_endpoint_v3',
    'frozen_baselines_20260811_all_seed0_9_eval_semantic_v2_testyears',
    'formal_dynamic_generalization_v4_20260814',
    'neurodiscovery_20260811_all_seed0_9_eval',
    'neurodiscovery_20260811_all_seed0_9_eval_semantic_v2',
    'optimization_protocol_semantic_v3_20260813',
    'formal_dynamic_generalization_v4_20260814_smoke',
    'hindcasting_v5_design_data_20260819',
    'optimization_protocol_20260812',
    'hindcasting_v5_design_preview_20260819',
    'formal_static_endpoint_v7_archive_v5_20260813_smoke',
    'formal_static_endpoint_v7_archive_v4_20260813_smoke',
    'formal_static_endpoint_v8_20260814_smoke',
    'formal_static_endpoint_v7_archive_v3_20260813_smoke',
    'formal_static_endpoint_v7_20260813_smoke'
)
$SupersededFrozenNames = @(
    'formal_dynamic_generalization_v2',
    'formal_static_endpoint_v7_v3',
    'formal_static_endpoint_v7_v4',
    'formal_static_endpoint_v7_v5'
)
$ProtocolDevelopmentNames = @(
    'dev_open_seed0_v1',
    'dev_closed_seed0_v1',
    'dev_closed_seed0_mut015_v1_part_a',
    'dev_closed_seed0_mut015_v1_part_b',
    'shared_index_cache_v4_open'
)
$ExcludedTopLevelNames = $ImmediateNames + @(
    'optimization_protocol_semantic_v4_20260814',
    'formal_dynamic_selected_mut015_v1_20260814',
    'formal_frozen_baselines_v1_20260814',
    'formal_static_supplement_v1_20260814',
    'formal_static_supplemental_v1_20260814'
)

function Assert-ImmediateChild {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$ExpectedParent
    )
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $parent = Split-Path -Parent $resolved
    if (-not [string]::Equals(
        $parent,
        $ExpectedParent,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing target outside expected parent: $resolved (expected $ExpectedParent)"
    }
    if ([string]::Equals(
        $resolved,
        $ExpectedParent,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to target an allowed root itself: $resolved"
    }
    return $resolved
}

function Get-TreeSha256 {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]]$Entries)
    $text = ($Entries | ForEach-Object {
        "$($_.path)`t$($_.bytes)`t$($_.sha256)"
    }) -join "`n"
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $content = [Text.Encoding]::UTF8.GetBytes($text)
        return ([BitConverter]::ToString($algorithm.ComputeHash($content))).Replace('-', '')
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-DirectoryInventory {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$ExpectedParent,
        [switch]$IncludeFileHashes
    )
    $resolved = Assert-ImmediateChild -Path $Path -ExpectedParent $ExpectedParent
    $files = @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force | Sort-Object FullName)
    $totalBytes = [int64]0
    foreach ($file in $files) {
        $totalBytes += [int64]$file.Length
    }
    if (-not $IncludeFileHashes) {
        return [pscustomobject][ordered]@{
            path = $resolved
            bytes = $totalBytes
            file_count = $files.Count
        }
    }

    $entries = @(
        foreach ($file in $files) {
            $relative = [IO.Path]::GetRelativePath($resolved, $file.FullName).Replace('\', '/')
            [pscustomobject][ordered]@{
                path = $relative
                bytes = [int64]$file.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash
            }
        }
    )
    return [pscustomobject][ordered]@{
        path = $resolved
        bytes = $totalBytes
        file_count = $files.Count
        tree_sha256 = Get-TreeSha256 -Entries $entries
        files = $entries
    }
}

function Assert-CategoryTotals {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [object[]]$Entries,
        [Parameter(Mandatory)] [int]$ExpectedDirectories,
        [Parameter(Mandatory)] [int64]$ExpectedBytes,
        [Parameter(Mandatory)] [int]$ExpectedFiles
    )
    $actualBytes = [int64](($Entries | Measure-Object -Property bytes -Sum).Sum)
    $actualFiles = [int](($Entries | Measure-Object -Property file_count -Sum).Sum)
    if (
        $Entries.Count -ne $ExpectedDirectories -or
        $actualBytes -ne $ExpectedBytes -or
        $actualFiles -ne $ExpectedFiles
    ) {
        throw (
            "$Name changed since audit: dirs=$($Entries.Count)/$ExpectedDirectories, " +
            "bytes=$actualBytes/$ExpectedBytes, files=$actualFiles/$ExpectedFiles"
        )
    }
}

function Resolve-DesignAsset {
    param(
        [Parameter(Mandatory)] [object]$Asset,
        [string]$DefaultBase = 'protocol_root'
    )
    $baseName = if ($Asset.PSObject.Properties.Name -contains 'path_base') {
        [string]$Asset.path_base
    }
    else {
        $DefaultBase
    }
    $base = switch ($baseName) {
        'protocol_root' { $ProtocolRoot }
        'repo_root' { $RepoRoot }
        default { throw "Unsupported design path base: $baseName" }
    }
    return [IO.Path]::GetFullPath((Join-Path $base ([string]$Asset.path)))
}

function Get-ProtectedPaths {
    $design = Get-Content -Raw -LiteralPath $DesignPath | ConvertFrom-Json
    $paths = [Collections.Generic.List[string]]::new()
    $paths.Add([IO.Path]::GetFullPath($DesignPath))
    $paths.Add([IO.Path]::GetFullPath($LockPath))
    $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.supersedes.design_path)))
    $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.selection_lock.path)))
    $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.selection_lock.protocol_path)))

    foreach ($name in @('knowledge_graph', 'extracted_claims')) {
        $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.canonical_release.$name.path)))
    }
    $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.dynamic_eligibility_lock.manifest_path)))
    $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.dynamic_eligibility_lock.matrix_path)))
    foreach ($asset in $design.historical_assets) {
        $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $asset.snapshot_manifest)))
        $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $asset.kge_checkpoint)))
    }
    foreach ($name in @(
        'neurodiscovery_part_x_manifest',
        'neurodiscovery_part_y_manifest',
        'baseline_generation_manifest',
        'baseline_completion_audit',
        'baseline_evaluation_manifest'
    )) {
        $paths.Add([IO.Path]::GetFullPath((Join-Path $ProtocolRoot $design.reusable_v1_results.$name.path)))
    }
    foreach ($name in @('neurodiscovery', 'baselines')) {
        $asset = $design.source_bundles.$name
        $manifest = Resolve-DesignAsset -Asset $asset
        $paths.Add($manifest)
        $bundle = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
        $bundleRoot = Split-Path -Parent $manifest
        foreach ($file in $bundle.files) {
            $paths.Add([IO.Path]::GetFullPath((Join-Path $bundleRoot $file.bundle_path)))
        }
        foreach ($reference in $bundle.references) {
            $paths.Add([IO.Path]::GetFullPath([string]$reference.path))
        }
    }
    return @($paths | Sort-Object -Unique)
}

function Assert-NoProtectedDescendant {
    param(
        [Parameter(Mandatory)] [object[]]$Targets,
        [Parameter(Mandatory)] [string[]]$ProtectedPaths
    )
    foreach ($target in $Targets) {
        $prefix = ([string]$target.path).TrimEnd('\') + '\'
        foreach ($protected in $ProtectedPaths) {
            if (
                [string]::Equals($protected, $target.path, [StringComparison]::OrdinalIgnoreCase) -or
                $protected.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
            ) {
                throw "Deletion target contains a locked asset: $($target.path) -> $protected"
            }
        }
    }
}

function Assert-CurrentDesignHash {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $DesignPath).Hash
    if ($actual -ne $ExpectedDesignSha256) {
        throw "Design SHA256 changed: $actual"
    }
    $lock = Get-Content -Raw -LiteralPath $LockPath | ConvertFrom-Json
    if ($lock.status -ne 'locked' -or $lock.design_sha256 -ne $ExpectedDesignSha256) {
        throw 'Design lock is missing, inactive, or points to a different SHA256.'
    }
}

function New-PreparationManifest {
    Assert-CurrentDesignHash
    if (-not [string]::Equals(
        (Split-Path -Parent $ManifestPath),
        $ProtocolRoot,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Cleanup manifest must be written directly under the locked protocol root: $ManifestPath"
    }

    $immediate = @(
        foreach ($name in $ImmediateNames) {
            Get-DirectoryInventory -Path (Join-Path $HindcastingRoot $name) -ExpectedParent $HindcastingRoot
        }
    )
    Assert-CategoryTotals -Name 'immediate' -Entries $immediate `
        -ExpectedDirectories 23 -ExpectedBytes 47459598353 -ExpectedFiles 34206

    $supersededFrozen = @(
        foreach ($name in $SupersededFrozenNames) {
            Get-DirectoryInventory -Path (Join-Path $FrozenRoot $name) -ExpectedParent $FrozenRoot
        }
    )
    Assert-CategoryTotals -Name 'superseded_frozen' -Entries $supersededFrozen `
        -ExpectedDirectories 4 -ExpectedBytes 27759483 -ExpectedFiles 576

    $developmentDirectories = @(
        Get-ChildItem -LiteralPath $HindcastingRoot -Directory -Force |
            Where-Object {
                $_.Name -notin $ExcludedTopLevelNames -and
                $_.Name -match '(?i)(smoke|dev|screen|ablation|matrix|summary|cache|diagnostic|repro)'
            } |
            Sort-Object Name
    )
    $developmentStats = @(
        foreach ($directory in $developmentDirectories) {
            Get-DirectoryInventory -Path $directory.FullName -ExpectedParent $HindcastingRoot
        }
    )
    Assert-CategoryTotals -Name 'development_and_smoke' -Entries $developmentStats `
        -ExpectedDirectories 134 -ExpectedBytes 3819841884 -ExpectedFiles 5200

    $protocolDevelopmentStats = @(
        foreach ($name in $ProtocolDevelopmentNames) {
            Get-DirectoryInventory -Path (Join-Path $ProtocolRoot $name) -ExpectedParent $ProtocolRoot
        }
    )
    Assert-CategoryTotals -Name 'protocol_development' -Entries $protocolDevelopmentStats `
        -ExpectedDirectories 5 -ExpectedBytes 433918571 -ExpectedFiles 273

    $allTargets = @($immediate + $supersededFrozen + $developmentStats + $protocolDevelopmentStats)
    Assert-NoProtectedDescendant -Targets $allTargets -ProtectedPaths (Get-ProtectedPaths)

    Write-Host 'Hashing development/smoke provenance (this reads about 4 GiB)...'
    $development = @(
        foreach ($directory in $developmentDirectories) {
            Get-DirectoryInventory -Path $directory.FullName -ExpectedParent $HindcastingRoot -IncludeFileHashes
        }
    )
    $protocolDevelopment = @(
        foreach ($name in $ProtocolDevelopmentNames) {
            Get-DirectoryInventory -Path (Join-Path $ProtocolRoot $name) -ExpectedParent $ProtocolRoot -IncludeFileHashes
        }
    )

    $manifest = [pscustomobject][ordered]@{
        schema_version = 'neurodiscovery-hindcasting-cleanup-provenance.v1'
        prepared_at = (Get-Date).ToString('o')
        execution_authorized_by_user = $true
        design_path = $DesignPath
        design_sha256 = $ExpectedDesignSha256
        deletion_scope = 'A-level superseded outputs, superseded frozen bundles, and B-level development/smoke outputs after per-file provenance capture'
        irreversible_delete = $true
        categories = [pscustomobject][ordered]@{
            immediate = $immediate
            superseded_frozen = $supersededFrozen
            development_and_smoke = $development
            protocol_development = $protocolDevelopment
        }
        totals = [pscustomobject][ordered]@{
            directories = 166
            bytes = [int64]51741118291
            files = 40255
        }
        explicitly_retained = @(
            (Join-Path $HindcastingRoot 'formal_dynamic_selected_mut015_v1_20260814\part_x'),
            (Join-Path $HindcastingRoot 'formal_dynamic_selected_mut015_v1_20260814\part_y'),
            (Join-Path $HindcastingRoot 'formal_frozen_baselines_v1_20260814'),
            (Join-Path $HindcastingRoot 'formal_static_supplemental_v1_20260814'),
            (Join-Path $FrozenRoot 'formal_static_endpoint_v8'),
            (Join-Path $ProtocolRoot 'shared_index_cache_v4')
        )
    }
    $json = $manifest | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($ManifestPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    $manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ManifestPath).Hash
    [pscustomobject][ordered]@{
        status = 'prepared'
        manifest_path = $ManifestPath
        manifest_sha256 = $manifestSha256
        directories = $manifest.totals.directories
        bytes = $manifest.totals.bytes
        files = $manifest.totals.files
    } | ConvertTo-Json
}

function Invoke-ManifestDeletion {
    Assert-CurrentDesignHash
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Missing prepared cleanup manifest: $ManifestPath"
    }
    $manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ManifestPath).Hash
    if (-not (Test-Path -LiteralPath $ManifestLockPath -PathType Leaf)) {
        throw "Missing cleanup-manifest lock: $ManifestLockPath"
    }
    $manifestLock = Get-Content -Raw -LiteralPath $ManifestLockPath | ConvertFrom-Json
    if (
        $manifestLock.status -ne 'locked_before_deletion' -or
        $manifestLock.manifest_sha256 -ne $manifestSha256 -or
        $manifestLock.design_sha256 -ne $ExpectedDesignSha256
    ) {
        throw 'Cleanup manifest does not match its pre-deletion lock.'
    }
    $manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
    if (
        $manifest.schema_version -ne 'neurodiscovery-hindcasting-cleanup-provenance.v1' -or
        $manifest.design_sha256 -ne $ExpectedDesignSha256 -or
        -not $manifest.execution_authorized_by_user
    ) {
        throw 'Cleanup manifest is not valid for this locked design and authorization.'
    }

    $categoryParents = [ordered]@{
        immediate = $HindcastingRoot
        superseded_frozen = $FrozenRoot
        development_and_smoke = $HindcastingRoot
        protocol_development = $ProtocolRoot
    }
    $verifiedTargets = [Collections.Generic.List[object]]::new()
    foreach ($categoryName in $categoryParents.Keys) {
        foreach ($entry in $manifest.categories.$categoryName) {
            $inventory = Get-DirectoryInventory `
                -Path ([string]$entry.path) `
                -ExpectedParent $categoryParents[$categoryName] `
                -IncludeFileHashes:($categoryName -in @('development_and_smoke', 'protocol_development'))
            if (
                $inventory.bytes -ne [int64]$entry.bytes -or
                $inventory.file_count -ne [int]$entry.file_count
            ) {
                throw "Target changed after preparation: $($entry.path)"
            }
            if (
                $categoryName -in @('development_and_smoke', 'protocol_development') -and
                $inventory.tree_sha256 -ne [string]$entry.tree_sha256
            ) {
                throw "Provenance tree SHA256 changed after preparation: $($entry.path)"
            }
            $verifiedTargets.Add([pscustomobject][ordered]@{
                category = $categoryName
                path = $inventory.path
                bytes = [int64]$inventory.bytes
                file_count = [int]$inventory.file_count
            })
        }
    }

    $actualBytes = [int64](($verifiedTargets | Measure-Object -Property bytes -Sum).Sum)
    $actualFiles = [int](($verifiedTargets | Measure-Object -Property file_count -Sum).Sum)
    if (
        $verifiedTargets.Count -ne [int]$manifest.totals.directories -or
        $actualBytes -ne [int64]$manifest.totals.bytes -or
        $actualFiles -ne [int]$manifest.totals.files
    ) {
        throw 'Prepared target totals no longer match the cleanup manifest.'
    }
    Assert-NoProtectedDescendant -Targets @($verifiedTargets) -ProtectedPaths (Get-ProtectedPaths)

    $freeBefore = [int64](Get-PSDrive -Name C).Free
    foreach ($target in $verifiedTargets) {
        $parent = $categoryParents[$target.category]
        $resolved = Assert-ImmediateChild -Path $target.path -ExpectedParent $parent
        Write-Host "Deleting $resolved"
        Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $resolved) {
            throw "Target still exists after deletion: $resolved"
        }
    }
    $freeAfter = [int64](Get-PSDrive -Name C).Free

    $receiptPath = Join-Path $ProtocolRoot 'formal_hindcasting_v2_cleanup_receipt_20260825.json'
    $receipt = [pscustomobject][ordered]@{
        schema_version = 'neurodiscovery-hindcasting-cleanup-receipt.v1'
        completed_at = (Get-Date).ToString('o')
        manifest_path = $ManifestPath
        manifest_sha256 = $manifestSha256
        design_sha256 = $ExpectedDesignSha256
        deleted_directories = $verifiedTargets.Count
        deleted_files = $actualFiles
        deleted_logical_bytes = $actualBytes
        free_bytes_before = $freeBefore
        free_bytes_after = $freeAfter
        free_bytes_change = [int64]($freeAfter - $freeBefore)
        retained_conditional_v1_results = $true
    }
    $receiptJson = $receipt | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText($receiptPath, $receiptJson + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    [pscustomobject][ordered]@{
        status = 'deleted'
        receipt_path = $receiptPath
        manifest_sha256 = $manifestSha256
        deleted_directories = $verifiedTargets.Count
        deleted_files = $actualFiles
        deleted_logical_bytes = $actualBytes
        free_bytes_change = [int64]($freeAfter - $freeBefore)
    } | ConvertTo-Json
}

switch ($Mode) {
    'Prepare' { New-PreparationManifest }
    'Execute' { Invoke-ManifestDeletion }
}
