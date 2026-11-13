$ErrorActionPreference = 'Stop'
$taskBase = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$archivePath = [System.IO.Path]::GetFullPath((Join-Path $taskBase '_superseded\cp4_frequency_20261023'))
if (-not $archivePath.StartsWith($taskBase + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside intended OP12 directory' }
$names = @('cp4_inchain_frequency_retest_20261023', 'cp4_cold_l2_lock1845_20261023', 'cp4_inchain_frequency_retest_20261023_transport.tar.gz', 'cp4_cold_l2_lock1845_20261023_transport.tar.gz')
foreach ($name in $names) {
    $src = [System.IO.Path]::GetFullPath((Join-Path $taskBase $name))
    $dst = [System.IO.Path]::GetFullPath((Join-Path $archivePath $name))
    if ((Split-Path -Parent $src) -ne $taskBase) { throw 'Unexpected source parent' }
    if (-not $dst.StartsWith($archivePath + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected archive target' }
    if (Test-Path -LiteralPath $dst) { throw ('Archive target exists: ' + $dst) }
    if (-not (Test-Path -LiteralPath $src)) { throw ('Source missing: ' + $src) }
}
New-Item -ItemType Directory -Path $archivePath -Force | Out-Null
foreach ($name in $names) {
    $src = [System.IO.Path]::GetFullPath((Join-Path $taskBase $name))
    Move-Item -LiteralPath $src -Destination $archivePath
    Write-Output ('Archived: ' + $name)
}
