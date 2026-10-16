<# Dry-run by default. -Apply removes allowlisted repository-local caches only. #>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$IncludeOwnedExternalCache,
    [string]$OwnedExternalCache,
    [string]$Repository = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = 'python',
    [string]$Report
)
$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'cache_cleanup.py'), '--repository', $Repository)
if ($Apply) { $arguments += '--apply' }
if ($IncludeOwnedExternalCache) { $arguments += '--include-owned-external-cache' }
if ($OwnedExternalCache) { $arguments += @('--owned-external-cache', $OwnedExternalCache) }
if ($Report) { $arguments += @('--report', $Report) }
& $Python @arguments
exit $LASTEXITCODE
