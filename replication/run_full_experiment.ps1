[CmdletBinding()]
param(
    [switch]$ConfirmOverwrite,

    [string]$Python,

    [ValidateSet("cpu", "cuda")]
    [string]$ChronosDevice = "cpu",

    [ValidateRange(1, 1024)]
    [int]$ChronosBatchSize = 8,

    [ValidateRange(1, 256)]
    [int]$PatchTSTThreads = 6
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ConfirmOverwrite) {
    throw "Pass -ConfirmOverwrite to acknowledge that generated files under results/ will be replaced."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found at '$Python'. Follow replication/README.md to create .venv."
}

function Invoke-CheckedPython {
    param([Parameter(Mandatory)][string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Push-Location $repoRoot
try {
    Write-Host "[1/9] Running automated tests"
    Invoke-CheckedPython -Arguments @("-m", "pytest", "-q")

    Write-Host "[2/9] Reproducing classical model-selection evidence"
    Invoke-CheckedPython -Arguments @(
        "reproduce_classical_selection.py",
        "--dataset",
        "all"
    )

    Write-Host "[3/9] Running the SPY classical forecasts"
    Invoke-CheckedPython -Arguments @("run_spy_classical.py")

    Write-Host "[4/9] Running the Airline and CISO classical forecasts"
    Invoke-CheckedPython -Arguments @("run_other_classical.py")

    Write-Host "[5/9] Running Chronos-2 zero-shot forecasts"
    Invoke-CheckedPython -Arguments @(
        "run_chronos_classical_comparison.py",
        "--device",
        $ChronosDevice,
        "--batch-size",
        $ChronosBatchSize.ToString()
    )

    Write-Host "[6/9] Running PatchTST supervised forecasts"
    Invoke-CheckedPython -Arguments @(
        "run_patchtst_comparison.py",
        "--dataset",
        "all",
        "--threads",
        $PatchTSTThreads.ToString()
    )

    Write-Host "[7/9] Reconciling regenerated forecasts with raw snapshots"
    Invoke-CheckedPython -Arguments @("audit_frozen_sources.py")

    Write-Host "[8/9] Regenerating final tables and figures"
    Invoke-CheckedPython -Arguments @("run_final_analysis.py")

    & git cat-file -e "e93958d^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Git commit e93958d is required to reproduce the SPY retry-effect claim."
    }

    Write-Host "[9/9] Reproducing the SPY retry-effect comparison"
    Invoke-CheckedPython -Arguments @("analyze_spy_retry_effect.py")

    Write-Host "Full experiment completed successfully. Review git diff before accepting regenerated artifacts."
}
finally {
    Pop-Location
}
