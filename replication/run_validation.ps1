[CmdletBinding()]
param(
    [string]$Python,
    [switch]$IncludeModelSelection,
    [switch]$SkipRetryEffect
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
    Write-Host "[1/4] Running automated tests"
    Invoke-CheckedPython -Arguments @("-m", "pytest", "-q")

    Write-Host "[2/4] Reconciling frozen forecasts with committed raw snapshots"
    Invoke-CheckedPython -Arguments @("audit_frozen_sources.py")

    Write-Host "[3/4] Regenerating the frozen final analysis"
    Invoke-CheckedPython -Arguments @("run_final_analysis.py")

    if ($SkipRetryEffect) {
        Write-Warning "Skipping the SPY retry-effect audit at the user's request."
    }
    else {
        & git cat-file -e "e93958d^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Git commit e93958d is unavailable. Use a full clone to reproduce " +
                "the retry-effect claim, or pass -SkipRetryEffect to omit that one check."
            )
        }

        Write-Host "[4/4] Reproducing the SPY retry-effect comparison"
        Invoke-CheckedPython -Arguments @("analyze_spy_retry_effect.py")
    }

    if ($IncludeModelSelection) {
        Write-Host "Reproducing classical model-selection tables"
        Invoke-CheckedPython -Arguments @(
            "reproduce_classical_selection.py",
            "--dataset",
            "all"
        )
    }

    Write-Host "Frozen-result validation completed successfully."
}
finally {
    Pop-Location
}
