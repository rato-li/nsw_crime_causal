# Bootstrap the nsw-crime-causal environment (Windows PowerShell)
# Usage:  .\scripts\bootstrap.ps1            # GPU (RTX 5080, cu128)
#         .\scripts\bootstrap.ps1 -Cpu       # CPU-only torch
param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

if ($Cpu) {
    pip install torch --index-url https://download.pytorch.org/whl/cpu
} else {
    # RTX 5080 (Blackwell, sm_120) requires the cu128 wheels
    pip install torch --index-url https://download.pytorch.org/whl/cu128
}

pip install -e ".[all,dev]"

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example - review credentials."
}

Write-Host "`nDone. Try: nswcausal run-all --synthetic"
