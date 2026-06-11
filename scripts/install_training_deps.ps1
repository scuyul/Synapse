param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment was not found at $VenvPath. Run setup first."
}

Write-Host "Upgrading pip"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}

Write-Host "Installing CUDA/training requirements. This can take a while."
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "requirements install failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Training dependencies installed."
