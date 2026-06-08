param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment at $VenvPath"
& $Python -m venv $VenvPath

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not created at $venvPython"
}

Write-Host "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Host "Installing CUDA runtime requirements"
& $venvPython -m pip install -r requirements.txt

Write-Host "Installing package in editable mode"
& $venvPython -m pip install -e .

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Verify CUDA with: .\.venv\Scripts\python.exe -m reefscape_rl.doctor"
