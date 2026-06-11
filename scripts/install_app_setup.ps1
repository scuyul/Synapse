param(
    [string]$Python = "",
    [string]$VenvPath = ".venv",
    [switch]$InstallTrainingDependencies
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
    $totalSteps = if ($InstallTrainingDependencies) { 3 } else { 2 }
    $step = 1

    Write-Progress `
        -Activity "Reefscape RL setup" `
        -Status "Creating Python virtual environment" `
        -PercentComplete (($step - 1) * 100 / $totalSteps)
    & (Join-Path $PSScriptRoot "setup_venv.ps1") `
        -Python $Python `
        -VenvPath $VenvPath `
        -SkipRequirements `
        -FastAppInstall
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment setup failed with exit code $LASTEXITCODE."
    }

    $step += 1
    if ($InstallTrainingDependencies) {
        Write-Progress `
            -Activity "Reefscape RL setup" `
            -Status "Installing training dependencies" `
            -PercentComplete (($step - 1) * 100 / $totalSteps)
        & (Join-Path $PSScriptRoot "install_training_deps.ps1") -VenvPath $VenvPath
        if ($LASTEXITCODE -ne 0) {
            throw "Training dependency install failed with exit code $LASTEXITCODE."
        }
        $step += 1
    }

    Write-Progress `
        -Activity "Reefscape RL setup" `
        -Status "Setup complete" `
        -PercentComplete (($step - 1) * 100 / $totalSteps)
    Write-Host ""
    Write-Host "Reefscape RL setup complete."
}
finally {
    Write-Progress -Activity "Reefscape RL setup" -Completed
    Pop-Location
}
