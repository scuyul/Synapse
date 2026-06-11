param(
    [string]$Output = "dist\reefscape-app.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

$goCommand = Get-Command go -ErrorAction SilentlyContinue
if (-not $goCommand) {
    throw "Go was not found on PATH. Install Go or build the Python tools directly."
}

$outputPath = Join-Path $repoRoot $Output
$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Push-Location (Join-Path $repoRoot "app")
try {
    go build -trimpath -o $outputPath .
    if ($LASTEXITCODE -ne 0) {
        throw "go build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Built app launcher: $outputPath"
