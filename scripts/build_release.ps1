param(
    [switch]$SkipChecks,
    [switch]$KeepDist,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

if (-not $PythonExe) {
    if ($env:VIRTUAL_ENV) {
        $venvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path -LiteralPath $venvPython) {
            $PythonExe = $venvPython
        }
    }
}

if (-not $PythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $PythonExe = $pythonCommand.Source
    }
}

if (-not $PythonExe) {
    throw "Could not find Python. Activate .venv or pass -PythonExe <path-to-python.exe>."
}

function Invoke-Step {
    param([string[]]$Command)

    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

if (-not $KeepDist) {
    Remove-Item -LiteralPath "dist" -Recurse -Force -ErrorAction SilentlyContinue
}

Invoke-Step @($PythonExe, "-m", "pip", "install", "--upgrade", "pip")
Invoke-Step @($PythonExe, "-m", "pip", "install", "-r", "requirements-dev.txt")
Invoke-Step @($PythonExe, "-m", "pip", "install", "build==1.3.0")

if (-not $SkipChecks) {
    Invoke-Step @($PythonExe, "-m", "ruff", "check", ".")
    Invoke-Step @($PythonExe, "-m", "ruff", "format", "--check", ".")
    Invoke-Step @($PythonExe, "-m", "unittest", "discover", "-s", "tests")
    Invoke-Step @($PythonExe, "-m", "compileall", "reefscape_rl", "scripts", "tests")
}

Invoke-Step @($PythonExe, "-m", "build")

if (-not (Test-Path -LiteralPath "dist")) {
    throw "Build completed without creating dist/."
}

$commit = git rev-parse HEAD
$version = & $PythonExe -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
$tag = @(git tag --points-at HEAD | Select-Object -First 1)
if ($null -eq $tag) {
    $tag = ""
}

$manifest = [ordered]@{
    project = "reefscape-rl"
    version = $version
    commit = $commit
    tag = $tag
    builtAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    artifacts = @(Get-ChildItem -Path "dist" -File | ForEach-Object {
        [ordered]@{
            name = $_.Name
            sizeBytes = $_.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        }
    })
}

$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "dist/release-manifest.json"

Write-Host ""
Write-Host "Release artifacts written to dist/:"
Get-ChildItem -Path "dist" -File | ForEach-Object {
    Write-Host ("- {0} ({1:n0} bytes)" -f $_.Name, $_.Length)
}
