param(
    [switch]$SkipChecks,
    [switch]$KeepDist
)

$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

if (-not $KeepDist) {
    Remove-Item -LiteralPath "dist" -Recurse -Force -ErrorAction SilentlyContinue
}

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install build==1.3.0

if (-not $SkipChecks) {
    python -m ruff check .
    python -m ruff format --check .
    python -m unittest discover -s tests
    python -m compileall reefscape_rl scripts tests
}

python -m build

$commit = git rev-parse HEAD
$version = python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
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
