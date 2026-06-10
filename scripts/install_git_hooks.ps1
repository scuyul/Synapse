$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

git config core.hooksPath .githooks

Write-Host "Installed repo-local Git hooks from .githooks/"
Write-Host "commit-msg will replace vague commit subjects before the commit is created."
Write-Host "pre-push will block vague outgoing commit subjects that still need cleanup."
