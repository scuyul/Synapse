param(
    [string]$Python = "",
    [string]$VenvPath = ".venv",
    [string]$PythonInstaller = "",
    [switch]$BootstrapPython,
    [switch]$SkipRequirements,
    [switch]$FastAppInstall
)

$ErrorActionPreference = "Stop"

function Show-SetupProgress {
    param(
        [string]$Status,
        [int]$Step,
        [int]$Total = 5
    )

    Write-Progress `
        -Activity "Python environment setup" `
        -Status $Status `
        -PercentComplete ([math]::Min(100, [math]::Max(0, (($Step - 1) * 100 / $Total))))
    Write-Host "[$Step/$Total] $Status"
}

function Test-PythonCommand {
    param([string]$Command)

    if (-not $Command) {
        return $false
    }

    try {
        if (Test-Path -LiteralPath $Command) {
            & $Command -c "import sys; print(sys.executable)" *> $null
        }
        else {
            $parts = $Command -split " "
            & $parts[0] @($parts | Select-Object -Skip 1) -c "import sys; print(sys.executable)" *> $null
        }
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-Python {
    param([string]$Preferred)

    if (Test-PythonCommand $Preferred) {
        return $Preferred
    }

    $packagedPython = Join-Path (Get-Location) ".python\python.exe"
    if (Test-PythonCommand $packagedPython) {
        return $packagedPython
    }

    $commands = @("py -3.13", "py -3", "python", "python3")
    foreach ($command in $commands) {
        try {
            $parts = $command -split " "
            & $parts[0] @($parts | Select-Object -Skip 1) -c "import sys; print(sys.executable)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return $command
            }
        }
        catch {
        }
    }

    $localPython = Get-ChildItem `
        -Path "$env:LOCALAPPDATA\Programs\Python" `
        -Filter "python.exe" `
        -Recurse `
        -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($localPython) {
        return $localPython.FullName
    }

    return ""
}

function Install-Python {
    param([string]$BundledInstaller)

    $version = "3.13.13"
    $installer = $BundledInstaller
    $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"

    if ($installer -and (Test-Path -LiteralPath $installer)) {
        Write-Host "Python was not found. Using bundled Python installer: $installer"
    }
    else {
        $installer = Join-Path $env:TEMP "python-$version-amd64.exe"
        Write-Host "Python was not found. Downloading Python $version..."
        Write-Host $url

        $response = Invoke-WebRequest -Uri $url -OutFile $installer -PassThru
        if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
            throw "Python download failed with HTTP status $($response.StatusCode)."
        }
    }

    Write-Host "Installing Python $version for the current user..."
    $arguments = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_launcher=1",
        "Include_pip=1",
        "Include_test=0",
        "SimpleInstall=1"
    )
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -PassThru
    if (-not $process.WaitForExit(180)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "Python installer did not finish within 3 minutes. Use the bundled .python runtime or install Python manually."
    }
    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }
}

function Invoke-Python {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    if (Test-Path -LiteralPath $Command) {
        & $Command @Arguments
    }
    else {
        $parts = $Command -split " "
        & $parts[0] @($parts | Select-Object -Skip 1) @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

$totalSteps = if ($FastAppInstall) { 3 } elseif ($SkipRequirements) { 4 } else { 5 }

Show-SetupProgress "Finding Python" 1 $totalSteps
$resolvedPython = Find-Python $Python
if (-not $resolvedPython -and $BootstrapPython) {
    Install-Python $PythonInstaller
    $resolvedPython = Find-Python ""
}

if (-not $resolvedPython) {
    throw "Python was not found. Install Python 3.11+ or rerun setup with -BootstrapPython."
}

Write-Host "Using Python: $resolvedPython"
Show-SetupProgress "Creating virtual environment at $VenvPath" 2 $totalSteps
Write-Host "Creating virtual environment at $VenvPath"
Invoke-Python $resolvedPython @("-m", "venv", $VenvPath)

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not created at $venvPython"
}

if ($FastAppInstall) {
    Show-SetupProgress "Skipping pip/package install for fast app setup" 3 $totalSteps
    Write-Host "Fast app install: skipping pip upgrade and editable package install."
}
else {
    Show-SetupProgress "Upgrading pip" 3 $totalSteps
    Write-Host "Upgrading pip"
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed with exit code $LASTEXITCODE."
    }
}

if ($SkipRequirements) {
    if (-not $FastAppInstall) {
        Show-SetupProgress "Skipping training requirements" 4 $totalSteps
    }
    Write-Host "Skipping CUDA/training requirements for fast app install."
    Write-Host "Install them later from the app with: Install Training Dependencies"
}
else {
    Show-SetupProgress "Installing CUDA/training requirements" 4 $totalSteps
    Write-Host "Installing CUDA runtime requirements"
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "requirements install failed with exit code $LASTEXITCODE."
    }
}

if (-not $FastAppInstall) {
    Show-SetupProgress "Installing editable package" $totalSteps $totalSteps
    Write-Host "Installing package in editable mode"
    & $venvPython -m pip install -e .
    if ($LASTEXITCODE -ne 0) {
        throw "editable install failed with exit code $LASTEXITCODE."
    }
}

Write-Progress -Activity "Python environment setup" -Completed
Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Verify CUDA with: .\.venv\Scripts\python.exe -m reefscape_rl.doctor"
