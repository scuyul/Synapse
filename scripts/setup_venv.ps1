param(
    [string]$Python = "",
    [string]$VenvPath = ".venv",
    [switch]$BootstrapPython
)

$ErrorActionPreference = "Stop"

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
    $version = "3.13.13"
    $installer = Join-Path $env:TEMP "python-$version-amd64.exe"
    $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"

    Write-Host "Python was not found. Downloading Python $version..."
    Invoke-WebRequest -Uri $url -OutFile $installer

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
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
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

$resolvedPython = Find-Python $Python
if (-not $resolvedPython -and $BootstrapPython) {
    Install-Python
    $resolvedPython = Find-Python ""
}

if (-not $resolvedPython) {
    throw "Python was not found. Install Python 3.11+ or rerun setup with -BootstrapPython."
}

Write-Host "Using Python: $resolvedPython"
Write-Host "Creating virtual environment at $VenvPath"
Invoke-Python $resolvedPython @("-m", "venv", $VenvPath)

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not created at $venvPython"
}

Write-Host "Upgrading pip"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}

Write-Host "Installing CUDA runtime requirements"
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "requirements install failed with exit code $LASTEXITCODE."
}

Write-Host "Installing package in editable mode"
& $venvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    throw "editable install failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Verify CUDA with: .\.venv\Scripts\python.exe -m reefscape_rl.doctor"
