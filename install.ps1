# Annolid One-Line Installation Script for Windows
#
# Usage (one-liner in PowerShell):
#   irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
#
# Usage (with options):
#   .\install.ps1 [-InstallDir DIR] [-Extras EXTRAS] [-NoGpu] [-NoInteractive]
#
# Options:
#   -InstallDir DIR      Directory to install annolid (default: .\annolid)
#   -VenvDir DIR         Directory for virtual environment (default: .venv)
#   -Extras EXTRAS       Comma-separated extras: sam3,image_editing,text_to_speech,qwen3_embedding
#   -NoGpu               Skip GPU/CUDA detection
#   -NoInteractive       Skip all prompts and use defaults

param(
    [string]$InstallDir = "",
    [string]$VenvDir = ".venv",
    [string]$Extras = "",
    [switch]$NoGpu,
    [switch]$NoInteractive,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$PythonMinVersion = [version]"3.10"
$AnnolidRepo = "https://github.com/healthonrails/annolid.git"

# =============================================================================
# Helper Functions
# =============================================================================
function Write-Header {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Blue
    Write-Host "              Annolid Installation Script                        " -ForegroundColor Blue
    Write-Host "     Annotate, Segment, and Track Anything You Need             " -ForegroundColor Blue
    Write-Host "==================================================================" -ForegroundColor Blue
    Write-Host ""
}

function Write-Step { param([string]$Message); Write-Host ">> $Message" -ForegroundColor Green }
function Write-Warning-Msg { param([string]$Message); Write-Host "!! $Message" -ForegroundColor Yellow }
function Write-Error-Msg { param([string]$Message); Write-Host "XX $Message" -ForegroundColor Red }
function Write-Success { param([string]$Message); Write-Host "OK $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message); Write-Host "-- $Message" -ForegroundColor Cyan }

function Prompt-YesNo {
    param([string]$Prompt, [bool]$Default = $false)

    if ($NoInteractive) { return $Default }

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $response = Read-Host "$Prompt $suffix"

    if ([string]::IsNullOrEmpty($response)) { return $Default }
    return $response -match "^[Yy]"
}

function Show-Help {
    Get-Help $PSCommandPath -Detailed
    exit 0
}

# =============================================================================
# Check Git
# =============================================================================
function Test-Git {
    Write-Step "Checking Git installation..."

    try {
        $null = Get-Command git -ErrorAction Stop
        Write-Success "Git found"
    } catch {
        Write-Warning-Msg "Git not found. Attempting automatic installation via Winget..."

        try {
            winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements

            # Simple check if command became available (might need restart)
            if (Get-Command git -ErrorAction SilentlyContinue) {
                Write-Success "Git installed successfully"
            } else {
                Write-Warning-Msg "Git installed but may need shell restart to be recognized."
            }
        } catch {
            Write-Error-Msg "Git automatic installation failed. Please install Git manually."
            Write-Host ""
            Write-Host "  Download from: https://git-scm.com/download/win"
            exit 1
        }
    }
}

# =============================================================================
# Check Python
# =============================================================================
function Test-Python {
    Write-Step "Checking Python installation..."

    $preferredVersions = @("3.11", "3.12", "3.13", "3.10")
    $script:PythonCmd = $null
    $script:PythonVersion = $null

    foreach ($ver in $preferredVersions) {
        foreach ($cmd in @("python$ver", "python")) {
            try {
                $actualVer = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($actualVer -eq $ver) {
                    $script:PythonCmd = $cmd
                    $script:PythonVersion = $actualVer
                    Write-Host "  Found preferred: $cmd (version $actualVer)"
                    break
                }
            } catch { continue }
        }
        if ($script:PythonCmd) { break }
    }

    if (-not $script:PythonCmd) {
        foreach ($cmd in @("python", "python3", "py")) {
            try {
                $version = & $cmd --version 2>&1
                if ($version -match "Python (\d+)\.(\d+)") {
                    $major = [int]$Matches[1]
                    $minor = [int]$Matches[2]
                    if ($major -ge 3 -and $minor -ge 10) {
                        $script:PythonCmd = $cmd
                        $script:PythonVersion = "$major.$minor"
                        Write-Host "  Found: $cmd (version $script:PythonVersion)"
                        break
                    }
                }
            } catch { continue }
        }
    }

    if (-not $script:PythonCmd) {
        Write-Error-Msg "Python $PythonMinVersion or higher not found."
        Write-Host ""
        Write-Host "  Download from: https://www.python.org/downloads/"
        Write-Host "  Or: winget install Python.Python.3.11"
        exit 1
    }

    Write-Success "Python version OK"
    return $script:PythonCmd
}

# =============================================================================
# Check FFmpeg
# =============================================================================
function Test-FFmpeg {
    Write-Step "Checking FFmpeg installation..."

    try {
        $ffmpegVersion = & ffmpeg -version 2>&1 | Select-Object -First 1
        Write-Host "  $ffmpegVersion"
        Write-Success "FFmpeg found"
    } catch {
        Write-Warning-Msg "FFmpeg not found. Attempting automatic installation via Winget..."

        try {
            winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
            Write-Success "FFmpeg installed successfully (may need shell restart)"
        } catch {
            Write-Warning-Msg "Automatic installation failed or canceled."
            Write-Host ""
            Write-Host "  To install: winget install Gyan.FFmpeg"
            Write-Host ""
            if (-not (Prompt-YesNo "  Continue without FFmpeg?" $true)) {
                exit 1
            }
        }
    }
}

# =============================================================================
# Check uv
# =============================================================================
function Test-Uv {
    Write-Step "Checking for uv (fast package installer)..."

    try {
        $uvVersion = & uv --version 2>&1
        Write-Host "  Found: $uvVersion"
        $script:UseUv = $true
    } catch {
        Write-Warning-Msg "uv not found. Using pip (slower but works)."
        Write-Host "  Install uv: irm https://astral.sh/uv/install.ps1 | iex"
        $script:UseUv = $false
    }
}

# =============================================================================
# Interactive Configuration
# =============================================================================
function Get-InteractiveConfig {
    if ($NoInteractive) { return }

    Write-Step "Configuration options..."
    Write-Host ""

    if ([string]::IsNullOrEmpty($InstallDir)) {
        Write-Host "  Where would you like to install Annolid?" -ForegroundColor Cyan
        $response = Read-Host "  Install directory [.\annolid]"
        $script:InstallDir = if ([string]::IsNullOrEmpty($response)) { ".\annolid" } else { $response }
    }

    Write-Host ""
    Write-Host "  Optional features (select with y/n):" -ForegroundColor Cyan

    $selectedExtras = @()
    if (Prompt-YesNo "    Include SAM3 (advanced segmentation)?" $false) { $selectedExtras += "sam3" }
    if (Prompt-YesNo "    Include image editing (diffusion models)?" $false) { $selectedExtras += "image_editing" }
    if (Prompt-YesNo "    Include text-to-speech?" $false) { $selectedExtras += "text_to_speech" }

    if ($selectedExtras.Count -gt 0) {
        $script:Extras = $selectedExtras -join ","
        Write-Info "Selected extras: $script:Extras"
    }

    Write-Host ""
}

# =============================================================================
# Clone Repository
# =============================================================================
function Clone-Repo {
    Write-Step "Cloning Annolid repository..."

    if ([string]::IsNullOrEmpty($InstallDir)) {
        $script:InstallDir = ".\annolid"
    }

    $script:InstallDir = [System.IO.Path]::GetFullPath($InstallDir)

    if (Test-Path $InstallDir) {
        if (Test-Path "$InstallDir\.git") {
            Write-Info "Annolid directory exists. Updating..."
            Set-Location $InstallDir
            git pull --recurse-submodules 2>$null
        } else {
            Write-Warning-Msg "Directory exists but is not a git repository."
            if (Prompt-YesNo "  Remove and re-clone?" $false) {
                Remove-Item -Recurse -Force $InstallDir
                git clone --recurse-submodules $AnnolidRepo $InstallDir
                Set-Location $InstallDir
            } else {
                Set-Location $InstallDir
            }
        }
    } else {
        git clone --recurse-submodules $AnnolidRepo $InstallDir
        Set-Location $InstallDir
    }

    $script:InstallDir = (Get-Location).Path
    Write-Success "Repository ready at $script:InstallDir"
}

# =============================================================================
# Create Virtual Environment
# =============================================================================
function New-Venv {
    Write-Step "Creating virtual environment..."

    $script:VenvPath = Join-Path $InstallDir $VenvDir

    if ($script:UseUv) {
        & uv venv $script:VenvPath
    } else {
        & $script:PythonCmd -m venv $script:VenvPath
    }

    $script:ActivateCmd = "$script:VenvPath\Scripts\Activate.ps1"
    Write-Success "Virtual environment created"
}

# =============================================================================
# Install Annolid
# =============================================================================
function Install-Annolid {
    Write-Step "Installing Annolid..."

    . $script:ActivateCmd

    $pipCmd = if ($script:UseUv) { @("uv", "pip") } else { @("pip") }

    Write-Host "  Upgrading pip..."
    & @pipCmd install --upgrade pip

    $installTarget = "-e ."
    if (-not [string]::IsNullOrEmpty($Extras)) {
        $installTarget = "-e .[$Extras]"
        Write-Host "  Including extras: $Extras"
    }

    Write-Host "  Installing annolid (this may take a few minutes)..."
    & @pipCmd install $installTarget

    Write-Host "  Installing SAM-HQ..."
    try {
        & @pipCmd install "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
    } catch {
        Write-Warning-Msg "SAM-HQ installation failed."
    }

    Write-Success "Annolid installed"
}

# =============================================================================
# Validate Installation
# =============================================================================
function Test-Installation {
    Write-Step "Validating installation..."

    . $script:ActivateCmd

    try {
        $null = Get-Command annolid -ErrorAction Stop
        Write-Success "annolid command available"
    } catch {
        Write-Error-Msg "annolid command not found"
        exit 1
    }

    Write-Host "  Checking imports..."
    & python -c "import annolid; print(f'  Annolid imported successfully')"

    Write-Host "  Checking PyTorch..."
    & python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"

    if (-not $NoGpu) {
        & python -c "import torch; cuda = torch.cuda.is_available(); print(f'  CUDA available: {cuda}')"
    }

    Write-Success "Installation validated"
}

# =============================================================================
# Print Summary
# =============================================================================
function Write-Summary {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host "              Installation Complete!                              " -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Annolid has been installed to: $script:InstallDir"
    Write-Host ""
    Write-Host "To get started:"
    Write-Host ""
    Write-Host "  1. Navigate to the directory:"
    Write-Host "     cd $script:InstallDir" -ForegroundColor Blue
    Write-Host ""
    Write-Host "  2. Activate the environment:"
    Write-Host "     $script:ActivateCmd" -ForegroundColor Blue
    Write-Host ""
    Write-Host "  3. Launch Annolid:"
    Write-Host "     annolid" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Documentation: https://annolid.com"
    Write-Host "Issues: https://github.com/healthonrails/annolid/issues"
    Write-Host ""

    if (Prompt-YesNo "Launch Annolid now?" $true) {
        Write-Step "Launching Annolid..."

        # We need to activate environment and run annolid
        . $script:ActivateCmd
        annolid
    }
}

# =============================================================================
# Main
# =============================================================================
if ($Help) { Show-Help }

Write-Header
Test-Git
$pythonCmd = Test-Python
Test-FFmpeg
Test-Uv
Get-InteractiveConfig
Clone-Repo
New-Venv
Install-Annolid
Test-Installation
Write-Summary
