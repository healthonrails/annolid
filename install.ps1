# Annolid One-Line Installation Script for Windows
#
# Usage (one-liner in PowerShell):
#   irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
#
# Usage (with options):
#   .\install.ps1 [-InstallDir DIR] [-VenvDir DIR] [-Profile PROFILE] [-Extras EXTRAS] [-Wheelhouse DIR] [-Offline] [-NoGpu] [-NoInteractive]
#
# Options:
#   -InstallDir DIR      Directory to install annolid (default: .\annolid)
#   -VenvDir DIR         Directory for virtual environment (default: .venv)
#   -Profile PROFILE     Install profile: minimal, gui, tracking, workstation, full (default: gui)
#   -Extras EXTRAS       Comma-separated extras: webengine,onnx,scientific,pdf,model_downloads,sleap,audio,ai_chat,training,ml,tracking,yolo,cutie,realtime,large_image,remote_video,sam3,image_editing,text_to_speech,qwen3_embedding,annolid_bot,bot,memory,all (GUI extras are always included)
#   -Wheelhouse DIR      Local wheelhouse directory containing .whl packages for offline/cached installs
#   -Offline             Disable network access (--no-index, skip git pull/uv download); requires -Wheelhouse
#   -NoGpu               Skip GPU/CUDA detection
#   -NoInteractive       Skip all prompts and use defaults

param(
    [string]$InstallDir = "",
    [string]$VenvDir = ".venv",
    [ValidateSet("minimal", "gui", "tracking", "workstation", "full")]
    [string]$Profile = "gui",
    [string]$Extras = "",
    [string]$Wheelhouse = $(if ($env:ANNOLID_WHEELHOUSE) { $env:ANNOLID_WHEELHOUSE } else { "" }),
    [switch]$Offline,
    [switch]$NoGpu,
    [switch]$NoInteractive,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$PythonMinVersion = [version]"3.10"
$AnnolidRepo = "https://github.com/healthonrails/annolid.git"
$script:OfflineActive = [bool]$Offline -or (@("1", "true", "yes", "on") -contains ([string]$env:ANNOLID_OFFLINE).Trim().ToLowerInvariant())
$script:WheelhouseDir = $null
$script:UseUvPython = $false
$script:UvPythonVersion = $null
$script:HasNvidiaGpu = $false
$script:HasCuda12 = $false
$script:CudaVersion = $null
$script:InstallExtras = @()
$script:InstallReportPath = $null
$script:SamHqStatus = "not_requested"
$script:FailedOptionalSteps = @()
$script:PytorchCudaIndexUrl = "https://download.pytorch.org/whl/cu124"
$script:OnnxRuntimeCuda12ExtraIndexUrl = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

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

function Convert-CommandOutputToString {
    param([object]$Value)

    if ($null -eq $Value) { return "" }
    return [string]::Join("", @($Value))
}

function Add-OptionalFailure {
    param([string]$Step)
    if (-not ($script:FailedOptionalSteps -contains $Step)) {
        $script:FailedOptionalSteps += $Step
    }
}

function Prompt-YesNo {
    param([string]$Prompt, [bool]$Default = $false)

    if ($NoInteractive) { return $Default }

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $response = Read-Host "$Prompt $suffix"

    if ([string]::IsNullOrEmpty($response)) { return $Default }
    return $response -match "^[Yy]"
}

function Show-Help {
    @"
Annolid one-line installer for Windows PowerShell

Usage:
  irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
  .\install.ps1 [-InstallDir DIR] [-VenvDir DIR] [-Profile PROFILE] [-Extras EXTRAS] [-Wheelhouse DIR] [-Offline] [-NoGpu] [-NoInteractive]

Options:
  -InstallDir DIR      Directory to install Annolid (default: .\annolid)
  -VenvDir DIR         Directory for virtual environment (default: .venv)
  -Profile PROFILE     Install profile: minimal, gui, tracking, workstation, full (default: gui)
  -Extras EXTRAS       Comma-separated extras: webengine,onnx,scientific,pdf,model_downloads,sleap,audio,ai_chat,training,ml,tracking,yolo,cutie,realtime,large_image,remote_video,sam3,image_editing,text_to_speech,qwen3_embedding,annolid_bot,bot,memory,all
  -Wheelhouse DIR      Local wheelhouse directory containing .whl packages for offline/cached installs
  -Offline             Disable network access (--no-index, skip git pull/uv download); requires -Wheelhouse
  -NoGpu               Skip GPU/CUDA detection and install CPU ONNX Runtime
  -NoInteractive       Skip prompts and use defaults
  -Help                Show this help message
"@ | Write-Host
    exit 0
}

function Initialize-OfflineConfig {
    if (-not [string]::IsNullOrWhiteSpace($Wheelhouse)) {
        $resolvedWh = [System.IO.Path]::GetFullPath($Wheelhouse)
        if (-not (Test-Path -LiteralPath $resolvedWh -PathType Container)) {
            Write-Error-Msg "Wheelhouse directory does not exist: $resolvedWh"
            exit 1
        }
        $script:WheelhouseDir = $resolvedWh
    }
    if ($script:OfflineActive -and -not $script:WheelhouseDir) {
        Write-Error-Msg "-Offline (or ANNOLID_OFFLINE=1) requires -Wheelhouse DIR (or ANNOLID_WHEELHOUSE)."
        exit 1
    }
}

function Get-PipIndexArgs {
    $argsList = @()
    if ($script:OfflineActive) {
        $argsList += "--no-index"
    }
    if ($script:WheelhouseDir) {
        $argsList += @("--find-links", $script:WheelhouseDir)
    }
    return @($argsList)
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
        if ($script:OfflineActive) {
            Write-Info "Offline mode active and Git not found; expecting an existing local Annolid checkout."
            return
        }
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

    $script:PythonCmd = $null
    $script:PythonArgs = @()
    $script:PythonVersion = $null

    $preferredCandidates = @(
        @{ Command = "py"; Args = @("-3.11"); Version = "3.11" },
        @{ Command = "py"; Args = @("-3.12"); Version = "3.12" },
        @{ Command = "py"; Args = @("-3.13"); Version = "3.13" },
        @{ Command = "py"; Args = @("-3.14"); Version = "3.14" },
        @{ Command = "py"; Args = @("-3.15"); Version = "3.15" },
        @{ Command = "py"; Args = @("-3.10"); Version = "3.10" },
        @{ Command = "python3.11"; Args = @(); Version = "3.11" },
        @{ Command = "python3.12"; Args = @(); Version = "3.12" },
        @{ Command = "python3.13"; Args = @(); Version = "3.13" },
        @{ Command = "python3.14"; Args = @(); Version = "3.14" },
        @{ Command = "python3.15"; Args = @(); Version = "3.15" },
        @{ Command = "python3.10"; Args = @(); Version = "3.10" }
    )

    foreach ($candidate in $preferredCandidates) {
        try {
            $actualVer = & $candidate.Command @($candidate.Args) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if (-not [string]::IsNullOrWhiteSpace($actualVer) -and $actualVer -eq $candidate.Version) {
                $script:PythonCmd = $candidate.Command
                $script:PythonArgs = @($candidate.Args)
                $script:PythonVersion = $actualVer
                $displayCmd = @($candidate.Command) + @($candidate.Args) -join " "
                Write-Host "  Found preferred: $displayCmd (version $actualVer)"
                break
            }
        } catch {
            continue
        }
    }

    if (-not $script:PythonCmd) {
        $fallbackCandidates = @(
            @{ Command = "python"; Args = @() },
            @{ Command = "python3"; Args = @() },
            @{ Command = "py"; Args = @() }
        )

        foreach ($candidate in $fallbackCandidates) {
            try {
                $version = & $candidate.Command @($candidate.Args) --version 2>&1
                if ($version -match "Python (\d+)\.(\d+)") {
                    $major = [int]$Matches[1]
                    $minor = [int]$Matches[2]
                    if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                        $script:PythonCmd = $candidate.Command
                        $script:PythonArgs = @($candidate.Args)
                        $script:PythonVersion = "$major.$minor"
                        Write-Host "  Found: $script:PythonCmd (version $script:PythonVersion)"
                        break
                    }
                }
            } catch { continue }
        }
    }

    if (-not $script:PythonCmd) {
        if ($script:UseUv) {
            $script:UseUvPython = $true
            $script:UvPythonVersion = "3.12"
            $script:PythonVersion = $script:UvPythonVersion
            Write-Info "System Python $PythonMinVersion+ not found; will create the environment with uv-managed Python $script:UvPythonVersion"
            Write-Success "Python version OK"
            return $null
        }

        Write-Error-Msg "Python $PythonMinVersion or higher not found."
        Write-Host ""
        Write-Host "  Download from: https://www.python.org/downloads/"
        Write-Host "  Or: winget install Python.Python.3.11"
        exit 1
    }

    # Future Python versions may need a compatibility catch-up period.
    if ($script:PythonVersion -match "^(\d+)\.(\d+)$") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]

        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Error-Msg "Python $PythonMinVersion or higher is required. Found: $script:PythonVersion"
            exit 1
        }

        if ($major -eq 3 -and ($minor -eq 14 -or $minor -eq 15)) {
            Write-Info "Python $script:PythonVersion detected. Annolid supports the default GUI/core install on Python $script:PythonVersion."
            Write-Info "Remote network video decoding still needs the optional remote_video extra and native FFmpeg development libraries."
        } elseif ($major -gt 3 -or ($major -eq 3 -and $minor -ge 16)) {
            Write-Warning-Msg "Python $script:PythonVersion detected. Annolid is not yet validated on Python 3.16+."
            if ($script:UseUv -and (Prompt-YesNo "  Use uv to install with Python 3.12?" $true)) {
                $script:UseUvPython = $true
                $script:UvPythonVersion = "3.12"
                Write-Info "Will create the environment with uv-managed Python $script:UvPythonVersion"
            } elseif (-not (Prompt-YesNo "  Continue with Python $script:PythonVersion anyway? (may fail)" $false)) {
                Write-Error-Msg "uv is required to provision a compatible Python version automatically."
                Write-Host "  Please install Python 3.12 manually, or install uv and rerun."
                exit 1
            }
        }
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
        if ($script:OfflineActive) {
            Write-Warning-Msg "FFmpeg not found (offline mode: skipping Winget download)."
            return
        }
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
# Check GPU
# =============================================================================
function Test-GPU {
    if ($NoGpu) {
        Write-Info "GPU detection skipped (--NoGpu)."
        $script:HasNvidiaGpu = $false
        $script:HasCuda12 = $false
        $script:CudaVersion = $null
        return
    }

    Write-Step "Checking for NVIDIA GPU..."

    try {
        $null = Get-Command nvidia-smi -ErrorAction Stop
        $gpuList = & nvidia-smi -L 2>$null
        if (-not [string]::IsNullOrWhiteSpace($gpuList)) {
            $script:HasNvidiaGpu = $true
            Write-Success "NVIDIA GPU detected"
            Write-Host "  $($gpuList -split "`n" | Select-Object -First 1)"

            try {
                $smiHeader = & nvidia-smi 2>$null | Select-String -Pattern "CUDA Version:\s*([0-9]+\.[0-9]+)" | Select-Object -First 1
                if ($smiHeader -and $smiHeader.Matches.Count -gt 0) {
                    $script:CudaVersion = $smiHeader.Matches[0].Groups[1].Value
                    $cudaMajor = [int]($script:CudaVersion.Split(".")[0])
                    $script:HasCuda12 = $cudaMajor -ge 12
                    Write-Info "Detected CUDA runtime version: $script:CudaVersion"
                    if ($script:HasCuda12) {
                        Write-Success "CUDA $script:CudaVersion detected; ONNX Runtime GPU (CUDA 12 feed) will be used."
                        if ($cudaMajor -gt 12) {
                            Write-Info "Using stable CUDA 12 ONNX Runtime wheels; NVIDIA drivers are backward compatible with CUDA 12 runtimes."
                        }
                    } else {
                        Write-Warning-Msg "CUDA $script:CudaVersion detected. ONNX Runtime GPU requires a CUDA 12-compatible NVIDIA driver."
                    }
                } else {
                    Write-Warning-Msg "Unable to parse CUDA version from nvidia-smi output."
                }
            } catch {
                Write-Warning-Msg "Could not detect CUDA runtime version."
            }
            return
        }
    } catch { }

    $script:HasNvidiaGpu = $false
    $script:HasCuda12 = $false
    $script:CudaVersion = $null
    Write-Info "No NVIDIA GPU detected. Using default PyTorch build."
}

# =============================================================================
# Install ONNX Runtime (Windows, CUDA-aware)
# =============================================================================
function Uninstall-PythonPackages {
    param([string[]]$Packages)

    if ($script:UseUv) {
        & $script:UvCmd pip uninstall @Packages | Out-Null
    } else {
        & pip uninstall -y @Packages | Out-Null
    }
}

function Install-OnnxRuntime {
    Write-Step "Installing ONNX Runtime..."

    . $script:ActivateCmd
    $pipIndexArgs = Get-PipIndexArgs

    if ($script:HasNvidiaGpu -and $script:HasCuda12 -and -not $NoGpu) {
        Write-Host "  Installing onnxruntime-gpu..."
        try {
            Uninstall-PythonPackages @("onnxruntime-gpu", "onnxruntime-directml")
            if ($script:OfflineActive) {
                if ($script:UseUv) {
                    & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu
                } else {
                    & pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu
                }
            } else {
                if ($script:UseUv) {
                    & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu --extra-index-url $script:OnnxRuntimeCuda12ExtraIndexUrl
                } else {
                    & pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu --extra-index-url $script:OnnxRuntimeCuda12ExtraIndexUrl
                }
            }
            Write-Success "onnxruntime-gpu (CUDA 12.x) installed"
            return
        } catch {
            Write-Warning-Msg "onnxruntime-gpu install failed. Falling back to CPU onnxruntime."
        }
    } elseif ($script:HasNvidiaGpu -and -not $NoGpu) {
        Write-Info "Skipping onnxruntime-gpu because a CUDA 12-compatible NVIDIA driver was not detected."
    }

    Uninstall-PythonPackages @("onnxruntime-gpu", "onnxruntime-directml")
    if ($script:UseUv) {
        & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime
    } else {
        & pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime
    }
    Write-Success "onnxruntime (CPU) installed"
}

# =============================================================================
# Check uv
# =============================================================================
function Test-Uv {
    Write-Step "Checking for uv (fast package installer)..."

    $script:UvCmd = $null

    try {
        $uvCmdResolved = (Get-Command uv -ErrorAction Stop).Path
        $script:UvCmd = $uvCmdResolved
    } catch { }

    if (-not $script:UvCmd) {
        if ($script:OfflineActive) {
            Write-Info "Offline mode active and uv not found; using system Python venv + pip."
            $script:UseUv = $false
            return
        }
        Write-Warning-Msg "uv not found. Installing via official installer..."

        try {
            # Official installer from Astral: https://docs.astral.sh/uv/getting-started/installation/#powershell
            irm https://astral.sh/uv/install.ps1 | iex
        } catch {
            Write-Error-Msg "Failed to install uv automatically."
            Write-Host "  Please install manually with:"
            Write-Host "    irm https://astral.sh/uv/install.ps1 | iex"
            exit 1
        }

        # Common install locations we can use immediately without a new shell
        $uvCandidates = @(
            "$env:LOCALAPPDATA\\Programs\\uv\\uv.exe",
            "$env:USERPROFILE\\.local\\bin\\uv.exe",
            "$env:USERPROFILE\\.local\\bin\\uv",
            "$env:LOCALAPPDATA\\Microsoft\\WindowsApps\\uv.exe"
        )

        foreach ($candidate in $uvCandidates) {
            if (Test-Path $candidate) {
                $script:UvCmd = $candidate
                $env:PATH = "$(Split-Path $candidate);$env:PATH"
                break
            }
        }

        if (-not $script:UvCmd) {
            try {
                $script:UvCmd = (Get-Command uv -ErrorAction Stop).Path
            } catch {
                Write-Error-Msg "uv installation completed but command still not found."
                Write-Host "  Please restart the shell or ensure uv is in PATH."
                exit 1
            }
        }
    }

    $uvVersion = & $script:UvCmd --version 2>&1
    Write-Host "  Found: $uvVersion"
    $script:UseUv = $true
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
    if (Prompt-YesNo "    Include YOLO/YOLOE workflows and training dashboards?" $false) {
        $selectedExtras += "yolo"
        $selectedExtras += "training"
    }
    if (Prompt-YesNo "    Include hosted AI chat providers?" $false) { $selectedExtras += "ai_chat" }
    if (Prompt-YesNo "    Include image editing (diffusion models)?" $false) { $selectedExtras += "image_editing" }
    if (Prompt-YesNo "    Include audio helpers?" $false) { $selectedExtras += "audio" }
    if (Prompt-YesNo "    Include text-to-speech?" $false) { $selectedExtras += "text_to_speech" }
    if (Prompt-YesNo "    Include realtime hardware/streaming integrations?" $false) { $selectedExtras += "realtime" }
    if (Prompt-YesNo "    Include Annolid Bot integrations (WhatsApp + Google Calendar + MCP)?" $false) { $selectedExtras += "annolid_bot" }

    if ($selectedExtras.Count -gt 0) {
        $script:Extras = $selectedExtras -join ","
        Write-Info "Selected extras: $script:Extras"
    }

    Write-Host ""
}

function Get-ProfileExtras {
    param([string]$ProfileName)

    switch ($ProfileName) {
        "minimal" { return @() }
        "gui" { return @() }
        "tracking" { return @("ml", "tracking", "cutie") }
        "workstation" { return @("ml", "tracking", "cutie", "sam3", "training", "webengine", "pdf") }
        "full" { return @("all") }
        default {
            Write-Error-Msg "Unknown install profile: $ProfileName"
            Write-Host "  Supported profiles: minimal, gui, tracking, workstation, full"
            exit 1
        }
    }
}

function Merge-Extras {
    param([string[]]$Values)

    $merged = New-Object System.Collections.Generic.List[string]
    foreach ($item in $Values) {
        if ([string]::IsNullOrWhiteSpace($item)) { continue }
        foreach ($value in $item.Split(",")) {
            $normalized = $value.Trim()
            if ([string]::IsNullOrEmpty($normalized)) { continue }
            if (-not ($merged -contains $normalized)) {
                $merged.Add($normalized)
            }
        }
    }
    return @($merged)
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

    if (Test-Path -LiteralPath $InstallDir) {
        if (!(Test-Path -LiteralPath "$InstallDir\annolid\__init__.py") -or
            !(Test-Path -LiteralPath "$InstallDir\pyproject.toml")) {
            throw "Directory is not an Annolid checkout. Choose a new -InstallDir."
        }
        Set-Location -LiteralPath $InstallDir
        if ($script:OfflineActive) {
            Write-Info "Offline mode active: using existing local Annolid checkout at $((Get-Location).Path) without git pull."
        } else {
            if (!(Test-Path -LiteralPath "$InstallDir\.git")) {
                throw "Directory is not a Git checkout. Choose a new -InstallDir or pass -Offline with -Wheelhouse."
            }
            $changes = & git status --porcelain --untracked-files=no
            if ($LASTEXITCODE -ne 0) { throw "Could not inspect checkout state." }
            if ($changes) { throw "Local tracked changes found. Commit or stash them before updating." }
            Write-Info "Updating the existing checkout (fast-forward only)..."
            & git pull --ff-only --recurse-submodules
            if ($LASTEXITCODE -ne 0) { throw "Update failed. Resolve the Git error above and rerun; the environment has not been changed." }
        }
    } else {
        if ($script:OfflineActive) {
            throw "Offline mode requires an existing Annolid source directory at $InstallDir (-InstallDir)."
        }
        & git clone --recurse-submodules $AnnolidRepo $InstallDir
        if ($LASTEXITCODE -ne 0) { throw "Clone failed. Check the network and destination path." }
        Set-Location -LiteralPath $InstallDir
    }

    $script:InstallDir = (Get-Location).Path
    Write-Success "Repository ready at $script:InstallDir"
}

# =============================================================================
# Create Virtual Environment
# =============================================================================
function New-Venv {
    Write-Step "Creating virtual environment..."

    $script:VenvPath = if ([System.IO.Path]::IsPathRooted($VenvDir)) { $VenvDir } else { Join-Path $InstallDir $VenvDir }
    $script:ActivateCmd = "$script:VenvPath\Scripts\Activate.ps1"
    if (Test-Path -LiteralPath $script:VenvPath) {
        if (!(Test-Path -LiteralPath "$script:VenvPath\pyvenv.cfg") -or
            !(Test-Path -LiteralPath $script:ActivateCmd) -or
            !(Test-Path -LiteralPath "$script:VenvPath\Scripts\python.exe")) {
            throw "Existing environment is invalid. Choose another -VenvDir or repair it."
        }
        & "$script:VenvPath\Scripts\python.exe" -c "import sys; assert sys.prefix != sys.base_prefix"
        if ($LASTEXITCODE -ne 0) { throw "Existing environment cannot run. Repair it before updating." }
        Write-Success "Reusing existing virtual environment"
        return
    }

    if ($script:UseUv) {
        if ($script:UseUvPython -and $script:UvPythonVersion) {
            & $script:UvCmd venv $script:VenvPath --python $script:UvPythonVersion
        } elseif (-not [string]::IsNullOrWhiteSpace($script:PythonVersion)) {
            & $script:UvCmd venv $script:VenvPath --python $script:PythonVersion
        } else {
            & $script:UvCmd venv $script:VenvPath
        }
    } else {
        & $script:PythonCmd @($script:PythonArgs) -m venv $script:VenvPath
    }

    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    $script:ActivateCmd = "$script:VenvPath\Scripts\Activate.ps1"
    Write-Success "Virtual environment created"
}

# =============================================================================
# Install Annolid
# =============================================================================
function Install-Annolid {
    Write-Step "Installing Annolid..."

    . $script:ActivateCmd
    $pipIndexArgs = Get-PipIndexArgs

    if ($script:OfflineActive) {
        if ($script:WheelhouseDir -and (Get-ChildItem -LiteralPath $script:WheelhouseDir -Filter "pip-*.whl" -ErrorAction SilentlyContinue)) {
            Write-Host "  Upgrading pip from offline wheelhouse..."
            if ($script:UseUv) { & $script:UvCmd pip install @pipIndexArgs --upgrade pip } else { & pip install @pipIndexArgs --upgrade pip }
        }
        if ($script:WheelhouseDir -and (Get-ChildItem -LiteralPath $script:WheelhouseDir -Filter "msvc_runtime-*.whl" -ErrorAction SilentlyContinue)) {
            if ($script:UseUv) { & $script:UvCmd pip install @pipIndexArgs --upgrade msvc-runtime } else { & pip install @pipIndexArgs --upgrade msvc-runtime }
        }
    } else {
        Write-Host "  Upgrading pip..."
        if ($script:UseUv) {
            & $script:UvCmd pip install @pipIndexArgs --upgrade pip
            & $script:UvCmd pip install @pipIndexArgs --upgrade msvc-runtime
        } else {
            & pip install @pipIndexArgs --upgrade pip
            & pip install @pipIndexArgs --upgrade msvc-runtime
        }
    }

    # GUI dependencies are always installed so `annolid` can launch immediately.
    $profileExtras = Get-ProfileExtras $Profile
    $installExtras = Merge-Extras (@("gui") + $profileExtras + @($Extras))
    $script:InstallExtras = $installExtras
    $extrasCsv = $installExtras -join ","
    $installTarget = ".[$extrasCsv]"
    Write-Host "  Profile: $Profile"
    Write-Host "  Including extras: $extrasCsv"
    if ($script:WheelhouseDir) {
        Write-Host "  Wheelhouse: $script:WheelhouseDir (offline=$($script:OfflineActive))"
    }

    if (Test-PytorchRequested $installExtras) {
        if ($script:OfflineActive) {
            Write-Host "  Installing PyTorch from offline wheelhouse..."
            if ($script:UseUv) {
                & $script:UvCmd pip install @pipIndexArgs torch torchvision
            } else {
                & pip install @pipIndexArgs torch torchvision
            }
        } elseif ($script:HasNvidiaGpu -and -not $NoGpu) {
            Write-Host "  Installing CUDA-enabled PyTorch..."
            try {
                if ($script:UseUv) {
                    & $script:UvCmd pip install @pipIndexArgs --index-url $script:PytorchCudaIndexUrl torch torchvision
                } else {
                    & pip install @pipIndexArgs --index-url $script:PytorchCudaIndexUrl torch torchvision
                }
                Write-Success "CUDA-enabled PyTorch installed"
            } catch {
                Write-Warning-Msg "CUDA-enabled PyTorch install failed. Falling back to default PyTorch build."
                if ($script:UseUv) {
                    & $script:UvCmd pip install @pipIndexArgs torch torchvision
                } else {
                    & pip install @pipIndexArgs torch torchvision
                }
            }
        }
    } else {
        Write-Info "Skipping upfront PyTorch download for lean profile ($Profile); ML runtimes install on demand when first used."
    }

    Write-Host "  Installing annolid (this may take a few minutes)..."
    if ($script:UseUv) {
        & $script:UvCmd pip install @pipIndexArgs -e $installTarget
        if ($LASTEXITCODE -ne 0 -and $script:OfflineActive) {
            & $script:UvCmd pip install @pipIndexArgs --no-build-isolation -e $installTarget
        }
    } else {
        & pip install @pipIndexArgs -e $installTarget
        if ($LASTEXITCODE -ne 0 -and $script:OfflineActive) {
            & pip install @pipIndexArgs --no-build-isolation -e $installTarget
        }
    }

    if ($LASTEXITCODE -ne 0) { throw "Annolid package installation failed. Review the error above and rerun." }

    if (Test-ExtraSelected $installExtras @("sam3", "sam", "segment_anything", "all")) {
        if ($script:OfflineActive) {
            if ($script:WheelhouseDir -and (Get-ChildItem -LiteralPath $script:WheelhouseDir -Filter "segment_anything-*.whl" -ErrorAction SilentlyContinue)) {
                Write-Host "  Installing SAM-HQ from offline wheelhouse..."
                try {
                    if ($script:UseUv) {
                        & $script:UvCmd pip install @pipIndexArgs segment-anything
                    } else {
                        & pip install @pipIndexArgs segment-anything
                    }
                    $script:SamHqStatus = "installed"
                } catch {
                    $script:SamHqStatus = "failed"
                    Add-OptionalFailure "sam_hq"
                }
            } else {
                $script:SamHqStatus = "skipped_offline"
                Write-Info "Offline mode active: skipping GitHub SAM-HQ VCS install (no segment_anything wheel in wheelhouse)."
            }
        } else {
            Write-Host "  Installing SAM-HQ..."
            $script:SamHqStatus = "requested"
            try {
                if ($script:UseUv) {
                    & $script:UvCmd pip install @pipIndexArgs "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
                } else {
                    & pip install @pipIndexArgs "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
                }
                $script:SamHqStatus = "installed"
            } catch {
                $script:SamHqStatus = "failed"
                Add-OptionalFailure "sam_hq"
                Write-Warning-Msg "SAM-HQ installation failed."
            }
        }
    } else {
        $script:SamHqStatus = "skipped"
        Write-Info "Skipping optional SAM-HQ install. Add -Extras sam3 when needed."
    }

    if (Test-OnnxRequested $installExtras) {
        Install-OnnxRuntime
    } else {
        Write-Info "Skipping upfront ONNX Runtime install for lean profile ($Profile); ONNX installs on demand when first used."
    }

    Write-Success "Annolid installed"
}

function Test-ExtraSelected {
    param(
        [string[]]$SelectedExtras,
        [string[]]$Needles
    )

    $selected = @($SelectedExtras | ForEach-Object { $_.Trim().ToLowerInvariant() })
    foreach ($needle in $Needles) {
        if ($selected -contains $needle.ToLowerInvariant()) {
            return $true
        }
    }
    return $false
}

function Test-PytorchRequested {
    param([string[]]$SelectedExtras)
    return Test-ExtraSelected $SelectedExtras @(
        "ml", "tracking", "cutie", "sam3", "yolo", "image_editing", "qwen3_embedding", "all"
    )
}

function Test-OnnxRequested {
    param([string[]]$SelectedExtras)
    return Test-ExtraSelected $SelectedExtras @("onnx", "onnx_gpu", "ml", "tracking", "all")
}

# =============================================================================
# Repair ONNX Runtime (Windows DLL dependencies)
# =============================================================================
function Repair-OnnxRuntime {
    if (-not (Test-OnnxRequested $script:InstallExtras)) {
        Write-Info "Skipping ONNX Runtime repair check; ONNX Runtime is not requested in this profile."
        return
    }
    Write-Step "Checking ONNX Runtime import..."

    . $script:ActivateCmd

    $ortImportOk = $true
    try {
        & python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import onnxruntime as ort; print(f'  ONNX Runtime: {ort.__version__}')"
    } catch {
        $ortImportOk = $false
    }

    if ($ortImportOk) {
        Write-Success "ONNX Runtime import OK"
        return
    }

    Write-Warning-Msg "ONNX Runtime failed to import. Repairing runtime dependencies..."
    $pipIndexArgs = Get-PipIndexArgs

    Uninstall-PythonPackages @("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
    if (-not $script:OfflineActive -or ($script:WheelhouseDir -and (Get-ChildItem -LiteralPath $script:WheelhouseDir -Filter "msvc_runtime-*.whl" -ErrorAction SilentlyContinue))) {
        if ($script:UseUv) {
            & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall msvc-runtime
        } else {
            & pip install @pipIndexArgs --upgrade --force-reinstall msvc-runtime
        }
    }

    if ($script:HasNvidiaGpu -and $script:HasCuda12 -and -not $NoGpu) {
        Write-Info "Attempting onnxruntime-gpu repair for CUDA 12.x..."
        if ($script:OfflineActive) {
            if ($script:UseUv) {
                & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu
            } else {
                & pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu
            }
        } else {
            if ($script:UseUv) {
                & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu --extra-index-url $script:OnnxRuntimeCuda12ExtraIndexUrl
            } else {
                & pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime-gpu --extra-index-url $script:OnnxRuntimeCuda12ExtraIndexUrl
            }
        }
    } else {
        Write-Info "Using CPU onnxruntime for repair."
        if ($script:UseUv) {
            & $script:UvCmd pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime
        } else {
            & pip install @pipIndexArgs --upgrade --force-reinstall onnxruntime
        }
    }

    try {
        & python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import onnxruntime as ort; providers = ort.get_available_providers(); print(f'  ONNX Runtime repaired: {ort.__version__}'); print(f'  Providers: {providers}')"
        Write-Success "ONNX Runtime repaired successfully"
    } catch {
        Write-Error-Msg "ONNX Runtime still fails to import."
        Write-Host "  Please install Microsoft Visual C++ Redistributable (x64), then rerun the installer."
        Write-Host "  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe"
        exit 1
    }
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

    Write-Host "  Checking Qt bindings..."
    try {
        & python -c "from qtpy import QtCore; print('  Qt binding import OK')"
    } catch {
        Write-Error-Msg "Qt bindings are missing. Reinstall with GUI extras:"
        Write-Host "  pip install -e "".[gui]"""
        exit 1
    }

    Write-Host "  Checking PyTorch availability..."
    $torchOk = $true
    try {
        & python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
    } catch {
        $torchOk = $false
    }

    if ($torchOk -and -not $NoGpu) {
        & python -c "import torch; cuda = torch.cuda.is_available(); print(f'  CUDA available: {cuda}')"
    } elseif (-not $torchOk) {
        Write-Info "PyTorch not installed; optional ML/tracking features will install or report their requirements on first use."
    }

    Write-Host "  Checking ONNX Runtime providers..."
    if (Test-OnnxRequested $script:InstallExtras) {
        & python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import onnxruntime as ort; providers = ort.get_available_providers(); print(f'  ONNX providers: {providers}')"

        if ($script:HasNvidiaGpu -and $script:HasCuda12 -and -not $NoGpu) {
            Write-Host "  Verifying CUDAExecutionProvider..."
            try {
                & python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import onnxruntime as ort; providers = set(ort.get_available_providers()); assert 'CUDAExecutionProvider' in providers, f'Missing CUDAExecutionProvider. Providers={providers}'"
                Write-Success "CUDAExecutionProvider available"
            } catch {
                Write-Error-Msg "CUDAExecutionProvider is missing despite CUDA 12.x detection."
                Write-Host "  Run this in the venv and retry:"
                Write-Host "    uv pip install onnxruntime-gpu --extra-index-url $script:OnnxRuntimeCuda12ExtraIndexUrl"
                exit 1
            }
        }
    } else {
        try {
            & python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import onnxruntime as ort; providers = ort.get_available_providers(); print(f'  ONNX providers: {providers}')" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Info "ONNX Runtime not installed; optional ONNX features will install on first use."
            }
        } catch {
            Write-Info "ONNX Runtime not installed; optional ONNX features will install on first use."
        }
    }

    Write-Host "  Checking AI polygon model path..."
    & python -c "import importlib.util; parent = importlib.util.find_spec('pycocotools'); raise SystemExit(0 if parent and importlib.util.find_spec('pycocotools.mask') else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import numpy as np; from annolid.utils.annotation_compat import get_ai_models; Model = get_ai_models()[0]; m = Model(); m.set_image(np.zeros((64, 64, 3), dtype=np.uint8)); pts = np.array([[16,16],[48,48]], dtype=np.float32); labels = np.array([1,1], dtype=np.int32); poly = m.predict_polygon_from_points(pts, labels); print(f'  AI polygon check OK (points={len(poly)})')"
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Msg "AI polygon validation failed despite optional pycocotools being installed."
            exit 1
        }
    } else {
        Write-Info "Skipping AI polygon check; optional pycocotools is not installed in this profile."
    }

    Write-Success "Installation validated"
}

# =============================================================================
# Install Report
# =============================================================================
function Write-InstallReport {
    Write-Step "Writing machine-readable install report..."

    . $script:ActivateCmd

    try {
        $script:InstallReportPath = Join-Path $script:InstallDir "annolid-install-report.json"
        $packageManager = if ($script:UseUv) { "uv" } else { "pip" }
        $pythonVersionOutput = & python -c "import platform; print(platform.python_version())" 2>$null
        $pythonVersionText = Convert-CommandOutputToString $pythonVersionOutput
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($pythonVersionText)) {
            $pythonVersion = $pythonVersionText.Trim()
        } else {
            $pythonVersion = if ([string]::IsNullOrWhiteSpace($script:PythonVersion)) { "unknown" } else { $script:PythonVersion }
        }
        $onnxProviders = @()
        $providersJson = & python -c "import json, os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); import onnxruntime as ort; print(json.dumps(ort.get_available_providers()))" 2>$null
        $providersText = Convert-CommandOutputToString $providersJson
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($providersText)) {
            $onnxProviders = @($providersText | ConvertFrom-Json)
        }

        $archValue = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
        $arch = if ($null -ne $archValue) { [string]$archValue } elseif ($env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE } else { "unknown" }
        $createdAt = Get-Date -Format "o"

        $report = [ordered]@{
            profile = $Profile
            extras = @($script:InstallExtras)
            install_dir = $script:InstallDir
            venv_path = $script:VenvPath
            python_version = $pythonVersion
            os = "windows"
            distro = "windows"
            arch = $arch
            package_manager = $packageManager
            gpu_requested = (-not [bool]$NoGpu)
            has_nvidia_gpu = [bool]$script:HasNvidiaGpu
            has_cuda12_driver = [bool]$script:HasCuda12
            cuda_version = $script:CudaVersion
            onnx_providers = @($onnxProviders)
            sam_hq_status = $script:SamHqStatus
            offline = [bool]$script:OfflineActive
            wheelhouse = $script:WheelhouseDir
            failed_optional_steps = @($script:FailedOptionalSteps)
            created_at = $createdAt
        }

        $report | ConvertTo-Json -Depth 4 | Set-Content -Path $script:InstallReportPath -Encoding UTF8
        Write-Success "Install report written to $script:InstallReportPath"
    } catch {
        $message = if ($_.Exception -and $_.Exception.Message) { $_.Exception.Message } else { "unknown error" }
        Write-Warning-Msg "Install report could not be written: $message"
    }
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
    Write-Host "Install report: $script:InstallReportPath"
    Write-Host ""
    Write-Host "Double-click Launch Annolid.cmd or Update Annolid.cmd in the install folder."
    Write-Host "To launch manually:"
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

    if (!$NoInteractive -and (Prompt-YesNo "Launch Annolid now?" $true)) {
        Write-Step "Launching Annolid..."

        # We need to activate environment and run annolid
        . $script:ActivateCmd
        $env:KMP_DUPLICATE_LIB_OK = "TRUE"
        annolid
    }
}

# =============================================================================
# Main
# =============================================================================
if ($Help) { Show-Help }
Initialize-OfflineConfig

Write-Header
Test-Git
Test-Uv
$pythonCmd = Test-Python
Test-FFmpeg
Test-GPU
Get-InteractiveConfig
Clone-Repo
New-Venv
Install-Annolid
Repair-OnnxRuntime
Test-Installation
Write-InstallReport
$shortcutArgs = @("scripts/create_desktop_shortcuts.py", "--install-dir", $script:InstallDir,
    "--venv-path", $script:VenvPath, "--profile", $Profile, "--extras", ($script:InstallExtras -join ","))
if ($NoGpu) { $shortcutArgs += "--no-gpu" }
& python @shortcutArgs
if ($LASTEXITCODE -ne 0) { Write-Warning-Msg "Shortcuts could not be created. Use the activation command below." }
Write-Summary
