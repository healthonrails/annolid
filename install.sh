#!/usr/bin/env bash
# Annolid One-Line Installation Script for macOS and Linux
#
# Usage (one-liner):
#   curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
#
# Usage (with options):
#   curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash -s -- [OPTIONS]
#
# Options:
#   --install-dir DIR   Directory to install annolid (default: ./annolid)
#   --venv-dir DIR      Directory for virtual environment (default: .venv inside install-dir)
#   --profile PROFILE   Install profile: minimal,gui,workstation,full (default: gui)
#   --extras EXTRAS     Comma-separated optional extras: audio,ai_chat,training,ml,tracking,yolo,cutie,realtime,large_image,remote_video,sam3,image_editing,text_to_speech,qwen3_embedding,annolid_bot,bot,memory,all (GUI extras are always included)
#   --no-gpu            Skip GPU/CUDA detection
#   --use-conda         Use conda instead of venv (requires conda/mamba)
#   --no-interactive    Skip all prompts and use defaults
#   --help              Show this help message

set -e

# =============================================================================
# Configuration
# =============================================================================
INSTALL_DIR=""
VENV_DIR=".venv"
EXTRAS=""
PROFILE="gui"
NO_GPU=false
USE_CONDA=false
NO_INTERACTIVE=false
PYTHON_MIN_VERSION="3.10"
ANNOLID_REPO="https://github.com/healthonrails/annolid.git"
USE_UV_PYTHON=false
USE_UV=false
UV_CMD="uv"
HAS_NVIDIA_GPU=false
HAS_CUDA12=false
CUDA_VERSION=""
INSTALL_REPORT_PATH=""
SAM_HQ_STATUS="not_requested"
FAILED_OPTIONAL_STEPS=""
PYTORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"
ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}${BOLD}              Annolid Installation Script                     ${NC}${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}     Annotate, Segment, and Track Anything You Need          ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✖ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

record_optional_failure() {
    local step="$1"
    if [[ -z "$FAILED_OPTIONAL_STEPS" ]]; then
        FAILED_OPTIONAL_STEPS="$step"
    elif [[ ",$FAILED_OPTIONAL_STEPS," != *",$step,"* ]]; then
        FAILED_OPTIONAL_STEPS="${FAILED_OPTIONAL_STEPS},${step}"
    fi
}

print_usage() {
    cat <<'EOF'
Annolid one-line installer for macOS and Linux

Usage:
  curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
  curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash -s -- [OPTIONS]

Options:
  --install-dir DIR   Directory to install annolid (default: ./annolid)
  --venv-dir DIR      Directory for virtual environment (default: .venv inside install-dir)
  --profile PROFILE   Install profile: minimal,gui,workstation,full (default: gui)
  --extras EXTRAS     Comma-separated optional extras: audio,ai_chat,training,ml,tracking,yolo,cutie,realtime,large_image,remote_video,sam3,image_editing,text_to_speech,qwen3_embedding,annolid_bot,bot,memory,all
  --no-gpu            Skip GPU/CUDA detection and install CPU ONNX Runtime
  --use-conda         Use conda instead of venv (requires conda/mamba)
  --no-interactive    Skip prompts and use defaults
  --help              Show this help message
EOF
}

show_help() {
    print_usage
    exit 0
}

require_arg_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        print_error "$option requires a value."
        print_usage
        exit 1
    fi
}

# Wrapper for read that works when script is piped via curl | bash
read_input() {
    local prompt="$1"
    local variable="$2"
    local default="$3"

    # Try to read specific variable value
    if [[ -t 0 ]]; then
        # Standard stdin is a TTY
        read -p "$prompt" -r "$variable"
    else
        # Script is being piped, try /dev/tty
        if [[ -e /dev/tty ]]; then
            read -p "$prompt" -r "$variable" < /dev/tty
        else
            # No TTY available (CI/CD?), use default
            eval "$variable='$default'"
        fi
    fi
}

prompt_yes_no() {
    local prompt="$1"
    local default="$2"

    if [[ "$NO_INTERACTIVE" == true ]]; then
        [[ "$default" == "y" ]] && return 0 || return 1
    fi

    # Prepare prompt string
    local full_prompt
    if [[ "$default" == "y" ]]; then
        full_prompt="$prompt [Y/n] "
    else
        full_prompt="$prompt [y/N] "
    fi

    local response
    if [[ -t 0 ]]; then
        read -p "$full_prompt" -n 1 -r response
    else
        if [[ -e /dev/tty ]]; then
            read -p "$full_prompt" -n 1 -r response < /dev/tty
        else
            response="$default"
        fi
    fi
    echo

    if [[ -z "$response" ]]; then
        response="$default"
    fi

    if [[ "$default" == "y" ]]; then
        [[ ! $response =~ ^[Nn]$ ]]
    else
        [[ $response =~ ^[Yy]$ ]]
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            require_arg_value "$1" "${2:-}"
            INSTALL_DIR="$2"
            shift 2
            ;;
        --venv-dir)
            require_arg_value "$1" "${2:-}"
            VENV_DIR="$2"
            shift 2
            ;;
        --profile)
            require_arg_value "$1" "${2:-}"
            PROFILE="$2"
            shift 2
            ;;
        --extras)
            require_arg_value "$1" "${2:-}"
            EXTRAS="$2"
            shift 2
            ;;
        --no-gpu)
            NO_GPU=true
            shift
            ;;
        --use-conda)
            USE_CONDA=true
            shift
            ;;
        --no-interactive)
            NO_INTERACTIVE=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# =============================================================================
# Detect Operating System
# =============================================================================
detect_os() {
    print_step "Detecting operating system..."

    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux*)
            OS_TYPE="linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO="$ID"
            else
                DISTRO="unknown"
            fi
            ;;
        Darwin*)
            OS_TYPE="macos"
            DISTRO="macos"
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac

    echo "  OS: $OS_TYPE ($DISTRO)"
    echo "  Architecture: $ARCH"

    if [[ "$OS_TYPE" == "macos" && "$ARCH" == "arm64" ]]; then
        IS_APPLE_SILICON=true
        echo "  Apple Silicon: Yes"
    else
        IS_APPLE_SILICON=false
    fi
}

# =============================================================================
# Check Git
# =============================================================================
check_git() {
    print_step "Checking Git installation..."

    if ! command -v git &> /dev/null; then
        print_warning "Git not found. Attempting automatic installation..."

        INSTALLED=false
        if [[ "$OS_TYPE" == "macos" ]]; then
             if command -v brew &> /dev/null; then
                brew install git
                INSTALLED=true
             else
                print_error "Homebrew not found. Please install git manually."
                echo "  xcode-select --install"
                exit 1
             fi
        elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
            sudo apt update && sudo apt install -y git
            INSTALLED=true
        elif [[ "$DISTRO" == "fedora" ]]; then
            sudo dnf install -y git
            INSTALLED=true
        fi

        if ! command -v git &> /dev/null; then
             print_error "Git installation failed. Please install manually."
             exit 1
        fi

        print_success "Git installed successfully"
    else
        print_success "Git found"
    fi
}

# =============================================================================
# Check Python Version
# =============================================================================
check_python() {
    print_step "Checking Python installation..."

    PREFERRED_VERSIONS=("3.11" "3.12" "3.13" "3.14" "3.10")
    PYTHON_CMD=""
    PYTHON_VERSION=""

    for ver in "${PREFERRED_VERSIONS[@]}"; do
        for cmd in "python$ver" "python${ver%.*}"; do
            if command -v "$cmd" &> /dev/null; then
                actual_ver=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
                if [[ "$actual_ver" == "$ver" ]]; then
                    PYTHON_CMD="$cmd"
                    PYTHON_VERSION="$actual_ver"
                    echo "  Found preferred: $PYTHON_CMD (version $PYTHON_VERSION)"
                    break 2
                fi
            fi
        done
    done

    if [[ -z "$PYTHON_CMD" ]]; then
        for cmd in "python3" "python"; do
            if command -v "$cmd" &> /dev/null; then
                PYTHON_VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
                PYTHON_MAJOR=$($cmd -c 'import sys; print(sys.version_info.major)')
                PYTHON_MINOR=$($cmd -c 'import sys; print(sys.version_info.minor)')

                if [[ "$PYTHON_MAJOR" -ge 3 && "$PYTHON_MINOR" -ge 10 ]]; then
                    PYTHON_CMD="$cmd"
                    echo "  Found: $PYTHON_CMD (version $PYTHON_VERSION)"
                    break
                fi
            fi
        done
    fi

    if [[ -z "$PYTHON_CMD" ]]; then
        print_error "Python $PYTHON_MIN_VERSION or higher not found."
        echo ""
        echo "Installation instructions:"
        if [[ "$OS_TYPE" == "macos" ]]; then
            echo "  brew install python@3.11"
        elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
            echo "  sudo apt install python3.11 python3.11-venv python3-pip"
        elif [[ "$DISTRO" == "fedora" ]]; then
            echo "  sudo dnf install python3.11"
        fi
        exit 1
    fi

    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
        print_error "Python $PYTHON_MIN_VERSION or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    if [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -eq 14 ]]; then
        print_info "Python 3.14 detected. Annolid supports the default GUI/core install on Python 3.14."
        print_info "Remote network video decoding still needs the optional remote_video extra and native FFmpeg development libraries."
    elif [[ "$PYTHON_MAJOR" -gt 3 || ( "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -ge 15 ) ]]; then
        print_warning "Python $PYTHON_VERSION detected. Annolid is not yet validated on Python 3.15+."
        echo ""
        echo "  Recommended: Install Python 3.11 or 3.12 for best compatibility."
        if [[ "$OS_TYPE" == "macos" ]]; then
            echo "    brew install python@3.11"
        elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
            echo "    sudo apt install python3.11 python3.11-venv"
        fi
        echo ""

        if [[ "$USE_UV" == true ]]; then
            echo "  uv can download a compatible Python version automatically."
            if prompt_yes_no "  Use uv to install with Python 3.11?" "y"; then
                PYTHON_VERSION="3.11"
                USE_UV_PYTHON=true
                print_success "Will use uv to create venv with Python 3.11"
                return
            fi
        fi

        if ! prompt_yes_no "  Continue with Python $PYTHON_VERSION anyway? (may fail)" "n"; then
            exit 1
        fi
    fi

    print_success "Python version OK"
}

# =============================================================================
# Check FFmpeg
# =============================================================================
check_ffmpeg() {
    print_step "Checking FFmpeg installation..."

    if command -v ffmpeg &> /dev/null; then
        FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1)
        echo "  $FFMPEG_VERSION"
        print_success "FFmpeg found"
    else
        print_warning "FFmpeg not found. Attempting automatic installation..."

        INSTALLED=false
        if [[ "$OS_TYPE" == "macos" ]]; then
             if command -v brew &> /dev/null; then
                brew install ffmpeg
                INSTALLED=true
             else
                print_warning "Homebrew not found. Cannot auto-install FFmpeg."
             fi
        elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
            sudo apt update && sudo apt install -y ffmpeg
            INSTALLED=true
        elif [[ "$DISTRO" == "fedora" ]]; then
            sudo dnf install -y ffmpeg
            INSTALLED=true
        fi

        if command -v ffmpeg &> /dev/null; then
             print_success "FFmpeg installed successfully"
        else
             print_warning "Automatic installation failed or not supported."
             print_warning "Video processing may be limited."

             # Fallback instructions
             echo "  To install manually:"
             if [[ "$OS_TYPE" == "macos" ]]; then
                 echo "    brew install ffmpeg"
             elif [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
                 echo "    sudo apt install ffmpeg"
             fi
             echo ""

             if ! prompt_yes_no "  Continue without FFmpeg?" "y"; then
                exit 1
             fi
        fi
    fi
}

# =============================================================================
# Check GPU
# =============================================================================
check_gpu() {
    if [[ "$NO_GPU" == true ]]; then
        print_info "GPU detection skipped (--no-gpu)."
        HAS_NVIDIA_GPU=false
        HAS_CUDA12=false
        CUDA_VERSION=""
        return
    fi

    print_step "Checking for NVIDIA GPU..."

    if command -v nvidia-smi &> /dev/null; then
        GPU_LINE=$(nvidia-smi -L 2>/dev/null | head -1 || true)
        if [[ -n "$GPU_LINE" ]]; then
            HAS_NVIDIA_GPU=true
            print_success "NVIDIA GPU detected"
            echo "  $GPU_LINE"

            CUDA_VERSION=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version:[[:space:]]*([0-9]+\.[0-9]+).*/\1/p' | head -1 || true)
            CUDA_MAJOR="${CUDA_VERSION%%.*}"
            if [[ "$CUDA_MAJOR" =~ ^[0-9]+$ && "$CUDA_MAJOR" -ge 12 ]]; then
                HAS_CUDA12=true
                print_success "CUDA $CUDA_VERSION detected; ONNX Runtime GPU will be installed."
                if [[ "$CUDA_MAJOR" -gt 12 ]]; then
                    print_info "Using stable CUDA 12 ONNX Runtime wheels; NVIDIA drivers are backward compatible with CUDA 12 runtimes."
                fi
            elif [[ -n "$CUDA_VERSION" ]]; then
                HAS_CUDA12=false
                print_warning "CUDA $CUDA_VERSION detected. ONNX Runtime GPU requires a CUDA 12-compatible NVIDIA driver."
                print_info "Annolid will install CPU ONNX Runtime for compatibility."
            else
                HAS_CUDA12=false
                print_warning "Unable to parse CUDA runtime version from nvidia-smi output."
                print_info "Annolid will install CPU ONNX Runtime for compatibility."
            fi
            return
        fi
    fi

    HAS_NVIDIA_GPU=false
    HAS_CUDA12=false
    CUDA_VERSION=""
    print_info "No NVIDIA GPU detected. Using default PyTorch build."
    if [[ "$OS_TYPE" == "macos" ]]; then
        print_info "macOS ONNX acceleration uses CoreML provider when available; there is no CUDA ONNX Runtime wheel for macOS."
    fi
}

# =============================================================================
# Install ONNX Runtime
# =============================================================================
install_onnx_runtime() {
    print_step "Installing ONNX Runtime..."

    if [[ "$HAS_NVIDIA_GPU" == true && "$HAS_CUDA12" == true && "$NO_GPU" == false ]]; then
        echo "  Installing onnxruntime-gpu for CUDA 12.x..."
        uninstall_packages onnxruntime-gpu
        if $PIP_CMD install --upgrade --force-reinstall onnxruntime-gpu --extra-index-url "$ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL"; then
            print_success "onnxruntime-gpu installed"
            return
        fi

        print_warning "onnxruntime-gpu install failed. Falling back to CPU onnxruntime."
    fi

    uninstall_packages onnxruntime-gpu
    $PIP_CMD install --upgrade --force-reinstall onnxruntime
    print_success "onnxruntime installed"
}

uninstall_packages() {
    if [[ "$USE_UV" == true && "$USE_CONDA" == false ]]; then
        "$UV_CMD" pip uninstall "$@" >/dev/null 2>&1 || true
    else
        pip uninstall -y "$@" >/dev/null 2>&1 || true
    fi
}

# =============================================================================
# Check uv
# =============================================================================
check_uv() {
    if [[ "$USE_CONDA" == true ]]; then
        return
    fi

    print_step "Checking for uv (fast package installer)..."

    if command -v uv &> /dev/null; then
        UV_CMD="$(command -v uv)"
        UV_VERSION=$("$UV_CMD" --version)
        echo "  Found: $UV_VERSION"
        USE_UV=true
        return
    fi

    print_warning "uv not found. Installing via official installer..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        UV_CANDIDATES=(
            "$HOME/.local/bin/uv"
            "$HOME/.cargo/bin/uv"
            "/opt/homebrew/bin/uv"
            "/usr/local/bin/uv"
        )
        for candidate in "${UV_CANDIDATES[@]}"; do
            if [[ -x "$candidate" ]]; then
                export PATH="$(dirname "$candidate"):$PATH"
                UV_CMD="$candidate"
                break
            fi
        done

        if command -v uv &> /dev/null || [[ -x "$UV_CMD" ]]; then
            if command -v uv &> /dev/null; then
                UV_CMD="$(command -v uv)"
            fi
            UV_VERSION=$("$UV_CMD" --version)
            print_success "uv installed: $UV_VERSION"
            USE_UV=true
        else
            print_error "uv installation completed but command not found in PATH."
            echo "  Please ensure your shell profile sources ~/.local/bin or the installer path."
            exit 1
        fi
    else
        print_error "Failed to install uv automatically."
        echo "  Please install manually:"
        echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# =============================================================================
# Interactive Configuration
# =============================================================================
interactive_config() {
    if [[ "$NO_INTERACTIVE" == true ]]; then
        return
    fi

    print_step "Configuration options..."
    echo ""

    # Installation directory
    if [[ -z "$INSTALL_DIR" ]]; then
        echo -e "  ${CYAN}Where would you like to install Annolid?${NC}"
        read_input "  Install directory [./annolid]: " INSTALL_DIR "./annolid"
        INSTALL_DIR="${INSTALL_DIR:-./annolid}"
    fi

    # Optional extras
    echo ""
    echo -e "  ${CYAN}Optional features (select with y/n):${NC}"

    SELECTED_EXTRAS=()

    if prompt_yes_no "    Include SAM3 (advanced segmentation)?" "n"; then
        SELECTED_EXTRAS+=("sam3")
    fi

    if prompt_yes_no "    Include YOLO/YOLOE workflows and training dashboards?" "n"; then
        SELECTED_EXTRAS+=("yolo" "training")
    fi

    if prompt_yes_no "    Include hosted AI chat providers?" "n"; then
        SELECTED_EXTRAS+=("ai_chat")
    fi

    if prompt_yes_no "    Include image editing (diffusion models)?" "n"; then
        SELECTED_EXTRAS+=("image_editing")
    fi

    if prompt_yes_no "    Include audio helpers?" "n"; then
        SELECTED_EXTRAS+=("audio")
    fi

    if prompt_yes_no "    Include text-to-speech?" "n"; then
        SELECTED_EXTRAS+=("text_to_speech")
    fi

    if prompt_yes_no "    Include realtime hardware/streaming integrations?" "n"; then
        SELECTED_EXTRAS+=("realtime")
    fi

    if prompt_yes_no "    Include Annolid Bot integrations (WhatsApp + Google Calendar + MCP)?" "n"; then
        SELECTED_EXTRAS+=("annolid_bot")
    fi

    if [[ ${#SELECTED_EXTRAS[@]} -gt 0 ]]; then
        EXTRAS=$(IFS=,; echo "${SELECTED_EXTRAS[*]}")
        print_info "Selected extras: $EXTRAS"
    fi

    echo ""
}

profile_extras() {
    case "$PROFILE" in
        minimal)
            echo ""
            ;;
        gui)
            echo "ml,tracking,cutie"
            ;;
        workstation)
            echo "tracking,sam3,training"
            ;;
        full)
            echo "all"
            ;;
        *)
            print_error "Unknown install profile: $PROFILE"
            echo "  Supported profiles: minimal, gui, workstation, full"
            exit 1
            ;;
    esac
}

merge_extras() {
    local profile_values="$1"
    local requested_values="$2"
    local merged=()
    local value
    local existing

    IFS=',' read -ra values <<< "${profile_values},${requested_values}"
    for value in "${values[@]}"; do
        value="$(echo "$value" | tr -d '[:space:]')"
        [[ -z "$value" ]] && continue
        existing=",${merged[*]},"
        existing="${existing// /,}"
        if [[ "$existing" != *",${value},"* ]]; then
            merged+=("$value")
        fi
    done

    if [[ ${#merged[@]} -gt 0 ]]; then
        (IFS=,; echo "${merged[*]}")
    else
        echo ""
    fi
}

# =============================================================================
# Clone Repository
# =============================================================================
clone_repo() {
    print_step "Cloning Annolid repository..."

    # Set default install directory
    if [[ -z "$INSTALL_DIR" ]]; then
        INSTALL_DIR="./annolid"
    fi

    # Expand path
    INSTALL_DIR=$(eval echo "$INSTALL_DIR")

    if [[ -d "$INSTALL_DIR" ]]; then
        if [[ -d "$INSTALL_DIR/.git" ]]; then
            print_info "Annolid directory exists. Updating..."
            cd "$INSTALL_DIR"
            git pull --recurse-submodules || true
        else
            print_warning "Directory $INSTALL_DIR exists but is not a git repository."
            if prompt_yes_no "  Remove and re-clone?" "n"; then
                rm -rf "$INSTALL_DIR"
                git clone --recurse-submodules "$ANNOLID_REPO" "$INSTALL_DIR"
                cd "$INSTALL_DIR"
            else
                cd "$INSTALL_DIR"
            fi
        fi
    else
        git clone --recurse-submodules "$ANNOLID_REPO" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    INSTALL_DIR=$(pwd)
    print_success "Repository ready at $INSTALL_DIR"
}

# =============================================================================
# Create Virtual Environment
# =============================================================================
create_venv() {
    print_step "Creating virtual environment..."

    VENV_PATH="$INSTALL_DIR/$VENV_DIR"

    if [[ "$USE_CONDA" == true ]]; then
        if command -v mamba &> /dev/null; then
            CONDA_CMD="mamba"
        elif command -v conda &> /dev/null; then
            CONDA_CMD="conda"
        else
            print_error "Neither conda nor mamba found."
            exit 1
        fi

        echo "  Using $CONDA_CMD..."
        $CONDA_CMD create -y -n annolid-env python=3.11
        ACTIVATE_CMD="conda activate annolid-env"

    elif [[ "$USE_UV" == true ]]; then
        if [[ "$USE_UV_PYTHON" == true ]]; then
            "$UV_CMD" venv "$VENV_PATH" --python 3.11
        else
            "$UV_CMD" venv "$VENV_PATH" --python "$PYTHON_VERSION"
        fi
        ACTIVATE_CMD="source $VENV_PATH/bin/activate"

    else
        $PYTHON_CMD -m venv "$VENV_PATH"
        ACTIVATE_CMD="source $VENV_PATH/bin/activate"
    fi

    print_success "Virtual environment created"
}

# =============================================================================
# Install Annolid
# =============================================================================
install_annolid() {
    print_step "Installing Annolid..."

    if [[ "$USE_CONDA" == true ]]; then
        eval "$(conda shell.bash hook)"
        conda activate annolid-env
    else
        source "$VENV_PATH/bin/activate"
    fi

    if [[ "$USE_UV" == true && "$USE_CONDA" == false ]]; then
        PIP_CMD="$UV_CMD pip"
    else
        PIP_CMD="pip"
    fi

    echo "  Upgrading pip..."
    $PIP_CMD install --upgrade pip

    if [[ "$HAS_NVIDIA_GPU" == true ]]; then
        echo "  Installing CUDA-enabled PyTorch..."
        if $PIP_CMD install --index-url "$PYTORCH_CUDA_INDEX_URL" torch torchvision; then
            print_success "CUDA-enabled PyTorch installed"
        else
            print_warning "CUDA-enabled PyTorch install failed. Falling back to default PyTorch build."
            $PIP_CMD install torch torchvision
        fi
    fi

    # GUI dependencies are always installed so `annolid` can launch immediately.
    PROFILE_EXTRAS="$(profile_extras)"
    NORMALIZED_EXTRAS="$(merge_extras "$PROFILE_EXTRAS" "$EXTRAS")"
    INSTALL_EXTRAS="$(merge_extras "gui" "$NORMALIZED_EXTRAS")"
    INSTALL_TARGET="-e .[${INSTALL_EXTRAS}]"
    echo "  Profile: $PROFILE"
    echo "  Including extras: $INSTALL_EXTRAS"

    echo "  Installing annolid (this may take a few minutes)..."
    $PIP_CMD install $INSTALL_TARGET

    if extra_selected "sam3" || extra_selected "sam" || extra_selected "segment_anything" || extra_selected "all"; then
        echo "  Installing SAM-HQ..."
        SAM_HQ_STATUS="requested"
        if $PIP_CMD install "segment-anything @ git+https://github.com/SysCV/sam-hq.git"; then
            SAM_HQ_STATUS="installed"
        else
            SAM_HQ_STATUS="failed"
            record_optional_failure "sam_hq"
            print_warning "SAM-HQ installation failed. Segment Anything may have limited functionality."
        fi
    else
        SAM_HQ_STATUS="skipped"
        print_info "Skipping optional SAM-HQ install. Add --extras sam3 when needed."
    fi

    install_onnx_runtime

    print_success "Annolid installed"
}

extra_selected() {
    local needle="$1"
    local selected=",${INSTALL_EXTRAS},"
    [[ "$selected" == *",${needle},"* ]]
}

# =============================================================================
# Validate Installation
# =============================================================================
validate_installation() {
    print_step "Validating installation..."

    if [[ "$USE_CONDA" == true ]]; then
        eval "$(conda shell.bash hook)"
        conda activate annolid-env
    else
        source "$VENV_PATH/bin/activate"
    fi

    if command -v annolid &> /dev/null; then
        print_success "annolid command available"
    else
        print_error "annolid command not found"
        exit 1
    fi

    echo "  Checking imports..."
    python -c "import annolid; print(f'  Annolid version: {annolid.__version__}')" 2>/dev/null || \
        python -c "import annolid; print('  Annolid imported successfully')"

    echo "  Checking Qt bindings..."
    if ! python -c "from qtpy import QtCore; print('  Qt binding import OK')"; then
        print_error "Qt bindings are missing. Reinstall with GUI extras:"
        echo "  pip install -e \".[gui]\""
        exit 1
    fi

    echo "  Checking PyTorch availability..."
    if python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"; then
        if [[ "$NO_GPU" == false ]]; then
            python -c "import torch; cuda = torch.cuda.is_available(); print(f'  CUDA available: {cuda}')"
            if [[ "$IS_APPLE_SILICON" == true ]]; then
                python -c "import torch; mps = torch.backends.mps.is_available(); print(f'  MPS (Apple Silicon) available: {mps}')"
            fi
        fi
    else
        print_info "PyTorch not installed; optional ML/tracking features will install or report their requirements on first use."
    fi

    echo "  Checking ONNX Runtime providers..."
    python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print(f'  ONNX providers: {providers}')"

    if [[ "$HAS_NVIDIA_GPU" == true && "$HAS_CUDA12" == true && "$NO_GPU" == false ]]; then
        echo "  Verifying CUDAExecutionProvider..."
        if python -c "import onnxruntime as ort; providers = set(ort.get_available_providers()); assert 'CUDAExecutionProvider' in providers, f'Missing CUDAExecutionProvider. Providers={providers}'"; then
            print_success "CUDAExecutionProvider available"
        else
            print_error "CUDAExecutionProvider is missing despite CUDA 12.x detection."
            echo "  Repair command:"
            echo "    $PIP_CMD install --upgrade --force-reinstall onnxruntime-gpu --extra-index-url $ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL"
            exit 1
        fi
    elif [[ "$IS_APPLE_SILICON" == true && "$NO_GPU" == false ]]; then
        if python -c "import onnxruntime as ort; raise SystemExit(0 if 'CoreMLExecutionProvider' in set(ort.get_available_providers()) else 1)"; then
            print_success "CoreMLExecutionProvider available"
        else
            print_warning "CoreMLExecutionProvider is not available; ONNX models will use CPUExecutionProvider."
        fi
    fi

    print_success "Installation validated"
}

# =============================================================================
# Install Report
# =============================================================================
write_install_report() {
    print_step "Writing machine-readable install report..."

    if [[ "$USE_CONDA" == true ]]; then
        eval "$(conda shell.bash hook)"
        conda activate annolid-env
    else
        source "$VENV_PATH/bin/activate"
    fi

    if [[ "$USE_CONDA" == true ]]; then
        PACKAGE_MANAGER="conda"
    elif [[ "$USE_UV" == true ]]; then
        PACKAGE_MANAGER="uv"
    else
        PACKAGE_MANAGER="pip"
    fi

    INSTALL_REPORT_PATH="$INSTALL_DIR/annolid-install-report.json"
    REPORT_PROFILE="$PROFILE" \
    REPORT_EXTRAS="$INSTALL_EXTRAS" \
    REPORT_INSTALL_DIR="$INSTALL_DIR" \
    REPORT_VENV_PATH="$VENV_PATH" \
    REPORT_OS_TYPE="$OS_TYPE" \
    REPORT_DISTRO="$DISTRO" \
    REPORT_ARCH="$ARCH" \
    REPORT_PACKAGE_MANAGER="$PACKAGE_MANAGER" \
    REPORT_GPU_REQUESTED="$([[ "$NO_GPU" == true ]] && echo "false" || echo "true")" \
    REPORT_HAS_NVIDIA_GPU="$HAS_NVIDIA_GPU" \
    REPORT_HAS_CUDA12="$HAS_CUDA12" \
    REPORT_CUDA_VERSION="$CUDA_VERSION" \
    REPORT_SAM_HQ_STATUS="$SAM_HQ_STATUS" \
    REPORT_FAILED_OPTIONAL_STEPS="$FAILED_OPTIONAL_STEPS" \
    REPORT_PATH="$INSTALL_REPORT_PATH" \
    python - <<'PY'
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


providers: list[str] = []
try:
    import onnxruntime as ort

    providers = list(ort.get_available_providers())
except Exception:
    providers = []

failed_steps = [
    item.strip()
    for item in os.environ.get("REPORT_FAILED_OPTIONAL_STEPS", "").split(",")
    if item.strip()
]
report = {
    "profile": os.environ["REPORT_PROFILE"],
    "extras": [
        item.strip()
        for item in os.environ.get("REPORT_EXTRAS", "").split(",")
        if item.strip()
    ],
    "install_dir": os.environ["REPORT_INSTALL_DIR"],
    "venv_path": os.environ["REPORT_VENV_PATH"],
    "python_version": platform.python_version(),
    "os": os.environ["REPORT_OS_TYPE"],
    "distro": os.environ["REPORT_DISTRO"],
    "arch": os.environ["REPORT_ARCH"],
    "package_manager": os.environ["REPORT_PACKAGE_MANAGER"],
    "gpu_requested": as_bool(os.environ["REPORT_GPU_REQUESTED"]),
    "has_nvidia_gpu": as_bool(os.environ["REPORT_HAS_NVIDIA_GPU"]),
    "has_cuda12_driver": as_bool(os.environ["REPORT_HAS_CUDA12"]),
    "cuda_version": os.environ.get("REPORT_CUDA_VERSION") or None,
    "onnx_providers": providers,
    "sam_hq_status": os.environ["REPORT_SAM_HQ_STATUS"],
    "failed_optional_steps": failed_steps,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
path = Path(os.environ["REPORT_PATH"])
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    print_success "Install report written to $INSTALL_REPORT_PATH"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Installation Complete! 🎉                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Annolid has been installed to: $INSTALL_DIR"
    echo "Install report: $INSTALL_REPORT_PATH"
    echo ""
    echo "To get started:"
    echo ""
    echo "  1. Navigate to the directory:"
    echo -e "     ${BLUE}cd $INSTALL_DIR${NC}"
    echo ""
    echo "  2. Activate the environment:"
    echo -e "     ${BLUE}$ACTIVATE_CMD${NC}"
    echo ""
    echo "  3. Launch Annolid:"
    echo -e "     ${BLUE}annolid${NC}"
    echo ""
    echo -e "${CYAN}Quick start (copy/paste):${NC}"
    echo -e "  ${BOLD}cd $INSTALL_DIR && $ACTIVATE_CMD && annolid${NC}"
    echo ""
    echo "Documentation: https://annolid.com"
    echo "Issues: https://github.com/healthonrails/annolid/issues"
    echo ""

    if prompt_yes_no "Launch Annolid now?" "y"; then
        print_step "Launching Annolid..."

        # We need to activate environment and run annolid
        if [[ "$USE_CONDA" == true ]]; then
             eval "$(conda shell.bash hook)"
             conda activate annolid-env
        else
             source "$VENV_PATH/bin/activate"
        fi

        annolid
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    print_header
    detect_os
    check_git
    check_uv
    check_python
    check_ffmpeg
    check_gpu
    interactive_config
    clone_repo
    create_venv
    install_annolid
    validate_installation
    write_install_report
    print_summary
}

main "$@"
