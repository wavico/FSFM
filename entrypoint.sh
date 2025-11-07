#!/bin/bash
set -e

echo "=========================================="
echo "FSFM-CVPR25 Environment Setup"
echo "=========================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. FACER 툴킷 설치
echo ""
echo "1. Checking FACER toolkit..."
FACER_PATH="/workspace/datasets/pretrain/preprocess/tools/facer"
if [ ! -d "$FACER_PATH" ]; then
    print_warning "FACER toolkit not found. Cloning from GitHub..."
    mkdir -p /workspace/datasets/pretrain/preprocess/tools
    cd /workspace/datasets/pretrain/preprocess/tools
    git clone https://github.com/FacePerceiver/facer.git
    if [ $? -eq 0 ]; then
        print_status "FACER toolkit installed successfully"
    else
        print_error "Failed to clone FACER toolkit"
    fi
else
    print_status "FACER toolkit already exists"
fi

# 2. 사전학습된 모델 가중치 다운로드
echo ""
echo "2. Checking pre-trained model weights..."
CHECKPOINT_DIR="/workspace/fsvfm/pretrain/checkpoint/pretrained_models"
if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
    print_warning "Pre-trained models not found. Downloading..."
    
    # huggingface_hub 패키지가 설치되어 있는지 확인
    python3 -c "import huggingface_hub" 2>/dev/null
    if [ $? -eq 0 ]; then
        # 환경 변수로 다운로드할 모델 선택 (기본값: ViT-B)
        MODEL_TO_DOWNLOAD=${DOWNLOAD_MODEL:-"vit-b"}
        
        print_warning "Downloading ${MODEL_TO_DOWNLOAD} model(s)..."
        python3 /usr/local/bin/setup_download_models.py \
            --model "${MODEL_TO_DOWNLOAD}" \
            --checkpoint-dir "/workspace/fsvfm/pretrain/checkpoint/"
        
        if [ $? -eq 0 ]; then
            print_status "Pre-trained models downloaded successfully"
        else
            print_error "Failed to download pre-trained models"
        fi
    else
        print_error "huggingface_hub not installed. Skipping model download."
        print_warning "Please run: pip install huggingface_hub"
    fi
else
    print_status "Pre-trained models already exist"
fi

# 3. 데이터셋 확인
echo ""
echo "3. Checking datasets..."
DATASET_DIR="/workspace/datasets/finetune_datasets"
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
    print_warning "Fine-tuning datasets not found"
    print_warning "Please download datasets from:"
    echo "   🤗 https://huggingface.co/datasets/Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM/tree/main/finetune_datasets"
    echo ""
    echo "   Or set AUTO_DOWNLOAD_DATASET=true environment variable to download automatically"
    
    # 자동 다운로드 옵션 (환경 변수로 제어)
    if [ "$AUTO_DOWNLOAD_DATASET" = "true" ]; then
        print_warning "AUTO_DOWNLOAD_DATASET is enabled. Downloading sample datasets..."
        # 여기에 huggingface-cli 또는 wget을 사용한 다운로드 로직 추가 가능
        # 예: huggingface-cli download Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM --repo-type dataset --local-dir /workspace/datasets/finetune_datasets
    fi
else
    print_status "Fine-tuning datasets found"
fi

# 4. 필요한 디렉토리 생성
echo ""
echo "4. Creating necessary directories..."
mkdir -p /workspace/data
mkdir -p /workspace/outputs
mkdir -p /workspace/logs
mkdir -p /workspace/checkpoints
mkdir -p /workspace/datasets/pretrain_datasets
print_status "Directories created"

# 5. Python 환경 확인
echo ""
echo "5. Checking Python environment..."
python3 --version
print_status "Python environment ready"

# 6. PyTorch 및 CUDA 확인
echo ""
echo "6. Checking PyTorch and CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status "PyTorch and CUDA ready"
else
    print_error "PyTorch not properly installed"
fi

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  - Training: cd fsvfm/pretrain && bash scripts/pretrain_FSVFM_ViT-B.sh"
echo "  - Fine-tuning: cd fsvfm/finetune/cross_dataset_DFD_and_DiFF && bash scripts_DFD/run_DfD-ViT-B.sh"
echo "  - Testing: cd fsvfm/finetune/cross_dataset_DFD_and_DiFF && bash scripts_DFD/run_DfD-ViT-B.sh"
echo ""

# Execute the main command
exec "$@"

