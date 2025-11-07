#!/bin/bash
set -e

echo "=========================================="
echo "Starting FSFM-CVPR25 Container"
echo "=========================================="

# Run setup script on first launch
if [ ! -f "/workspace/.setup_complete" ]; then
    echo "First time setup detected. Running initialization..."
    bash /workspace/scripts/setup.sh
else
    echo "Setup already completed. Skipping initialization..."
fi

# Print useful information
echo ""
echo "=========================================="
echo "Container is ready!"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo "  - Fine-tune on DfD:  cd fsvfm/finetune/cross_dataset_DFD_and_DiFF && bash scripts_DFD/run_DfD-ViT-B.sh"
echo "  - Fine-tune on DiFF: cd fsvfm/finetune/cross_dataset_DFD_and_DiFF && bash scripts_DiFF/run_DiFF-ViT-B.sh"
echo "  - Fine-tune on FAS:  cd fsvfm/finetune/cross_domain_FAS && bash scripts/run_base.sh"
echo ""
echo "Environment Variables:"
echo "  - MODEL_SIZE: ${MODEL_SIZE:-ViT-B (default)}"
echo "  - DATASET_TYPE: ${DATASET_TYPE:-minimal (default)}"
echo "  - AUTO_DOWNLOAD_DATASETS: ${AUTO_DOWNLOAD_DATASETS:-false (default)}"
echo ""
echo "To download additional models:"
echo "  bash scripts/download_models.sh"
echo ""
echo "To download datasets:"
echo "  export DATASET_TYPE=dfd  # or diff, fas, all"
echo "  bash scripts/download_datasets.sh"
echo ""
echo "=========================================="

# Execute the main command
exec "$@"

