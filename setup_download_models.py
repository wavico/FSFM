#!/usr/bin/env python3
"""
자동 모델 가중치 다운로드 스크립트
"""
import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Please run: pip install huggingface_hub")
    sys.exit(1)


def download_model(model_name, checkpoint_dir="./checkpoint/"):
    """특정 모델 다운로드"""
    models = {
        "vit-s": {
            "name": "FS-VFM ViT-S",
            "files": [
                "pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth",
                "pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-te-599.pth",
                "pretrained_models/FS-VFM_ViT-S_VF2_600e/pretrain_ds_mean_std.txt",
            ]
        },
        "vit-b": {
            "name": "FS-VFM ViT-B",
            "files": [
                "pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth",
                "pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-te-600.pth",
                "pretrained_models/FS-VFM_ViT-B_VF2_600e/pretrain_ds_mean_std.txt",
            ]
        },
        "vit-l": {
            "name": "FS-VFM ViT-L",
            "files": [
                "pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-599.pth",
                "pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-te-599.pth",
                "pretrained_models/FS-VFM_ViT-L_VF2_600e/pretrain_ds_mean_std.txt",
            ]
        },
        "fsfm-vit-b": {
            "name": "FSFM ViT-B (CVPR25)",
            "files": [
                "pretrained_models/VF2_ViT-B/checkpoint-400.pth",
                "pretrained_models/VF2_ViT-B/checkpoint-te-400.pth",
                "pretrained_models/VF2_ViT-B/pretrain_ds_mean_std.txt",
            ]
        }
    }
    
    if model_name not in models:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(models.keys())}")
        return False
    
    model_info = models[model_name]
    print(f"\nDownloading {model_info['name']}...")
    
    repo_id = "Wolowolo/fsfm-3c"
    
    for filename in model_info['files']:
        print(f"  Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"  ✓ {filename} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            return False
    
    print(f"✓ {model_info['name']} downloaded successfully!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained FSFM models")
    parser.add_argument(
        "--model",
        type=str,
        default="vit-b",
        choices=["vit-s", "vit-b", "vit-l", "fsfm-vit-b", "all"],
        help="Model to download (default: vit-b)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoint/",
        help="Directory to save checkpoints (default: ./checkpoint/)"
    )
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("="*50)
    print("FSFM Pre-trained Model Downloader")
    print("="*50)
    
    if args.model == "all":
        print("\nDownloading all models...")
        models_to_download = ["vit-s", "vit-b", "vit-l", "fsfm-vit-b"]
        success_count = 0
        for model in models_to_download:
            if download_model(model, args.checkpoint_dir):
                success_count += 1
        
        print(f"\n{'='*50}")
        print(f"Downloaded {success_count}/{len(models_to_download)} models successfully")
        print(f"{'='*50}")
    else:
        if download_model(args.model, args.checkpoint_dir):
            print(f"\n{'='*50}")
            print("Download complete!")
            print(f"{'='*50}")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()

