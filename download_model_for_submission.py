#!/usr/bin/env python3
"""
모델 다운로드 및 submission용 model 폴더에 배치하는 스크립트
FSFM ViT-B 모델을 다운로드하여 task.ipynb에서 사용할 수 있도록 준비합니다.
"""
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub가 설치되어 있지 않습니다.")
    print("다음 명령어로 설치하세요: pip install huggingface_hub")
    exit(1)

def main():
    print("="*60)
    print("FSFM ViT-B 모델 다운로드 (제출용)")
    print("="*60)

    # 다운로드할 모델 파일 (FSFM ViT-B CVPR25 버전)
    repo_id = "Wolowolo/fsfm-3c"
    model_file = "pretrained_models/VF2_ViT-B/checkpoint-400.pth"

    # 임시 다운로드 디렉토리
    temp_dir = "./temp_checkpoint/"
    os.makedirs(temp_dir, exist_ok=True)

    # model 폴더 생성
    model_dir = Path("./model/")
    model_dir.mkdir(exist_ok=True)

    print(f"\n1. 모델 다운로드 중...")
    print(f"   Repository: {repo_id}")
    print(f"   File: {model_file}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"   ✓ 다운로드 완료: {downloaded_path}")
    except Exception as e:
        print(f"   ✗ 다운로드 실패: {e}")
        return False

    # 다운로드된 파일을 model 폴더로 복사
    source_file = Path(temp_dir) / model_file
    target_file = model_dir / "fsfm_vit_base_checkpoint.pth"

    print(f"\n2. 모델 파일을 제출용 폴더로 복사 중...")
    print(f"   Source: {source_file}")
    print(f"   Target: {target_file}")

    try:
        shutil.copy2(source_file, target_file)
        print(f"   ✓ 복사 완료")
    except Exception as e:
        print(f"   ✗ 복사 실패: {e}")
        return False

    # 파일 크기 확인
    file_size = target_file.stat().st_size / (1024**3)  # GB
    print(f"\n3. 모델 파일 정보:")
    print(f"   경로: {target_file}")
    print(f"   크기: {file_size:.2f} GB")

    print(f"\n{'='*60}")
    print("✓ 모델 준비 완료!")
    print(f"{'='*60}")
    print(f"\n다음 파일이 생성되었습니다:")
    print(f"  {target_file.absolute()}")
    print(f"\ntask.ipynb에서 이 모델을 사용할 수 있습니다.")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
