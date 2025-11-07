#!/usr/bin/env python3
"""
대회 제출 전 불필요한 파일/폴더 정리 스크립트
- 대회 서버에서 ./data/는 자동으로 제공되므로 로컬 data는 삭제
- 학습 데이터(datasets)도 제출 불필요
- 기타 불필요한 파일 정리
"""

import os
import shutil
from pathlib import Path

def get_size_str(size_bytes):
    """파일 크기를 사람이 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_dir_size(path):
    """디렉토리 크기 계산"""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except:
                    pass
    except:
        pass
    return total

def backup_and_remove(path, backup_suffix="_backup"):
    """폴더를 백업하고 삭제"""
    path = Path(path)
    if not path.exists():
        return False, 0

    size = get_dir_size(path)
    backup_path = Path(str(path) + backup_suffix)

    # 기존 백업이 있으면 삭제
    if backup_path.exists():
        print(f"  기존 백업 삭제: {backup_path}")
        shutil.rmtree(backup_path, ignore_errors=True)

    # 백업 생성
    print(f"  백업 생성: {path} → {backup_path}")
    shutil.move(str(path), str(backup_path))

    return True, size

def remove_directory(path):
    """디렉토리 삭제 (백업 없이)"""
    path = Path(path)
    if not path.exists():
        return False, 0

    size = get_dir_size(path)
    print(f"  삭제: {path} ({get_size_str(size)})")
    shutil.rmtree(path, ignore_errors=True)

    return True, size

def cleanup_for_submission():
    print("=" * 70)
    print("대회 제출 전 불필요한 파일 정리")
    print("=" * 70)

    current_dir = Path.cwd()
    print(f"\n현재 디렉토리: {current_dir}")
    print(f"작업 디렉토리 이름: {current_dir.name}")

    # 제출에 불필요한 폴더 목록
    folders_to_backup = [
        "data",           # 대회 서버에서 제공
        "datasets",       # 학습 데이터 (제출 불필요)
    ]

    folders_to_remove = [
        ".venv",          # 가상환경
        "venv",
        ".git",           # Git 저장소
        "__pycache__",    # Python 캐시
        "runs",           # Tensorboard 로그
        "logs",           # 로그 파일
        "outputs",        # 학습 출력
        "pretrain",       # 사전학습 데이터
        ".ipynb_checkpoints",  # Jupyter 체크포인트
    ]

    print("\n" + "=" * 70)
    print("1단계: 중요 폴더 백업 (복구 가능)")
    print("=" * 70)

    total_backed_up = 0
    backed_up_folders = []

    for folder in folders_to_backup:
        if Path(folder).exists():
            backed_up, size = backup_and_remove(folder)
            if backed_up:
                total_backed_up += size
                backed_up_folders.append(folder)
                print(f"  ✓ {folder} 백업됨 ({get_size_str(size)})")
        else:
            print(f"  - {folder} 없음 (건너뜀)")

    print(f"\n총 백업된 크기: {get_size_str(total_backed_up)}")

    print("\n" + "=" * 70)
    print("2단계: 불필요한 폴더 삭제 (백업 없음)")
    print("=" * 70)

    total_removed = 0
    removed_folders = []

    for folder in folders_to_remove:
        # 최상위 레벨과 하위 폴더 모두 검색
        found_any = False

        # 최상위 레벨 확인
        if Path(folder).exists():
            removed, size = remove_directory(folder)
            if removed:
                total_removed += size
                removed_folders.append(folder)
                found_any = True

        # 하위 폴더에서도 검색 (__pycache__ 등)
        if folder.startswith("__") or folder.startswith("."):
            for item in Path(".").rglob(folder):
                if item.is_dir() and item.exists():
                    removed, size = remove_directory(item)
                    if removed:
                        total_removed += size
                        found_any = True

        if not found_any:
            print(f"  - {folder} 없음 (건너뜀)")

    print(f"\n총 삭제된 크기: {get_size_str(total_removed)}")

    print("\n" + "=" * 70)
    print("3단계: 깨진 심볼릭 링크 검색 및 제거")
    print("=" * 70)

    # 깨진 심볼릭 링크 찾기
    broken_links = []
    for item in Path(".").rglob("*"):
        try:
            if item.is_symlink() and not item.exists():
                broken_links.append(item)
        except:
            pass

    if broken_links:
        print(f"발견된 깨진 심볼릭 링크: {len(broken_links)}개")
        for link in broken_links[:10]:  # 처음 10개만 표시
            print(f"  - {link}")
        if len(broken_links) > 10:
            print(f"  ... 외 {len(broken_links) - 10}개")

        # 삭제
        for link in broken_links:
            try:
                link.unlink()
            except:
                pass
        print(f"\n  ✓ {len(broken_links)}개의 깨진 링크 삭제됨")
    else:
        print("  ✓ 깨진 심볼릭 링크가 없습니다")

    print("\n" + "=" * 70)
    print("4단계: 현재 디렉토리 구조 확인")
    print("=" * 70)

    # 남은 폴더들 확인
    remaining = []
    for item in Path(".").iterdir():
        if item.is_dir() and not item.name.startswith("."):
            size = get_dir_size(item)
            remaining.append((item.name, size))

    remaining.sort(key=lambda x: x[1], reverse=True)

    print("\n남은 폴더 (크기순):")
    total_size = 0
    for name, size in remaining:
        print(f"  {name:30s} {get_size_str(size):>12s}")
        total_size += size

    print(f"\n총 디렉토리 크기: {get_size_str(total_size)}")

    print("\n" + "=" * 70)
    print("✅ 정리 완료!")
    print("=" * 70)

    print("\n📋 정리 요약:")
    print(f"  • 백업된 폴더: {len(backed_up_folders)}개 ({get_size_str(total_backed_up)})")
    print(f"  • 삭제된 폴더: {len(removed_folders)}개 ({get_size_str(total_removed)})")
    print(f"  • 절감된 공간: {get_size_str(total_backed_up + total_removed)}")
    print(f"  • 현재 크기: {get_size_str(total_size)}")

    print("\n🎯 다음 단계:")
    print("  1. jupyter notebook task.ipynb 실행")
    print("  2. 마지막 셀에서 aif.submit() 실행")
    print("  3. 제출 완료!")

    print("\n♻️  복구 방법 (제출 후):")
    if backed_up_folders:
        print("  다음 명령어로 백업 복구:")
        for folder in backed_up_folders:
            print(f"    mv {folder}_backup {folder}")

    print("\n⚠️  주의사항:")
    print("  • 제출 전 반드시 task.ipynb의 key 값을 확인하세요")
    print("  • 제출 시 ./data/는 대회 서버에서 자동으로 제공됩니다")
    print("  • 모델 가중치(./model/)는 반드시 포함되어야 합니다")

    print("\n" + "=" * 70)

    return backed_up_folders

def restore_backups():
    """백업 복구"""
    print("=" * 70)
    print("백업 복구")
    print("=" * 70)

    backup_folders = [f for f in Path(".").iterdir() if f.name.endswith("_backup") and f.is_dir()]

    if not backup_folders:
        print("복구할 백업이 없습니다.")
        return

    for backup in backup_folders:
        original_name = backup.name.replace("_backup", "")
        original_path = Path(original_name)

        if original_path.exists():
            print(f"⚠️  {original_name}이(가) 이미 존재합니다. 건너뜁니다.")
            continue

        print(f"복구 중: {backup} → {original_path}")
        shutil.move(str(backup), str(original_path))
        print(f"  ✓ 복구됨")

    print("\n✅ 복구 완료!")
    print("=" * 70)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_backups()
    else:
        try:
            backed_up = cleanup_for_submission()
            print("\n✅ 스크립트 실행 완료!")
        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
