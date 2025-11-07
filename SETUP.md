# 🚀 자동 설정 가이드 (Automated Setup Guide)

이 문서는 FSFM-CVPR25 프로젝트를 다른 환경에서 클론받았을 때 자동으로 필요한 파일들을 다운로드하는 방법을 설명합니다.

## 📦 자동으로 설치되는 항목

Docker 컨테이너를 시작하면 다음 항목들이 자동으로 확인 및 설치됩니다:

1. ✅ **FACER 툴킷** - Face parsing을 위한 도구
2. ✅ **사전학습된 모델 가중치** - ViT-S/B/L 모델
3. ⚠️ **데이터셋** - 옵션으로 자동 다운로드 가능 (대용량)

---

## 🐳 Docker로 자동 설정하기 (권장)

### 1️⃣ 기본 사용 (ViT-B 모델 자동 다운로드)

```bash
# 레포지토리 클론
git clone https://github.com/your-repo/FSFM-CVPR25.git
cd FSFM-CVPR25

# Docker 컨테이너 빌드 및 시작
docker-compose up -d

# 컨테이너 접속
docker exec -it fsfm-training bash
```

컨테이너 시작 시 자동으로:
- FACER 툴킷이 클론됩니다
- FS-VFM ViT-B 모델이 다운로드됩니다
- 필요한 디렉토리가 생성됩니다

### 2️⃣ 다른 모델 다운로드

`docker-compose.yml` 파일에서 환경변수를 수정:

```yaml
environment:
  - DOWNLOAD_MODEL=vit-l  # ViT-Large 모델 다운로드
```

**사용 가능한 옵션:**
- `vit-s` - FS-VFM ViT-Small (가장 빠름, 성능 낮음)
- `vit-b` - FS-VFM ViT-Base (기본값, 균형잡힌 성능)
- `vit-l` - FS-VFM ViT-Large (가장 느림, 성능 높음)
- `fsfm-vit-b` - FSFM ViT-Base (CVPR25 버전)
- `all` - 모든 모델 다운로드 (약 10GB+)

### 3️⃣ 데이터셋 자동 다운로드 (선택사항)

⚠️ **주의**: 데이터셋은 매우 큰 용량(수십~수백 GB)입니다!

```yaml
environment:
  - AUTO_DOWNLOAD_DATASET=true  # 주석 해제
```

---

## 💻 Docker 없이 수동 설정

### 1️⃣ Python 환경 설정

```bash
# Conda 환경 생성
conda create -n fsfm3c python=3.9.21
conda activate fsfm3c

# PyTorch 설치
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ FACER 툴킷 설치

```bash
cd datasets/pretrain/preprocess/tools
git clone https://github.com/FacePerceiver/facer.git
cd ../../../../
```

### 3️⃣ 사전학습된 모델 다운로드

**방법 A: 자동 스크립트 사용**
```bash
python3 setup_download_models.py --model vit-b
```

**방법 B: 기존 스크립트 사용**
```bash
cd fsvfm/pretrain
python download_pretrained_weitghts.py
```

**방법 C: 수동 다운로드**
- 🤗 [Hugging Face Model Hub](https://huggingface.co/Wolowolo/fsfm-3c/tree/main/pretrained_models)에서 다운로드
- `fsvfm/pretrain/checkpoint/pretrained_models/` 폴더에 저장

### 4️⃣ 데이터셋 다운로드

**테스트만 하는 경우:**
```bash
# Hugging Face에서 전처리된 데이터셋 다운로드
# 🤗 https://huggingface.co/datasets/Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM/tree/main/finetune_datasets

# datasets/finetune_datasets/ 폴더에 압축 해제
```

**전체 학습을 하는 경우:**
- VGGFace2, FaceForensics++, DiFF 등의 원본 데이터셋 다운로드
- README.md의 데이터셋 전처리 섹션 참고

---

## 🔍 설정 확인

### 모든 것이 제대로 설치되었는지 확인:

```bash
# FACER 툴킷 확인
ls -la datasets/pretrain/preprocess/tools/facer

# 모델 가중치 확인
ls -la fsvfm/pretrain/checkpoint/pretrained_models/

# Python 환경 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🎯 빠른 테스트 실행

설정이 완료되면 바로 테스트 가능:

```bash
# Deepfake Detection 테스트
cd fsvfm/finetune/cross_dataset_DFD_and_DiFF
bash scripts_DFD/run_DfD-ViT-B.sh
```

---

## 📁 필요한 디렉토리 구조

자동 설정 후 다음과 같은 구조가 생성됩니다:

```
FSFM-CVPR25/
├── datasets/
│   ├── pretrain/
│   │   └── preprocess/
│   │       └── tools/
│   │           └── facer/          # ✅ 자동 클론
│   └── finetune_datasets/          # ⚠️ 수동 다운로드 필요
├── fsvfm/
│   └── pretrain/
│       └── checkpoint/
│           └── pretrained_models/  # ✅ 자동 다운로드
├── data/                           # ⚠️ 원본 데이터 (매우 큼)
├── outputs/                        # 학습 결과
├── logs/                           # 로그 파일
└── checkpoints/                    # 체크포인트 저장
```

---

## ❓ 문제 해결

### 문제 1: 모델 다운로드 실패
```bash
# huggingface_hub 재설치
pip install --upgrade huggingface_hub

# 수동으로 다운로드 스크립트 실행
python3 setup_download_models.py --model vit-b
```

### 문제 2: FACER 툴킷 에러
```bash
# FACER 재설치
cd datasets/pretrain/preprocess/tools
rm -rf facer
git clone https://github.com/FacePerceiver/facer.git
```

### 문제 3: CUDA 메모리 부족
```bash
# docker-compose.yml에서 GPU 설정 조정
environment:
  - CUDA_VISIBLE_DEVICES=0  # 사용할 GPU 번호 지정
```

### 문제 4: 디스크 공간 부족
```bash
# Docker 이미지 및 컨테이너 정리
docker system prune -a --volumes

# 필요없는 모델 삭제 (예: all 다운로드 후 일부만 사용)
rm -rf fsvfm/pretrain/checkpoint/pretrained_models/FS-VFM_ViT-L_VF2_600e
```

---

## 🔗 추가 자료

- 📖 [메인 README](./README.md) - 전체 프로젝트 문서
- 🤗 [Hugging Face Models](https://huggingface.co/Wolowolo/fsfm-3c)
- 🤗 [Hugging Face Datasets](https://huggingface.co/datasets/Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM)
- 📝 [Paper (arXiv)](https://arxiv.org/abs/2510.10663)

---

## 💡 팁

1. **디스크 공간 확보**: 전체 설정에는 약 100GB+ 필요
2. **네트워크**: 모델 다운로드에 안정적인 인터넷 연결 필요
3. **GPU 메모리**: 최소 8GB VRAM 권장 (ViT-L은 16GB+)
4. **처음 사용**: ViT-B 모델로 시작하는 것을 권장

---

**문제가 해결되지 않으면 [Issue](https://github.com/wolo-wolo/FSFM-CVPR25/issues)를 열어주세요!** 🙏

