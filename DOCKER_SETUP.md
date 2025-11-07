# 🐳 Docker 자동화 설정 가이드

이 가이드는 FSFM-CVPR25 프로젝트를 Docker를 통해 **완전 자동화**하여 실행하는 방법을 설명합니다.

## 🎯 자동화된 기능

✅ **환경 설정** - CUDA, cuDNN, Python 패키지 자동 설치  
✅ **FACER 툴킷** - Face parsing 툴킷 자동 클론  
✅ **사전학습 모델** - 🤗 Hugging Face에서 모델 가중치 자동 다운로드  
✅ **데이터셋** - 전처리된 데이터셋 자동 다운로드 (옵션)  

---

## 📋 사전 요구사항

- Docker Engine 20.10+
- Docker Compose 1.29+
- NVIDIA Docker Runtime (GPU 사용 시)
- 최소 50GB 디스크 공간 (전체 데이터셋 다운로드 시 100GB+)

---

## 🚀 빠른 시작

### 1️⃣ 저장소 클론
```bash
git clone https://github.com/wolo-wolo/FSFM-CVPR25.git
cd FSFM-CVPR25
```

### 2️⃣ Docker 컨테이너 빌드 및 실행
```bash
docker-compose up -d
```

### 3️⃣ 컨테이너 접속
```bash
docker exec -it fsfm-training bash
```

**첫 실행 시 자동으로 수행되는 작업:**
- ✅ FACER 툴킷 설치
- ✅ 사전학습된 ViT-B 모델 다운로드
- ✅ 필요한 디렉토리 생성

---

## ⚙️ 고급 설정

### 환경 변수 설정

`docker-compose.yml` 파일에서 다음 환경 변수를 수정할 수 있습니다:

```yaml
environment:
  # 다운로드할 모델 크기 선택
  - MODEL_SIZE=ViT-B  # Options: ViT-S, ViT-B, ViT-L, all
  
  # 다운로드할 데이터셋 타입 선택
  - DATASET_TYPE=minimal  # Options: minimal, dfd, diff, fas, all
  
  # 첫 실행 시 데이터셋 자동 다운로드 여부
  - AUTO_DOWNLOAD_DATASETS=false  # true로 설정하면 자동 다운로드
  
  # 사용할 GPU 번호
  - CUDA_VISIBLE_DEVICES=0  # 0,1,2,3 등으로 변경 가능
```

### 모델 크기 옵션

| 옵션 | 설명 | 다운로드 크기 |
|------|------|---------------|
| `ViT-S` | Small 모델만 | ~400MB |
| `ViT-B` | Base 모델만 (기본값) | ~350MB |
| `ViT-L` | Large 모델만 | ~1.2GB |
| `all` | 모든 모델 | ~2GB |

### 데이터셋 타입 옵션

| 옵션 | 설명 | 크기 |
|------|------|------|
| `minimal` | 테스트용 샘플 (기본값) | ~1GB |
| `dfd` | Deepfake Detection 데이터셋 | ~20GB |
| `diff` | Diffusion face forgery 데이터셋 | ~15GB |
| `fas` | Face Anti-Spoofing 데이터셋 | ~10GB |
| `all` | 모든 데이터셋 (경고: 매우 큼!) | ~50GB+ |

---

## 🔧 수동 다운로드

### 추가 모델 다운로드
컨테이너 내부에서:
```bash
# 특정 모델 다운로드
export MODEL_SIZE=ViT-L
bash scripts/download_models.sh

# 모든 모델 다운로드
export MODEL_SIZE=all
bash scripts/download_models.sh
```

### 추가 데이터셋 다운로드
컨테이너 내부에서:
```bash
# Deepfake Detection 데이터셋
export DATASET_TYPE=dfd
bash scripts/download_datasets.sh

# 모든 데이터셋 (주의: 매우 큼!)
export DATASET_TYPE=all
bash scripts/download_datasets.sh
```

---

## 📁 디렉토리 구조

```
FSFM-CVPR25/
├── docker-compose.yml           # Docker Compose 설정
├── Dockerfile                   # Docker 이미지 정의
├── docker-entrypoint.sh         # 컨테이너 시작 스크립트
├── scripts/
│   ├── setup.sh                # 초기 설정 스크립트
│   ├── download_models.sh      # 모델 다운로드 스크립트
│   └── download_datasets.sh    # 데이터셋 다운로드 스크립트
├── fsvfm/
│   └── pretrain/
│       └── checkpoint/
│           └── pretrained_models/  # 다운로드된 모델 저장소
└── datasets/
    ├── pretrain/
    │   └── preprocess/
    │       └── tools/
    │           └── facer/      # 자동 설치됨
    └── finetune_datasets/      # 다운로드된 데이터셋
```

---

## 🎯 실행 예제

### Deepfake Detection 학습
```bash
docker exec -it fsfm-training bash
cd fsvfm/finetune/cross_dataset_DFD_and_DiFF
bash scripts_DFD/run_DfD-ViT-B.sh
```

### Diffusion Face Forgery Detection 학습
```bash
docker exec -it fsfm-training bash
cd fsvfm/finetune/cross_dataset_DFD_and_DiFF
bash scripts_DiFF/run_DiFF-ViT-B.sh
```

### Face Anti-Spoofing 학습
```bash
docker exec -it fsfm-training bash
cd fsvfm/finetune/cross_domain_FAS
bash scripts/run_base.sh
```

---

## 🔍 트러블슈팅

### 1. 모델 다운로드 실패
```bash
# 수동으로 재시도
docker exec -it fsfm-training bash
bash scripts/download_models.sh
```

### 2. FACER 툴킷 설치 실패
```bash
# 수동 설치
docker exec -it fsfm-training bash
cd datasets/pretrain/preprocess/tools
git clone https://github.com/FacePerceiver/facer.git
```

### 3. GPU 인식 안 됨
```bash
# NVIDIA Docker Runtime 확인
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# docker-compose.yml에서 GPU 설정 확인
```

### 4. 디스크 공간 부족
```bash
# 불필요한 Docker 이미지/볼륨 정리
docker system prune -a --volumes

# 특정 데이터셋만 다운로드
export DATASET_TYPE=dfd  # 또는 diff, fas
bash scripts/download_datasets.sh
```

### 5. 초기 설정 재실행
```bash
# .setup_complete 파일 삭제 후 컨테이너 재시작
docker exec -it fsfm-training rm /workspace/.setup_complete
docker restart fsfm-training
```

---

## 💡 유용한 명령어

### 컨테이너 상태 확인
```bash
docker-compose ps
docker logs fsfm-training
```

### 컨테이너 중지/시작
```bash
docker-compose stop
docker-compose start
```

### 컨테이너 재시작
```bash
docker-compose restart
```

### 컨테이너 완전 재구축
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### TensorBoard 실행
```bash
# 컨테이너 내부에서
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# 브라우저에서 http://localhost:6006 접속
```

---

## 📊 리소스 요구사항

### 최소 사양
- CPU: 4 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB+ VRAM
- Disk: 50GB

### 권장 사양
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA GPU with 24GB+ VRAM
- Disk: 100GB+ SSD

---

## 🔗 참고 링크

- [메인 README](./README.md)
- [🤗 Hugging Face 모델](https://huggingface.co/Wolowolo/fsfm-3c)
- [🤗 Hugging Face 데이터셋](https://huggingface.co/datasets/Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM)
- [프로젝트 페이지](https://fsfm-3c.github.io/fsvfm.html)
- [논문](https://arxiv.org/pdf/2510.10663)

---

## ❓ FAQ

**Q: 첫 실행 시 얼마나 걸리나요?**  
A: 네트워크 속도에 따라 다르지만, 모델 다운로드는 5-10분, 전체 데이터셋 다운로드는 1-3시간 소요될 수 있습니다.

**Q: 인터넷 없이 사용할 수 있나요?**  
A: 모델과 데이터셋을 미리 다운로드한 후 볼륨 마운트하면 오프라인에서도 사용 가능합니다.

**Q: 여러 GPU를 사용하려면?**  
A: `docker-compose.yml`에서 `CUDA_VISIBLE_DEVICES=0,1,2,3`로 변경하고, 학습 스크립트의 `--nproc_per_node` 값을 조정하세요.

**Q: Windows에서 사용할 수 있나요?**  
A: WSL2 + Docker Desktop + NVIDIA CUDA on WSL을 설치하면 Windows에서도 사용 가능합니다.

---

## 📝 라이선스

이 프로젝트는 [CC BY-NC 4.0 라이선스](./LICENSE) 하에 배포됩니다.

