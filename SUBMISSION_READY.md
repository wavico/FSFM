# ✅ Competition Submission Ready

**Date:** 2025-11-05
**Model:** FSFM ViT-Base (FS-VFM_ViT-B_VF2_600e)
**Directory Size:** 1.7 GB

---

## 📋 Submission Contents

- ✅ `task.ipynb` - Competition notebook (43 KB)
- ✅ `model/fsfm_vit_base_checkpoint.pth` - Model weights (1.5 GB)
- ✅ `fsvfm/` - Source code (548 KB)
- ✅ No data/ or datasets/ folders (competition provides data)
- ✅ No broken symbolic links
- ✅ No unnecessary files (.venv, .git, __pycache__, etc.)

---

## 🚀 Submission Methods

### Method 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook task.ipynb
```

In the notebook:
1. Locate the last cell with submission code
2. Set your competition key: `key = "YOUR_KEY_HERE"`
3. Run: `aif.submit()`
4. Wait for upload to complete

### Method 2: Manual ZIP Upload

```bash
# Create submission zip (excludes backups)
zip -r fsfm_submission.zip . \
  -x "data_backup/*" "datasets_backup/*" ".conda/*" \
  "SUBMISSION_READY.md" "cleanup_for_submission.py"

# Upload fsfm_submission.zip to competition platform
```

---

## ⚠️ Important Notes

### Competition Environment
- The competition server provides `./data/` automatically
- Network is closed after `pip install` phase
- 3-hour execution time limit
- GPU with sufficient VRAM (~1 GB required for inference)

### Model Details
- **Model:** Vision Transformer Base (ViT-B)
- **Pretrained:** FS-VFM_ViT-B_VF2_600e (600 epochs)
- **Input Size:** 224×224 RGB
- **Output:** Binary classification (0=Real, 1=Fake)
- **Face Detection:** Multi-method fallback (Mediapipe → dlib → Haar Cascade)

### Performance Estimates
- **VRAM Usage:** ~0.8 GB (FP32 inference)
- **Processing Speed:** ~4.15 files/sec
- **Estimated Time:** ~5 minutes for 1,200 test files

---

## 🔄 Recovery After Submission

If you need to restore your working environment:

```bash
# Restore backed up folders
mv data_backup data
mv datasets_backup datasets

# Or use the cleanup script's restore function
python cleanup_for_submission.py restore
```

---

## 📊 What Was Cleaned

**Total Space Freed:** 4.37 GB

### Backed Up (Recoverable)
- `data/` → `data_backup/` (0 B)
- `datasets/` → `datasets_backup/` (14.18 MB)

### Deleted (Not Recoverable)
- `.venv/` (4.33 GB) - Virtual environment
- `.git/` (808 KB) - Git repository
- `__pycache__/` (Multiple locations) - Python cache
- `runs/` (6.29 KB) - Tensorboard logs
- `pretrain/` (2.85 KB) - Pretrain scripts
- `.ipynb_checkpoints/` (34.87 KB) - Jupyter checkpoints
- `aif.zip` (2.2 GB) - Old submission file
- Deprecated submission scripts

---

## ✅ Pre-Submission Checklist

Before submitting, verify:

- [ ] `task.ipynb` exists and has your competition key
- [ ] `model/fsfm_vit_base_checkpoint.pth` exists (1.5 GB)
- [ ] No `data/` or `datasets/` folders in current directory
- [ ] `fsvfm/` source code is present
- [ ] Tested locally (optional but recommended)
- [ ] Have backup of original repository (if needed)

---

## 🎯 Expected Output Format

The submission will generate `predictions.csv`:

```csv
filename,label
video_001.mp4,1
video_002.mp4,0
video_003.mp4,1
...
```

Where:
- `filename`: Input file name (string)
- `label`: 0 (Real) or 1 (Fake)

---

## 🆘 Troubleshooting

### If submission fails:

1. **Check model file:**
   ```bash
   ls -lh model/fsfm_vit_base_checkpoint.pth
   ```
   Should be ~1.5 GB

2. **Verify task.ipynb:**
   ```bash
   jupyter notebook task.ipynb
   ```
   Test run locally before submission

3. **Check for broken symlinks:**
   ```bash
   find . -xtype l
   ```
   Should return nothing

4. **Verify directory size:**
   ```bash
   du -sh .
   ```
   Should be ~1.7 GB

### If you need to re-download the model:

```python
from huggingface_hub import hf_hub_download
import os

os.makedirs("./model", exist_ok=True)
hf_hub_download(
    repo_id="Wolowolo/fsfm-3c",
    filename="pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth",
    local_dir="./model/",
    local_dir_use_symlinks=False
)
# Then rename to: ./model/fsfm_vit_base_checkpoint.pth
```

---

## 📚 Additional Resources

- **Model Repository:** https://huggingface.co/Wolowolo/fsfm-3c
- **Competition Platform:** (Your competition URL)
- **FSFM Paper:** CVPR 2025

---

**Good luck with your submission! 🚀**
