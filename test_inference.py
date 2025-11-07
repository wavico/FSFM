import os
import sys
from PIL import Image
import cv2
from pathlib import Path
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import time
import warnings
from functools import partial
import argparse
warnings.filterwarnings('ignore')

# Mediapipe 로그 억제
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ============================================================
# FSFM Vision Transformer Model Definition
# ============================================================
import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# ============================================================
# Face Detection Utilities (Mediapipe + Haar Cascade only)
# ============================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VIDEO_EXTS = {".avi", ".mp4", ".AVI", ".MP4"}

HAAR_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

if MEDIAPIPE_AVAILABLE:
    try:
        mp_face_detection = mp.solutions.face_detection
        MEDIAPIPE_DETECTOR = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
    except:
        MEDIAPIPE_DETECTOR = None
        MEDIAPIPE_AVAILABLE = False
else:
    MEDIAPIPE_DETECTOR = None

def detect_face_haar(image_np, target_size=(224, 224)):
    """Face detection using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = HAAR_FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    expansion = int(max(w, h) * 0.3)
    x = max(0, x - expansion)
    y = max(0, y - expansion)
    w = min(image_np.shape[1] - x, w + 2 * expansion)
    h = min(image_np.shape[0] - y, h + 2 * expansion)
    cropped_np = image_np[y:y + h, x:x + w]
    face_img = Image.fromarray(cropped_np).resize(target_size, Image.BICUBIC)
    return face_img

def detect_face_mediapipe(image_np, target_size=(224, 224)):
    """Face detection using Mediapipe"""
    if MEDIAPIPE_DETECTOR is None:
        return None
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGB)
    results = MEDIAPIPE_DETECTOR.process(image_rgb)
    if not results.detections:
        return None
    detection = max(results.detections, key=lambda d: d.score[0])
    h, w = image_np.shape[:2]
    bbox = detection.location_data.relative_bounding_box
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    margin = int(max(x2 - x1, y2 - y1) * 0.1)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    cropped_np = image_np[y1:y2, x1:x2]
    if cropped_np.size == 0:
        return None
    face_img = Image.fromarray(cropped_np).resize(target_size, Image.BICUBIC)
    return face_img

def detect_and_crop_face_multi(image: Image.Image, target_size=(224, 224)):
    """Multi-method face detection with fallback (Mediapipe -> Haar -> Full Image)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)

    # Try Mediapipe first
    if MEDIAPIPE_AVAILABLE:
        try:
            face_img = detect_face_mediapipe(image_np, target_size)
            if face_img:
                return face_img
        except:
            pass

    # Try Haar Cascade
    try:
        face_img = detect_face_haar(image_np, target_size)
        if face_img:
            return face_img
    except:
        pass

    # Fallback: resize original image
    resized_img = Image.fromarray(image_np).resize(target_size, Image.BICUBIC)
    return resized_img

def process_video_frames(video_path, num_frames=10, max_duration=10):
    """Extract and process frames from video"""
    face_images = []
    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return face_images
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return face_images
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            max_frames = int(fps * max_duration)
            total_frames = min(total_frames, max_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_img = detect_and_crop_face_multi(image)
            if face_img:
                face_images.append(face_img)
    except:
        pass
    finally:
        if cap is not None:
            cap.release()
    return face_images

# ============================================================
# Main Inference Logic
# ============================================================
if __name__ == "__main__":
    print("Starting inference test...")

    # Model path and test data path
    model_weights_path = "./model/fsfm_vit_base_checkpoint.pth"
    test_dataset_path = Path("./data")
    output_csv_path = Path("submission.csv")

    # Load model
    print("Loading model...")
    model = vit_base_patch16(num_classes=2, global_pool=True, drop_path_rate=0.1)

    # Load checkpoint with safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(model_weights_path, map_location='cpu', weights_only=True)
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    # Remove head weights if shape mismatch
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    load_result = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Model loaded: {load_result}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model ready on {device}")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get all files from data directory
    files = []
    for root, dirs, filenames in os.walk(test_dataset_path):
        for filename in filenames:
            files.append(Path(root) / filename)

    total_files = len(files)
    print(f"Processing {total_files} test files...")

    # CSV header
    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

    results = []
    start_time = time.time()

    # Process files
    for idx, file_path in enumerate(tqdm(files, desc="Processing", ncols=80)):
        face_images = []
        ext = file_path.suffix.lower()
        predicted_class = 0

        try:
            if ext in IMAGE_EXTS:
                image = Image.open(file_path)
                face_img = detect_and_crop_face_multi(image)
                if face_img:
                    face_images = [face_img]
            elif ext in VIDEO_EXTS:
                face_images = process_video_frames(file_path, num_frames=10, max_duration=10)

            # Inference
            if len(face_images) > 0:
                with torch.no_grad():
                    batch = face_images[:min(len(face_images), 10)]
                    img_tensors = torch.stack([transform(img) for img in batch]).to(device)

                    logits_list = []
                    for img_tensor in img_tensors:
                        logits = model(img_tensor.unsqueeze(0))
                        logits_list.append(logits)

                    avg_logits = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
                    probs = F.softmax(avg_logits, dim=1)
                    predicted_class = torch.argmax(probs).item()

                    del img_tensors, logits_list, avg_logits, probs
                    if device == "cuda":
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            predicted_class = 0

        results.append([file_path.name, int(predicted_class)])

    # Write results
    with open(output_csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)

    elapsed_total = time.time() - start_time
    print(f"\n✓ Test completed in {elapsed_total:.2f}s")
    print(f"✓ Results written to {output_csv_path}")
    print(f"✓ Processed {len(results)}/{total_files} files")
