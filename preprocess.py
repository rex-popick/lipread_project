# preprocess.py

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Settings
VIDEO_DIR  = "data/videos_small"
CACHE_DIR  = "data/cache_small"
ALIGN_DIR  = "data/grid_align"    # if you need transcripts too
LABELS_JSON= "labels.json"
MOUTH_IDX  = [78,95,88,178,87,14,317,402,318,324,308]
PAD_X       = 5
PAD_Y       = 8
IMG_SIZE    = (112,112)

os.makedirs(CACHE_DIR, exist_ok=True)

# Load labels (optional—only if you also want to cache text)
with open(LABELS_JSON) as f:
    labels = json.load(f)

# Initialize FaceMesh once
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.5
)

for fname in tqdm(sorted(os.listdir(VIDEO_DIR))):
    if not fname.endswith(".mpg"):
        continue

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, fname))
    frames = []

    while True:
        ret, img = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        h, w, _ = img.shape
        xs = [int(lm[i].x * w) for i in MOUTH_IDX]
        ys = [int(lm[i].y * h) for i in MOUTH_IDX]

        x0 = max(0, min(xs) - PAD_X)
        x1 = min(w, max(xs) + PAD_X)
        y0 = max(0, min(ys) - PAD_Y)
        y1 = min(h, max(ys) + PAD_Y)

        mouth = img[y0:y1, x0:x1]
        mouth = cv2.resize(mouth, IMG_SIZE)
        frames.append(mouth)

    cap.release()

    # Save as [T, H, W, C] uint8
    arr = np.stack(frames, axis=0)
    np.save(os.path.join(CACHE_DIR, fname.replace(".mpg", ".npy")), arr)