# datasets.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import mediapipe as mp

class LipreadingDataset(Dataset):
    """
    PyTorch Dataset for lip-reading:
      - Expects a directory of video files (mp4, avi, mov, mpg).
      - `labels` is a dict mapping filename → transcript.
      - Uses MediaPipe FaceMesh to detect and crop the mouth region.
      - Outputs tensors of shape [C, T, H, W], where H=W=112.
    """
    SUPPORTED_EXTS = ('.mp4', '.avi', '.mov', '.mpg')
    # Inner‐lip landmarks for a tighter box
    MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    MIN_PAD_X = 5
    MIN_PAD_Y = 8
    SHIFT_Y   = 0

    def __init__(self, video_dir, labels, transform=None):
        super().__init__()
        self.video_dir = video_dir
        self.labels    = labels
        self.transform = transform

        # Collect all video paths
        self.video_paths = [
            os.path.join(video_dir, f)
            for f in sorted(os.listdir(video_dir))
            if f.lower().endswith(self.SUPPORTED_EXTS)
        ]

        # Do NOT create FaceMesh here:
        self.face_mesh = None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Lazy‐init FaceMesh inside worker
        if self.face_mesh is None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        path = self.video_paths[idx]
        filename = os.path.basename(path)

        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, img = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w, _ = img.shape

                xs = [int(lm[i].x * w) for i in self.MOUTH_IDX]
                ys = [int(lm[i].y * h) for i in self.MOUTH_IDX]

                mouth_w = max(xs) - min(xs)
                mouth_h = max(ys) - min(ys)
                pad_x = max(self.MIN_PAD_X, int(0.1 * mouth_w))
                pad_y = max(self.MIN_PAD_Y, int(0.2 * mouth_h))

                x0 = max(0, min(xs) - pad_x)
                x1 = min(w, max(xs) + pad_x)
                raw_t = min(ys) - pad_y
                raw_b = max(ys) + pad_y
                y0 = max(0, raw_t - self.SHIFT_Y)
                y1 = min(h, raw_b - self.SHIFT_Y)

                mouth = img[y0:y1, x0:x1]
                mouth = cv2.resize(mouth, (112, 112))
                frames.append(mouth)

        cap.release()

        if not frames:
            video_tensor = torch.zeros((3, 1, 112, 112), dtype=torch.float32)
        else:
            video_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1)
                for f in frames
            ], dim=1).float() / 255.0

        transcript = self.labels.get(filename, "")
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, transcript