"""
Clipper — PyTorch Dataset for MSR-VTT and MSVD.

Annotation JSON format:
[
  {
    "video_id":   "video0",
    "video_path": "data/msrvtt/videos/video0.mp4",
    "captions":   ["a man is talking", "someone speaks to camera", ...]
  },
  ...
]
"""
import json
import random
import numpy as np
import torch
import cv2
from PIL              import Image
from torch.utils.data import Dataset

from config import ClipperConfig


class VideoTextDataset(Dataset):

    def __init__(self, ann_path: str, config: ClipperConfig,
                 preprocess, tokenizer, split: str = "train"):
        self.config     = config
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.split      = split

        with open(ann_path) as f:
            self.data = json.load(f)

        print(f"[{split}] Loaded {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def _sample_frames(self, video_path: str) -> list:
        """
        Uniformly samples config.num_frames RGB frames from the full video.
        Returns list of numpy arrays (H, W, 3).
        """
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)
        idxs  = np.linspace(0, total - 1, self.config.num_frames, dtype=int)

        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()

        while len(frames) < self.config.num_frames:
            frames.append(
                frames[-1] if frames
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )
        return frames[:self.config.num_frames]

    def __getitem__(self, idx: int):
        entry = self.data[idx]

        # ── Frames → tensor ───────────────────────────────────────
        frames = self._sample_frames(entry["video_path"])
        imgs   = [self.preprocess(Image.fromarray(f)) for f in frames]
        frames_tensor = torch.stack(imgs)              # [T, C, H, W]

        # ── Caption → tokens ──────────────────────────────────────
        # Training: random caption per step (data augmentation)
        # Eval: always use first caption for reproducibility
        caps    = entry.get("captions", [""])
        caption = random.choice(caps) if self.split == "train" else caps[0]
        tokens  = self.tokenizer([caption])[0]         # [77]

        return {
            "frames":   frames_tensor,
            "tokens":   tokens,
            "video_id": entry["video_id"]
        }


def collate_fn(batch):
    """Stacks frames and tokens into batched tensors."""
    return {
        "frames":    torch.stack([b["frames"] for b in batch]),  # [B, T, C, H, W]
        "tokens":    torch.stack([b["tokens"] for b in batch]),  # [B, 77]
        "video_ids": [b["video_id"] for b in batch]
    }
