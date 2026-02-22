"""
Clipper — Offline Indexing Pipeline

Segments all videos in data/videos/ into fixed-window clips,
encodes each clip with the full visual pipeline, and stores:
  - FAISS index of global clip embeddings  (for ANN search)
  - frame_embeds.npy                       (for local reranking)
  - clips_meta.jsonl                       (metadata per clip)

Usage:
    python indexer.py
"""
import os
import json
import torch
import numpy as np
import faiss
import cv2
from PIL        import Image
from pathlib    import Path
from tqdm       import tqdm

from config          import ClipperConfig
from clipper_utils   import get_video_duration, fixed_window_clips
from model           import ClipperModel


def sample_frames(video_path: str, t_start: float,
                  t_end: float, num_frames: int) -> list:
    """
    Uniformly samples num_frames RGB numpy arrays from [t_start, t_end].
    Pads with the last frame if the video is shorter than expected.
    """
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0   = int(t_start * fps)
    f1   = max(f0 + 1, int(t_end * fps) - 1)
    idxs = np.linspace(f0, f1, num_frames, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()

    while len(frames) < num_frames:
        frames.append(
            frames[-1] if frames
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
    return frames[:num_frames]


@torch.no_grad()
def encode_clip(model: ClipperModel, frames: list,
                device: str):
    """
    Encodes a list of RGB frames through the full visual pipeline.

    Returns:
      clip_embed   : (D,)    float32 — stored in FAISS
      frame_embeds : (T, D)  float32 — stored in frame cache for reranking
    """
    imgs = [model.preprocess(Image.fromarray(f)) for f in frames]
    x    = torch.stack(imgs).unsqueeze(0).to(device)   # [1, T, C, H, W]
    clip_e, frame_e = model.encode_video(x)
    return (
        clip_e.squeeze(0).cpu().numpy().astype(np.float32),
        frame_e.squeeze(0).cpu().numpy().astype(np.float32)
    )


def build_index(config: ClipperConfig = None):
    config = config or ClipperConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Indexer running on: {device}")

    # Load model
    model = ClipperModel(config).eval().to(device)
    if os.path.exists(config.weights_path):
        ckpt = torch.load(config.weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.config.use_custom_modules = True
        print(f"Loaded trained weights → custom modules: ON")
    else:
        print("No weights found → indexing in zero-shot mode")

    # Collect video files
    exts   = ("*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov")
    videos = [p for ext in exts for p in Path(config.video_dir).glob(ext)]
    if not videos:
        raise FileNotFoundError(f"No videos found in {config.video_dir}")
    print(f"Found {len(videos)} video(s)\n")

    all_clip_vecs  = []
    all_frame_vecs = []
    all_meta       = []
    clip_id        = 0

    for vp in tqdm(videos, desc="Videos"):
        dur = get_video_duration(str(vp))
        if dur < 1.0:
            continue
        clips = fixed_window_clips(dur, config.clip_len_s, config.stride_s)

        for t0, t1 in tqdm(clips, desc=f"  {vp.name}", leave=False):
            frames          = sample_frames(str(vp), t0, t1,
                                            config.num_frames)
            clip_e, frame_e = encode_clip(model, frames, device)

            all_clip_vecs.append(clip_e)
            all_frame_vecs.append(frame_e)
            all_meta.append({
                "clip_id":    clip_id,
                "video_id":   vp.stem,
                "filename":   vp.name,
                "video_path": str(vp),
                "t_start":    t0,
                "t_end":      t1
            })
            clip_id += 1

    N = len(all_clip_vecs)
    D = all_clip_vecs[0].shape[0]
    print(f"\nTotal clips: {N}  |  Embed dim: {D}")

    # Build FAISS index
    clip_matrix  = np.stack(all_clip_vecs).astype(np.float32)
    frame_matrix = np.stack(all_frame_vecs).astype(np.float32)

    if N <= 50_000:
        index = faiss.IndexFlatIP(D)
    else:
        nlist     = min(1024, max(8, N // 40))
        quantizer = faiss.IndexFlatIP(D)
        index     = faiss.IndexIVFFlat(quantizer, D, nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        print(f"Training IVFFlat (nlist={nlist})...")
        index.train(clip_matrix)

    index.add(clip_matrix)

    # Save everything
    os.makedirs(os.path.dirname(config.index_path), exist_ok=True)
    faiss.write_index(index, config.index_path)
    np.save(config.frame_cache_path, frame_matrix)
    with open(config.meta_path, "w") as f:
        for row in all_meta:
            f.write(json.dumps(row) + "\n")

    print(f"\n✅  FAISS index    → {config.index_path}")
    print(f"✅  Frame cache    → {config.frame_cache_path}"
          f"  shape={frame_matrix.shape}")
    print(f"✅  Metadata       → {config.meta_path}")


if __name__ == "__main__":
    build_index()
