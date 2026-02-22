"""
Clipper — Evaluation Module

Computes standard text-to-video retrieval metrics:
  R@1, R@5, R@10  (higher is better)
  MdR             (Median Rank — lower is better)
  MnR             (Mean Rank   — lower is better)
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm           import tqdm

from config import ClipperConfig


@torch.no_grad()
def run_evaluation(model, val_loader, device: str,
                   config: ClipperConfig) -> dict:
    """
    Builds full text and video embedding matrices from the validation set,
    computes the complete (N × N) similarity matrix, then derives metrics.

    Uses chunked local similarity to avoid OOM on large validation sets.
    """
    model.eval()

    all_text_embeds    = []
    all_concept_embeds = []
    all_clip_embeds    = []
    all_frame_embeds   = []

    for batch in tqdm(val_loader, desc="  Encoding val", leave=False):
        frames = batch["frames"].to(device)
        tokens = batch["tokens"].to(device)

        with autocast(enabled=config.fp16):
            t_emb, c_emb = model.encode_text(tokens)
            v_emb, f_emb = model.encode_video(frames, t_emb)

        all_text_embeds.append(t_emb.cpu())
        all_concept_embeds.append(c_emb.cpu())
        all_clip_embeds.append(v_emb.cpu())
        all_frame_embeds.append(f_emb.cpu())

    # Full matrices
    T_mat = torch.cat(all_text_embeds,    dim=0)  # [N, D]
    C_mat = torch.cat(all_concept_embeds, dim=0)  # [N, Q, D]
    V_mat = torch.cat(all_clip_embeds,    dim=0)  # [N, D]
    F_mat = torch.cat(all_frame_embeds,   dim=0)  # [N, T, D]
    N     = T_mat.shape[0]

    # ── Global similarity matrix ──────────────────────────────────
    scale = model.similarity.logit_scale.exp().clamp(max=100).cpu()
    G_sim = scale * (T_mat @ V_mat.T)              # [N, N]

    # ── Local similarity matrix (chunked to avoid OOM) ────────────
    CHUNK = 64
    L_sim = torch.zeros(N, N)

    for i in range(0, N, CHUNK):
        c_batch = C_mat[i:i+CHUNK].to(device)
        for j in range(0, N, CHUNK):
            f_batch = F_mat[j:j+CHUNK].to(device)
            with autocast(enabled=config.fp16):
                ls = model.similarity.local_similarity(c_batch, f_batch)
            L_sim[i:i+CHUNK, j:j+CHUNK] = ls.cpu()

    # ── Fused score ───────────────────────────────────────────────
    α = config.similarity_alpha
    S = α * G_sim + (1.0 - α) * L_sim             # [N, N]

    # ── Retrieval metrics ─────────────────────────────────────────
    # Diagonal of S contains correct (text_i, video_i) pair scores
    ranks = []
    for i in range(N):
        sorted_ids = S[i].argsort(descending=True).tolist()
        rank       = sorted_ids.index(i) + 1       # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1":  float((ranks <= 1).mean()  * 100),
        "R@5":  float((ranks <= 5).mean()  * 100),
        "R@10": float((ranks <= 10).mean() * 100),
        "MdR":  float(np.median(ranks)),
        "MnR":  float(np.mean(ranks))
    }
