from dataclasses import dataclass


@dataclass
class ClipperConfig:
    # ── CLIP backbone ─────────────────────────────────────────────
    clip_model:  str   = "ViT-B-32"
    embed_dim:   int   = 512          # 512 for ViT-B-32, 768 for ViT-L-14
    num_frames:  int   = 12

    # ── Clip segmentation ─────────────────────────────────────────
    clip_len_s:  float = 4.0          # each segment = 4 seconds
    stride_s:    float = 2.0          # 50% overlap between segments

    # ── Query Module (cross-attention) ────────────────────────────
    num_query_vectors:    int   = 8
    num_attn_heads:       int   = 8   # must divide embed_dim evenly
    query_dropout:        float = 0.1

    # ── Similarity fusion ─────────────────────────────────────────
    # final_score = alpha * global + (1 - alpha) * local
    similarity_alpha:     float = 0.7

    # ── Other modules ─────────────────────────────────────────────
    excitation_reduction: int   = 4   # bottleneck ratio in excitation
    use_custom_modules:   bool  = False  # set True after fine-tuning

    # ── Paths ─────────────────────────────────────────────────────
    video_dir:         str = "data/videos/"
    index_path:        str = "data/index/clips.faiss"
    meta_path:         str = "data/index/clips_meta.jsonl"
    frame_cache_path:  str = "data/index/frame_embeds.npy"
    weights_path:      str = "data/checkpoints/clipper.pt"
    best_ckpt_path:    str = "data/checkpoints/clipper_best.pt"

    # ── Training ──────────────────────────────────────────────────
    train_ann_path:    str   = "data/msrvtt/train_ann.json"
    val_ann_path:      str   = "data/msrvtt/val_ann.json"
    batch_size:        int   = 32
    num_epochs:        int   = 10
    lr:                float = 1e-4
    weight_decay:      float = 0.01
    warmup_steps:      int   = 200
    grad_clip:         float = 1.0
    accum_steps:       int   = 4      # gradient accumulation steps
    fp16:              bool  = True   # mixed precision
    num_workers:       int   = 4
    eval_every:        int   = 1      # evaluate every N epochs

    # ── Retrieval ─────────────────────────────────────────────────
    top_k:        int = 10
    rerank_top_k: int = 50            # FAISS fetches 50, rerank → top_k
