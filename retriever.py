"""
Clipper — Online Retrieval Engine

Search flow:
  1. Encode text query → text_embed [D] + concept_embeds [Q, D]
  2. FAISS global search → top rerank_top_k clips by cosine sim
  3. Load pre-cached frame_embeds for those clips → [K, T, D]
  4. Local MaxSim: concept_embeds ↔ frame_embeds → reranked scores
  5. Fused = alpha * global + (1-alpha) * local → sort → top_k
"""
import json
import os
import torch
import numpy as np
import faiss
import torch.nn.functional as F

from config  import ClipperConfig
from model   import ClipperModel


class ClipperRetriever:

    def __init__(self, config: ClipperConfig = None):
        self.config = config or ClipperConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Load model ────────────────────────────────────────────
        self.model = ClipperModel(self.config).eval().to(self.device)

        if os.path.exists(self.config.weights_path):
            ckpt = torch.load(
                self.config.weights_path, map_location=self.device
            )
            self.model.load_state_dict(ckpt, strict=False)
            self.model.config.use_custom_modules = True
            print("Loaded trained weights — custom modules: ON")
        else:
            print("Zero-shot mode — no trained weights found")

        # ── Load FAISS index ──────────────────────────────────────
        self.index = faiss.read_index(self.config.index_path)

        # ── Load metadata ─────────────────────────────────────────
        with open(self.config.meta_path) as f:
            self.metadata = [json.loads(l) for l in f]

        # ── Load pre-cached frame embeddings ─────────────────────
        # Shape: (N_clips, T, D) — avoids re-encoding at query time
        self.frame_cache = np.load(self.config.frame_cache_path)

        print(f"Clipper ready — {len(self.metadata)} clips indexed")

    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _encode_query(self, query: str):
        """
        Encodes text query into global and concept vectors.

        Returns:
          text_vec       : (D,)    numpy  → for FAISS
          text_tensor    : [1, D]  torch  → for similarity module
          concept_tensor : [1, Q, D] torch → for local reranking
        """
        tokens = self.model.tokenizer([query]).to(self.device)
        t_emb, c_emb = self.model.encode_text(tokens)
        return (
            t_emb.squeeze(0).cpu().numpy().astype(np.float32),
            t_emb,
            c_emb
        )

    # ─────────────────────────────────────────────────────────────
    def _global_search(self, text_vec: np.ndarray, k: int) -> list:
        """FAISS inner product search → returns list of candidate dicts."""
        scores, ids = self.index.search(text_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if 0 <= idx < len(self.metadata):
                c = dict(self.metadata[idx])
                c["global_score"] = round(float(score), 4)
                c["score"]        = round(float(score), 4)
                c["local_score"]  = None
                results.append(c)
        return results

    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _local_rerank(self, concept_tensor: torch.Tensor,
                      candidates: list) -> list:
        """
        Reranks candidates using Global + Local MaxSim fusion.

        concept_tensor : [1, Q, D]
        candidates     : list of dicts with clip_id and global_score
        """
        ids         = [c["clip_id"] for c in candidates]

        # Load pre-cached frame embeddings for these clips
        frame_batch = torch.tensor(
            self.frame_cache[ids], dtype=torch.float32
        ).to(self.device)                                     # [K, T, D]

        # Reconstruct clip embeddings from frame means
        clip_vecs = F.normalize(
            torch.tensor(
                np.stack([self.frame_cache[i].mean(0) for i in ids]),
                dtype=torch.float32
            ).to(self.device), dim=-1
        )                                                     # [K, D]

        # Derive text_embed from concept_tensor mean
        text_tensor = F.normalize(
            concept_tensor.mean(dim=1), dim=-1
        )                                                     # [1, D]

        # Full fused similarity: [1, K]
        fused = self.model.similarity(
            text_tensor, clip_vecs, concept_tensor, frame_batch
        ).squeeze(0).cpu().numpy()

        # Compute individual local score per clip for transparency
        for i, c in enumerate(candidates):
            c["score"] = round(float(fused[i]), 4)
            c["local_score"] = round(
                float(
                    self.model.similarity.local_similarity(
                        concept_tensor,
                        frame_batch[i:i+1]
                    ).item()
                ), 4
            )

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    # ─────────────────────────────────────────────────────────────
    def search(self, query: str,
               top_k:  int  = None,
               rerank: bool = True) -> list:
        """
        End-to-end text → clip retrieval.

        Args:
          query  : natural language description
          top_k  : number of results (default: config.top_k)
          rerank : apply local MaxSim reranking (default: True)

        Returns list of dicts:
          {
            clip_id      : int    — unique clip index in the index
            video_id     : str    — stem of the video filename
            filename     : str    — full filename
            video_path   : str    — path to the source video file
            t_start      : float  — clip start in seconds
            t_end        : float  — clip end in seconds
            score        : float  — final fused similarity score
            global_score : float  — CLIP cosine similarity (global)
            local_score  : float  — MaxSim local similarity
          }
        """
        k       = top_k or self.config.top_k
        fetch_k = self.config.rerank_top_k if rerank else k

        text_vec, text_tensor, concept_tensor = self._encode_query(query)
        candidates = self._global_search(text_vec, fetch_k)

        if not candidates:
            return []

        if rerank:
            candidates = self._local_rerank(concept_tensor, candidates)

        return candidates[:k]
