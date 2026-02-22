"""
All custom modules for the Clipper architecture:

  1. MotionEnhancementModule    — depthwise temporal conv + sigmoid gating
  2. TextGuidedExcitationModule — text-guided frame relevance weighting
  3. SoftmaxAggregationModule   — discriminative attention pooling
  4. QueryModule                — 8 concept vectors via cross-attention
  5. SimilarityModule           — alpha * global + (1-alpha) * local (MaxSim)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  1. Motion Enhancement Module
# ══════════════════════════════════════════════════════════════════════
class MotionEnhancementModule(nn.Module):
    """
    Captures inter-frame temporal dynamics on top of frozen CLIP features.

    Architecture:
      - Depthwise 1-D temporal convolution slides over T frames
        and detects local motion patterns between adjacent frames.
      - A learned sigmoid gate controls how much motion signal
        is blended back into the original frame embeddings.
      - LayerNorm stabilizes the residual output.

    Input  : frame_embeds [B, T, D]
    Output : enhanced frame_embeds [B, T, D]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Depthwise: one 1D filter per channel → D × kernel parameters only
        self.temporal_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1,
            groups=embed_dim
        )
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        # frame_embeds: [B, T, D]
        x      = frame_embeds.transpose(1, 2)                    # [B, D, T]
        motion = self.temporal_conv(x).transpose(1, 2)           # [B, T, D]
        gate   = torch.sigmoid(self.gate(frame_embeds))          # [B, T, D]
        return self.norm(frame_embeds + gate * motion)           # [B, T, D]


# ══════════════════════════════════════════════════════════════════════
#  2. Text-Guided Excitation Module
# ══════════════════════════════════════════════════════════════════════
class TextGuidedExcitationModule(nn.Module):
    """
    Weights each frame by its relevance to the text query.

    The text query cross-attends to frame embeddings through a
    shared bottleneck projection, producing a scalar relevance
    weight per frame. Frames semantically aligned with the query
    get amplified; irrelevant frames are suppressed.

    Falls back to self-excitation (no text_embed) during offline
    indexing where the query is unavailable.

    Input  : frame_embeds [B, T, D], text_embed [B, D] (optional)
    Output : weighted_frames [B, T, D], weights [B, T]
    """

    def __init__(self, embed_dim: int, reduction: int = 4):
        super().__init__()
        r = embed_dim // reduction            # bottleneck dimension
        self.text_proj  = nn.Linear(embed_dim, r)
        self.frame_proj = nn.Linear(embed_dim, r)
        self.weight_out = nn.Linear(r, 1)

    def forward(self, frame_embeds: torch.Tensor,
                text_embed: torch.Tensor = None):
        # frame_embeds: [B, T, D]
        # text_embed:   [B, D]  (optional)
        f = self.frame_proj(frame_embeds)                        # [B, T, r]

        if text_embed is not None:
            t           = self.text_proj(text_embed).unsqueeze(1)  # [B, 1, r]
            interaction = f + t.expand_as(f)                       # [B, T, r]
        else:
            interaction = f                                        # self-excite

        weights         = torch.sigmoid(
            self.weight_out(torch.tanh(interaction))
        ).squeeze(-1)                                              # [B, T]
        weighted_frames = frame_embeds * weights.unsqueeze(-1)    # [B, T, D]
        return weighted_frames, weights


# ══════════════════════════════════════════════════════════════════════
#  3. Softmax Aggregation Module
# ══════════════════════════════════════════════════════════════════════
class SoftmaxAggregationModule(nn.Module):
    """
    Replaces naive mean-pooling with learned attention pooling.

    Learns a scalar relevance score per frame and produces a single
    clip vector as a softmax-weighted sum of frame embeddings.
    Key frames (containing the main action/event) contribute more
    than empty or repetitive frames.

    Input  : weighted_frames [B, T, D]   (output of ExcitationModule)
    Output : clip_embed [B, D]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)

    def forward(self, weighted_frames: torch.Tensor) -> torch.Tensor:
        # weighted_frames: [B, T, D]
        scores = self.attn_proj(weighted_frames).squeeze(-1)     # [B, T]
        attn   = F.softmax(scores, dim=-1)                       # [B, T]
        return (weighted_frames * attn.unsqueeze(-1)).sum(1)     # [B, D]


# ══════════════════════════════════════════════════════════════════════
#  4. Query Module — 8 Concept Vectors via Cross-Attention
# ══════════════════════════════════════════════════════════════════════
class QueryModule(nn.Module):
    """
    Expands one CLIP text embedding [B, D] into Q concept vectors [B, Q, D].

    Cross-Attention mechanism:
      Q (query)  = Q learnable slot parameters  [B, Q, D]
      K (key)    = text_embed                   [B, 1, D]
      V (value)  = text_embed                   [B, 1, D]

    Each slot independently attends to the text embedding and extracts
    a different semantic aspect (subject, action, scene, attribute, etc.).
    Specialization emerges from contrastive training — not hard-coded.

    Residual connection keeps each slot's identity even after attending
    to the same single text token. Without it all Q outputs collapse.

    Input  : text_embed [B, D]
    Output : concept_embeds [B, Q, D]
    """

    def __init__(self, embed_dim: int, num_queries: int = 8,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim   = embed_dim

        # Learnable slot queries — shape [Q, D]
        # Xavier uniform init: more stable than randn for attention
        self.slots = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.xavier_uniform_(self.slots.unsqueeze(0))

        # Multi-head cross-attention: slots attend to text_embed
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )

        # Feed-forward refinement (4x expansion with GELU)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        """
        text_embed : [B, D]
        returns    : concept_embeds [B, Q, D]

        Steps:
          1. Expand slots to batch: [Q, D] → [B, Q, D]
          2. text_embed as KV:      [B, D] → [B, 1, D]
          3. Cross-attention: each slot pulls its semantic aspect from text
          4. Residual + LayerNorm
          5. FFN + Residual + LayerNorm
        """
        B = text_embed.shape[0]

        # Step 1 — Expand slots to batch
        queries = self.slots.unsqueeze(0).expand(B, -1, -1)   # [B, Q, D]

        # Step 2 — Text embedding as single-token Key & Value
        kv = text_embed.unsqueeze(1)                           # [B, 1, D]

        # Step 3 — Cross-attention
        attn_out, _ = self.cross_attn(
            query = queries,
            key   = kv,
            value = kv
        )                                                       # [B, Q, D]

        # Step 4 — Residual + LN
        x = self.norm1(attn_out + queries)                     # [B, Q, D]

        # Step 5 — FFN + Residual + LN
        x = self.norm2(x + self.ffn(x))                        # [B, Q, D]

        return x


# ══════════════════════════════════════════════════════════════════════
#  5. Similarity Module — Global + Local Fusion
# ══════════════════════════════════════════════════════════════════════
class SimilarityModule(nn.Module):
    """
    Two-stage similarity scoring fused by alpha weighting:

    Global similarity:
      text_embed [B_t, D]  ×  clip_embed [B_v, D]
      → [B_t, B_v] cosine scores scaled by temperature τ
      → Used for: fast FAISS ANN search

    Local similarity (MaxSim — adapted from ColBERT):
      concept_embeds [B_t, Q, D]  ×  frame_embeds [B_v, T, D]
      Step 1: einsum → [B_t, B_v, Q, T]  all pairwise scores
      Step 2: max over T → [B_t, B_v, Q]  best frame per concept
      Step 3: mean over Q → [B_t, B_v]    average concept scores
      → Used for: reranking top-K candidates

    Fusion:
      sim_fused = alpha * sim_global + (1 - alpha) * sim_local
      alpha = 0.7 → global dominant, local refines top-K
    """

    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        # Learnable log-scale temperature
        # init = log(1/0.07) ≈ 2.659  → exp(2.659) ≈ 14.3
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    def global_similarity(self, text_embed: torch.Tensor,
                          clip_embed: torch.Tensor) -> torch.Tensor:
        """
        text_embed : [B_t, D]  L2-normalized
        clip_embed : [B_v, D]  L2-normalized
        returns    : [B_t, B_v] scaled cosine similarities
        """
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (text_embed @ clip_embed.T)

    def local_similarity(self, concept_embeds: torch.Tensor,
                         frame_embeds: torch.Tensor) -> torch.Tensor:
        """
        concept_embeds : [B_t, Q, D]  L2-normalized concept vectors
        frame_embeds   : [B_v, T, D]  per-frame CLIP vectors

        MaxSim algorithm:
          1. Score every (concept, frame) pair across all (text, video)
          2. Max over T:  each concept finds its best-matching frame
          3. Mean over Q: average across all semantic concepts
        """
        scale = self.logit_scale.exp().clamp(max=100.0)

        # L2-normalize
        nc = F.normalize(concept_embeds.float(), dim=-1)  # [B_t, Q, D]
        nf = F.normalize(frame_embeds.float(),   dim=-1)  # [B_v, T, D]

        # All pairwise concept-frame dot products
        # t=text batch, q=concept idx, d=dim, v=video batch, f=frame idx
        scores = torch.einsum('tqd, vfd -> tvqf', nc, nf) # [B_t, B_v, Q, T]

        # Max over frames: best frame per concept
        max_scores = scores.max(dim=-1).values             # [B_t, B_v, Q]

        # Mean over concepts: average semantic aspect scores
        local_sim  = max_scores.mean(dim=-1)               # [B_t, B_v]

        return scale * local_sim

    def forward(self, text_embed: torch.Tensor,
                clip_embed: torch.Tensor,
                concept_embeds: torch.Tensor = None,
                frame_embeds:   torch.Tensor = None) -> torch.Tensor:
        """
        Full fused similarity: alpha * global + (1 - alpha) * local

        Falls back to global-only if concept_embeds or frame_embeds
        are None (safe for zero-shot mode).

        text_embed     : [B_t, D]
        clip_embed     : [B_v, D]
        concept_embeds : [B_t, Q, D]
        frame_embeds   : [B_v, T, D]
        returns        : [B_t, B_v] fused similarity scores
        """
        g_sim = self.global_similarity(text_embed, clip_embed)

        if (concept_embeds is None
                or frame_embeds is None
                or self.alpha == 1.0):
            return g_sim

        l_sim = self.local_similarity(concept_embeds, frame_embeds)
        return self.alpha * g_sim + (1.0 - self.alpha) * l_sim
