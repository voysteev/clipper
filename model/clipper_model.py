"""
ClipperModel — Full text-to-video clip retrieval architecture.

Text branch:
  raw text → CLIP text encoder (frozen)
           → QueryModule (cross-attention) → 8 concept vectors
           → L2-normalize

Video branch:
  frames   → CLIP ViT encoder (frozen)
           → MotionEnhancementModule
           → TextGuidedExcitationModule
           → SoftmaxAggregationModule
           → L2-normalize → clip vector

Similarity:
  global : text_embed  ↔ clip_embed          (for FAISS)
  local  : concept_embeds ↔ frame_embeds     (for reranking)
  fused  : 0.7 * global + 0.3 * local
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from config import ClipperConfig
from model.modules import (
    MotionEnhancementModule,
    TextGuidedExcitationModule,
    SoftmaxAggregationModule,
    QueryModule,
    SimilarityModule,
)


class ClipperModel(nn.Module):

    def __init__(self, config: ClipperConfig):
        super().__init__()
        self.config = config

        # ── Frozen CLIP backbone ──────────────────────────────────
        self.clip_model, _, self.preprocess = \
            open_clip.create_model_and_transforms(
                config.clip_model, pretrained="openai"
            )
        self.tokenizer = open_clip.get_tokenizer(config.clip_model)

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # ── Trainable custom modules ──────────────────────────────
        D = config.embed_dim

        self.motion      = MotionEnhancementModule(D)
        self.excitation  = TextGuidedExcitationModule(
            D, config.excitation_reduction
        )
        self.aggregation = SoftmaxAggregationModule(D)
        self.query_mod   = QueryModule(
            D,
            num_queries = config.num_query_vectors,
            num_heads   = config.num_attn_heads,
            dropout     = config.query_dropout
        )
        self.similarity  = SimilarityModule(
            alpha = config.similarity_alpha
        )

    # ─────────────────────────────────────────────────────────────
    #  TEXT ENCODING
    # ─────────────────────────────────────────────────────────────
    def encode_text(self, tokens: torch.Tensor):
        """
        tokens         : [B, 77]
        returns:
          text_embed     : [B, D]      global L2-normalized text vector
          concept_embeds : [B, Q, D]   Q L2-normalized concept vectors
        """
        with torch.no_grad():
            raw = self.clip_model.encode_text(tokens).float()  # [B, D]

        if self.config.use_custom_modules:
            concept_embeds = self.query_mod(raw)               # [B, Q, D]
        else:
            # Zero-shot: replicate single text vector as Q identical slots
            concept_embeds = raw.unsqueeze(1).expand(
                -1, self.config.num_query_vectors, -1
            ).clone()

        text_embed     = F.normalize(raw,            dim=-1)
        concept_embeds = F.normalize(concept_embeds, dim=-1)
        return text_embed, concept_embeds

    # ─────────────────────────────────────────────────────────────
    #  VIDEO ENCODING
    # ─────────────────────────────────────────────────────────────
    def encode_video(self, frames: torch.Tensor,
                     text_embed: torch.Tensor = None):
        """
        frames       : [B, T, C, H, W]
        text_embed   : [B, D]  optional — enables text-guided excitation

        returns:
          clip_embed   : [B, D]      global L2-normalized clip vector
          frame_embeds : [B, T, D]   per-frame CLIP vectors (for local sim)
        """
        B, T, C, H, W = frames.shape

        # Step 1 — CLIP ViT encoder (frozen)
        with torch.no_grad():
            raw = self.clip_model.encode_image(
                frames.view(B * T, C, H, W)
            ).float()                                          # [B*T, D]
        frame_embeds = raw.view(B, T, -1)                     # [B, T, D]

        if self.config.use_custom_modules:
            # Step 2 — Motion Enhancement
            frame_embeds = self.motion(frame_embeds)

            # Step 3 — Text-Guided Excitation
            frame_embeds, _ = self.excitation(frame_embeds, text_embed)

            # Step 4 — Softmax Aggregation
            clip_embed = self.aggregation(frame_embeds)

        else:
            # Zero-shot: plain mean pooling
            clip_embed = frame_embeds.mean(dim=1)              # [B, D]

        clip_embed = F.normalize(clip_embed, dim=-1)
        return clip_embed, frame_embeds

    # ─────────────────────────────────────────────────────────────
    #  TRAINING FORWARD — symmetric InfoNCE loss
    # ─────────────────────────────────────────────────────────────
    def forward(self, frames: torch.Tensor, tokens: torch.Tensor):
        """
        Called during fine-tuning only.

        frames : [B, T, C, H, W]
        tokens : [B, 77]

        Returns dict with:
          loss        : scalar contrastive loss
          similarity  : [B, B] similarity matrix
          logit_scale : current temperature value
        """
        text_embed, concept_embeds = self.encode_text(tokens)
        clip_embed, frame_embeds   = self.encode_video(frames, text_embed)

        sim = self.similarity(
            text_embed, clip_embed, concept_embeds, frame_embeds
        )                                                      # [B, B]

        B      = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        # Symmetric cross-entropy (InfoNCE)
        loss = (
            F.cross_entropy(sim,   labels) +
            F.cross_entropy(sim.T, labels)
        ) / 2.0

        return {
            "loss":        loss,
            "similarity":  sim,
            "logit_scale": self.similarity.logit_scale.exp().item()
        }
