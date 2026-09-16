"""
HSTU: Hierarchical Sequential Transduction Units.

A from-scratch PyTorch implementation of the sequence encoder proposed in
"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers
for Generative Recommendations" (Zhai et al., ICML 2024). HSTU replaces
softmax self-attention with a normalized, pointwise (SiLU) attention that is
cheap at long sequence lengths and multiplicatively gates its output --
which is what makes it well suited to generative / retrieval-style
recommendation instead of NLP-style token generation.

One HSTU layer, given input X (B, L, D):
    U, V, Q, K = split(SiLU(X @ W_uvqk))            # pointwise projection
    A          = SiLU(Q @ K^T + rel_pos_bias + rel_time_bias) / L   # pointwise "attention"
    O          = LayerNorm(A @ V)
    Y          = (O * U) @ W_out                     # gate with U, project
    X_out      = X + Y                                # residual

Relative position bias: a learned embedding indexed by clipped (i - j).
Relative time bias: a learned embedding indexed by log-bucketed |t_i - t_j|.
Both are added to the QK^T scores before the SiLU pointwise nonlinearity,
so the model can attend more/less depending on how far apart and how long
ago two interactions were -- important for clickstream data where the gap
between hits varies from seconds (same session) to weeks (return visit).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bucketize_time_delta(delta_seconds: torch.Tensor, num_buckets: int = 64) -> torch.Tensor:
    """Log-scale bucketing of a (nonnegative) time gap in seconds into
    [0, num_buckets - 1], matching the log-bucket relative-time-bias scheme
    used in HSTU / TransAct-style sequential recommenders."""
    delta = torch.clamp(delta_seconds, min=0).float()
    log_delta = torch.log1p(delta)
    max_log = math.log1p(60 * 60 * 24 * 365)  # cap at ~1 year
    bucket = (log_delta / max_log * (num_buckets - 1)).long()
    return torch.clamp(bucket, 0, num_buckets - 1)


class RelativeAttentionBias(nn.Module):
    """Produces an additive (L, L) bias combining relative position and
    relative (log-bucketed) time."""

    def __init__(self, num_heads: int, max_positions: int = 512, num_time_buckets: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.max_positions = max_positions
        self.num_time_buckets = num_time_buckets
        # relative position index range: [-(max_positions-1), max_positions-1]
        self.pos_bias = nn.Embedding(2 * max_positions - 1, num_heads)
        self.time_bias = nn.Embedding(num_time_buckets, num_heads)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.time_bias.weight)

    def forward(self, seq_len: int, timestamps: torch.Tensor) -> torch.Tensor:
        """timestamps: (B, L) unix seconds. Returns (B, H, L, L) bias."""
        device = timestamps.device
        pos = torch.arange(seq_len, device=device)
        rel_pos = pos[None, :] - pos[:, None]  # (L, L), rel_pos[i, j] = j - i
        rel_pos = torch.clamp(rel_pos, -(self.max_positions - 1), self.max_positions - 1)
        rel_pos_idx = rel_pos + (self.max_positions - 1)
        pos_bias = self.pos_bias(rel_pos_idx)  # (L, L, H)
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, L, L)

        delta = (timestamps[:, None, :] - timestamps[:, :, None]).abs()  # (B, L, L)
        time_idx = bucketize_time_delta(delta, self.num_time_buckets)  # (B, L, L)
        time_bias = self.time_bias(time_idx)  # (B, L, L, H)
        time_bias = time_bias.permute(0, 3, 1, 2)  # (B, H, L, L)

        return pos_bias + time_bias


class HSTULayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.2, max_positions: int = 512, num_time_buckets: int = 64):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # one fused projection producing U, V, Q, K (each size `dim`)
        self.uvqk_proj = nn.Linear(dim, 4 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = RelativeAttentionBias(num_heads, max_positions, num_time_buckets)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        timestamps: (B, L) unix seconds, used for relative time bias
        attn_mask: (B, 1, L, L) additive mask, -inf-ish for disallowed
                   (causal + padding) positions, 0 elsewhere. Since we do not
                   use softmax, "-inf" would break SiLU, so this mask is
                   instead a {0, 1} multiplicative keep-mask (see below).
        """
        B, L, D = x.shape
        h = self.input_norm(x)
        uvqk = F.silu(self.uvqk_proj(h))
        u, v, q, k = uvqk.chunk(4, dim=-1)

        def split_heads(t):
            return t.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, L, hd)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        scores = scores + self.rel_bias(L, timestamps)

        # HSTU's "pointwise aggregated attention": SiLU instead of softmax,
        # normalized by the (valid) sequence length so magnitudes stay
        # comparable across different amounts of causal history.
        attn = F.silu(scores) * attn_mask  # attn_mask zeroes out disallowed pairs
        valid_counts = attn_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        attn = attn / valid_counts
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        out = self.attn_norm(out)

        gated = out * u
        y = self.out_proj(gated)
        y = self.dropout(y)
        return x + y


class HSTUEncoder(nn.Module):
    """Stack of HSTU layers over an (item, action, side-feature) token
    stream, producing contextualized per-position hidden states.

    Feature composition is split into two towers, mirroring the standard
    "item tower" / "context tower" split used in production recsys (e.g.
    two-tower retrieval models), rather than blending every feature into one
    undifferentiated bag:

      - Item tower (`compute_item_repr`): item id + the item's own static
        content features (section, content type, word count bucket, ...).
        This is a property of the ARTICLE alone, independent of who is
        reading it or how -- so it's also what gets exported as the
        standalone, content-aware article embedding (see
        generate_embeddings.py), not just a bare item-id lookup.
      - Context tower: the per-hit action plus how-this-happened features
        (device, referrer/channel, subscriber status, ...), which only make
        sense attached to a specific interaction, not the article itself.

    Both towers feed into the same HSTU attention stack; the split only
    changes how article-level vs. interaction-level information is
    composed before that point.
    """

    def __init__(
        self,
        num_items: int,
        num_actions: int,
        item_cat_vocab_sizes: dict[str, int],
        context_cat_vocab_sizes: dict[str, int],
        numeric_bucket_vocab_sizes: dict[str, int],
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items, dim, padding_idx=0)
        self.action_emb = nn.Embedding(num_actions, dim, padding_idx=0)

        self.item_cat_embs = nn.ModuleDict(
            {name: nn.Embedding(size, dim // 4, padding_idx=0) for name, size in item_cat_vocab_sizes.items()}
        )
        self.context_cat_embs = nn.ModuleDict(
            {name: nn.Embedding(size, dim // 4, padding_idx=0) for name, size in context_cat_vocab_sizes.items()}
        )
        # Numeric hit features (word count, image count, ...) are static
        # per article, so they belong in the item tower, not the context
        # tower. They're quantile-bucketized upstream
        # (preprocessing.NumericBucketizer) and embedded exactly like any
        # other categorical feature -- this lets the model learn
        # non-linear/threshold effects per bucket instead of assuming a
        # linear relationship.
        self.numeric_bucket_embs = nn.ModuleDict(
            {name: nn.Embedding(size, dim // 4, padding_idx=0) for name, size in numeric_bucket_vocab_sizes.items()}
        )

        item_side_dim = dim // 4 * (len(item_cat_vocab_sizes) + len(numeric_bucket_vocab_sizes))
        self.item_side_proj = nn.Linear(item_side_dim, dim) if item_side_dim > 0 else None

        context_side_dim = dim // 4 * len(context_cat_vocab_sizes)
        self.context_side_proj = nn.Linear(context_side_dim, dim) if context_side_dim > 0 else None

        self.input_norm = nn.LayerNorm(dim)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [HSTULayer(dim, num_heads, dropout=dropout, max_positions=max_seq_len) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def compute_item_repr(self, item_ids: torch.Tensor, item_cat: dict, numeric: dict) -> torch.Tensor:
        """Content-aware article representation: item id embedding plus all
        of that article's own static features (section, content type,
        word/image/link/video count buckets, ...), independent of any
        specific hit or user. Shape-agnostic in the non-batch dims, so it
        works both inside forward() (item_ids: (B, L)) and for standalone
        per-article export (item_ids: (N, 1))."""
        repr_ = self.item_emb(item_ids)
        side_parts = [emb(item_cat[name]) for name, emb in self.item_cat_embs.items()]
        side_parts += [emb(numeric[name]) for name, emb in self.numeric_bucket_embs.items()]
        if side_parts and self.item_side_proj is not None:
            repr_ = repr_ + self.item_side_proj(torch.cat(side_parts, dim=-1))
        return repr_

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch keys:
          item_ids:   (B, L) long
          action_ids: (B, L) long
          timestamps: (B, L) long (unix seconds)
          padding_mask: (B, L) bool, True where token is real (not PAD)
          item_cat:  {name: (B, L) long}
          ctx_cat:   {name: (B, L) long}
          numeric:   {name: (B, L) long}  -- quantile bucket indices
        Returns: (B, L, D) contextualized hidden states.
        """
        item_ids = batch["item_ids"]
        action_ids = batch["action_ids"]
        padding_mask = batch["padding_mask"]
        B, L = item_ids.shape

        x = self.compute_item_repr(item_ids, batch["item_cat"], batch["numeric"]) + self.action_emb(action_ids)

        context_parts = [emb(batch["ctx_cat"][name]) for name, emb in self.context_cat_embs.items()]
        if context_parts and self.context_side_proj is not None:
            x = x + self.context_side_proj(torch.cat(context_parts, dim=-1))

        x = self.input_dropout(self.input_norm(x))

        # causal mask (can't see future) AND padding mask (can't attend to
        # pad tokens), expressed as a {0,1} keep-mask of shape (B, 1, L, L).
        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        key_valid = padding_mask.unsqueeze(1).expand(B, L, L)  # can attend to key j only if j is real
        keep = (causal.unsqueeze(0) & key_valid).unsqueeze(1).float()  # (B, 1, L, L)

        for layer in self.layers:
            x = layer(x, batch["timestamps"].float(), keep)

        x = self.final_norm(x)
        return x
