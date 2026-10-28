# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.nn as nn
from einops import repeat
from torch.utils.checkpoint import checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from functools import partial
from .attention import flash_attention
from .model import WanModel as BaseWanModel
from utils.training_correctness import _split_modality_time_embeddings

__all__ = ["WanModel", "WanAttentionBlock"]


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # modality-specific layers (RGB = 0, ray = 1, depth = 2)
        self.q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.k = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.v = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.o = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        if qk_norm:
            self.norm_q = nn.ModuleList([WanRMSNorm(dim, eps=eps) for _ in range(3)])
            self.norm_k = nn.ModuleList([WanRMSNorm(dim, eps=eps) for _ in range(3)])
        else:
            self.norm_q = nn.ModuleList([nn.Identity() for _ in range(3)])
            self.norm_k = nn.ModuleList([nn.Identity() for _ in range(3)])

        # Modality embedding: 0 for RGB (zeros), 1 for ray, 2 for depth
        ray_emb_init = torch.randn(3, 1, 1, dim)
        ray_emb_init[0].fill_(0)
        self.ray_embedding = nn.Parameter(ray_emb_init)

    def forward(self, x, L, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x (Tensor): Shape [B, 3L, C] — layout [rgb | ray (fused) | depth]
            L (int): RGB token count (= ray = depth), hoisted out of the forward
                (computed once at top-level model.forward) so this block has no
                .item()/data-dependent control flow -> torch.compile-able.
            seq_lens (Tensor): Shape [B]
            grid_sizes (Tensor): Shape [B, 3], (F, H, W) per-modality grid
            freqs (Tensor): RoPE freqs
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        x_rgb   = x[:, :L,    :]
        x_ray   = x[:, L:2*L, :]
        x_depth = x[:, 2*L:3*L, :]

        # Modality-specific projections: stream 0 for rgb, stream 1 for fused ray, stream 2 for depth
        q_rgb   = self.norm_q[0](self.q[0](x_rgb  )).view(b, L, n, d)
        k_rgb   = self.norm_k[0](self.k[0](x_rgb  )).view(b, L, n, d)
        v_rgb   = self.v[0](x_rgb  ).view(b, L, n, d)

        q_ray   = self.norm_q[1](self.q[1](x_ray)).view(b, L, n, d)
        k_ray   = self.norm_k[1](self.k[1](x_ray)).view(b, L, n, d)
        v_ray   = self.v[1](x_ray).view(b, L, n, d)

        q_depth = self.norm_q[2](self.q[2](x_depth)).view(b, L, n, d)
        k_depth = self.norm_k[2](self.k[2](x_depth)).view(b, L, n, d)
        v_depth = self.v[2](x_depth).view(b, L, n, d)

        # (seq_lens == 3*L by construction; assert removed for torch.compile.)
        q_rgb   = rope_apply(q_rgb,   grid_sizes, freqs)
        q_ray   = rope_apply(q_ray,   grid_sizes, freqs)
        q_depth = rope_apply(q_depth, grid_sizes, freqs)
        k_rgb   = rope_apply(k_rgb,   grid_sizes, freqs)
        k_ray   = rope_apply(k_ray,   grid_sizes, freqs)
        k_depth = rope_apply(k_depth, grid_sizes, freqs)

        new_q = torch.cat([q_rgb, q_ray, q_depth], dim=1)
        new_k = torch.cat([k_rgb, k_ray, k_depth], dim=1)
        new_v = torch.cat([v_rgb, v_ray, v_depth], dim=1)

        x = flash_attention(
            q=new_q,
            k=new_k,
            v=new_v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        x = x.flatten(2)
        x_rgb   = self.o[0](x[:, :L,    :])
        x_ray   = self.o[1](x[:, L:2*L, :])
        x_depth = self.o[2](x[:, 2*L:3*L, :])
        return torch.cat([x_rgb, x_ray, x_depth], dim=1)


class WanT2VCrossAttention(WanSelfAttention):
    """Text cross-attention. SHARED across the three modality streams -- unlike the
    self-attention / norms / FFNs above, which are genuinely per-modality (MoT).

    It inherits WanSelfAttention.__init__, which allocates 3 experts for q/k/v/o.
    The forward below only ever indexes [0], so experts 1 and 2 were dead weight:
    8 tensors x dim^2 x 30 blocks = 2.264 B parameters at dim=3072, i.e. 15.3% of the
    14.821 B model, allocated, FSDP-sharded and checkpointed but never receiving a
    gradient.

    We now allocate a single expert. Keys are unchanged for the weights that were
    actually used (they stay `cross_attn.{q,k,v,o}.0.*`), so old checkpoints still load
    -- the loader uses strict=False, and the stale `.1`/`.2` entries are simply ignored.
    Numerics are bit-identical: expert 0 is exactly what ran before.

    Note this leaves the text conditioning modality-agnostic. Making it truly MoT (a
    per-modality q/o with shared k/v from the text context) is a real architecture
    change that would need retraining -- experts 1 and 2 hold random init, never
    trained -- so it is deliberately NOT done here.
    """

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        # Replace the 3-expert lists with 1-element lists, preserving the `.0` key names.
        self.q = nn.ModuleList([self.q[0]])
        self.k = nn.ModuleList([self.k[0]])
        self.v = nn.ModuleList([self.v[0]])
        self.o = nn.ModuleList([self.o[0]])
        self.norm_q = nn.ModuleList([self.norm_q[0]])
        self.norm_k = nn.ModuleList([self.norm_k[0]])
        # ray_embedding is a self-attention-only modality bias; cross-attn never uses it.
        del self.ray_embedding

    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q[0](self.q[0](x)).view(b, -1, n, d)
        k = self.norm_k[0](self.k[0](context)).view(b, -1, n, d)
        v = self.v[0](context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o[0](x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q[0](self.q[0](x)).view(b, -1, n, d)
        k = self.norm_k[0](self.k[0](context)).view(b, -1, n, d)
        v = self.v[0](context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o[0](x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # modality-specific norms and FFNs (RGB = 0, ray = 1, depth = 2)
        self.norm1 = nn.ModuleList([WanLayerNorm(dim, eps) for _ in range(3)])
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = nn.ModuleList([WanLayerNorm(dim, eps) for _ in range(3)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ffn_dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(ffn_dim, dim),
                )
                for _ in range(3)
            ]
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        L,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x (Tensor): [B, 3L, C] — layout [rgb | ray (fused) | depth]
            e (Tensor): [B, F, 6, C] — time modulation per real frame (F=3*T_patch)
            temporal_mask (Tensor): [B, 3L], 0=RGB, 1=ray, 2=depth
            seq_lens (Tensor): [B]
            grid_sizes (Tensor): [B, 3] — (F, H, W) per-modality
            freqs (Tensor): RoPE freqs
        """
        tokens_per_frame = x.shape[1] // e.shape[1]
        e = self.modulation[:, None] + e
        e = repeat(e, "b f1 n c -> n b (f1 f2) c", f2=tokens_per_frame)

        x_rgb   = x[:, :L,    :]
        x_ray   = x[:, L:2*L, :]
        x_depth = x[:, 2*L:3*L, :]

        x_norm1 = torch.empty_like(x)
        x_norm1[:, :L,    :] = self.norm1[0](x_rgb)
        x_norm1[:, L:2*L, :] = self.norm1[1](x_ray)
        x_norm1[:, 2*L:3*L, :] = self.norm1[2](x_depth)

        y = self.self_attn(
            x_norm1 * (1 + e[1]) + e[0], L, seq_lens, grid_sizes, freqs
        )
        x = x + y * e[2]

        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)

            # L from enclosing scope (passed into forward); no .item().
            x_rgb   = x[:, :L,    :]
            x_ray   = x[:, L:2*L, :]
            x_depth = x[:, 2*L:3*L, :]

            x_norm2 = torch.empty_like(x)
            x_norm2[:, :L,    :] = self.norm2[0](x_rgb)
            x_norm2[:, L:2*L, :] = self.norm2[1](x_ray)
            x_norm2[:, 2*L:3*L, :] = self.norm2[2](x_depth)

            e3, e4, e5 = e[3], e[4], e[5]

            y_rgb   = self.ffn[0](x_norm2[:, :L,    :] * (1 + e4[:, :L,    :]) + e3[:, :L,    :])
            y_ray   = self.ffn[1](x_norm2[:, L:2*L, :] * (1 + e4[:, L:2*L, :]) + e3[:, L:2*L, :])
            y_depth = self.ffn[2](x_norm2[:, 2*L:3*L, :] * (1 + e4[:, 2*L:3*L, :]) + e3[:, 2*L:3*L, :])

            y_full = torch.empty_like(x)
            y_full[:, :L,    :] = y_rgb
            y_full[:, L:2*L, :] = y_ray
            y_full[:, 2*L:3*L, :] = y_depth

            x = x + y_full * e5
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        tokens_per_frame = x.shape[1] // e.shape[1]
        e = self.modulation[:, None] + e[:, :, None]
        e = repeat(e, "b f1 n c -> n b (f1 f2) c", f2=tokens_per_frame)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone with Mixture-of-Transformers for RGB + Plücker raymaps.

    Token layout: [rgb_tokens | ray_d_tokens | ray_m_tokens]
    Stream 0 (RGB): processes rgb tokens.
    Stream 1 (Ray): processes both ray_d and ray_m tokens with shared weights.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        ray_in_dim=24,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        if model_type == "ti2v": 
            model_type = "t2v"

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.ray_in_dim = ray_in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.gradient_checkpointing_indices = []

        # RGB patch embedding
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        # Ray patch embedding — takes the channel-concat of ray_d + ray_m (the real
        # packed channels only, no zero padding) to produce a single fused token block.
        # ray_in_dim = 2 * (vae_stride[0] * 3): 3 Plücker channels per ray stream,
        # vae_stride[0] temporal frames stacked into channels, two streams concatenated.
        # Trained from scratch (Xavier), NOT tiled/copied from the RGB embedding.
        self.patch_embedding_ray = nn.Conv3d(
            ray_in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        # Depth patch embedding
        self.patch_embedding_depth = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads,
                    window_size, qk_norm, cross_attn_norm, eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.head_rgb = Head(dim, out_dim, patch_size, eps)
        self.head_ray = Head(dim, out_dim, patch_size, eps)
        self.head_mod = Head(dim, out_dim, patch_size, eps)
        self.head_depth = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.freqs = self._build_freqs()

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()

    def _build_freqs(self):
        # RoPE frequency table. Kept as a method (not inlined in __init__) so it can be
        # rebuilt after a meta-device construction: self.freqs is a plain attribute, not
        # a Parameter or registered buffer, so it is neither in state_dict nor moved by
        # .to(). Under `with torch.device("meta")` it would be created on the meta device
        # and forward's `self.freqs.to(device)` would raise "Cannot copy out of meta
        # tensor". The resume path rebuilds it on a real device via this method.
        d = self.dim // self.num_heads
        return torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def gradient_checkpointing_enable(self, p=0):
        cut_off = 0.5
        indices = []
        for i in range(self.num_layers):
            if (i + 1) * p >= cut_off:
                cut_off += 1
                indices.append(i)
        self.gradient_checkpointing_indices = tuple(indices)

    def forward(
        self,
        x,
        rays,
        depth,
        modality_mask,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Args:
            x (Tensor): RGB latents [B, C_in, T, H, W]
            rays (Tensor): Channel-concat ray latents [B, ray_in_dim, T, H, W]
                           — first ray_in_dim/2 channels are ray_d, last half ray_m
                           (real packed channels only, no zero padding)
            modality_mask (Tensor): [B, 3*T] — 0=RGB, 1=ray (fused), 2=depth
            t (Tensor): timesteps [B] or [B, 3*T]
            context (List[Tensor]): text embeddings
            seq_len (int): max sequence length
            clip_fea, y: optional image conditioning (i2v)

        Returns:
            Tensor: [B, C_out, 4*T, H, W] (rgb | ray_d | ray_m | depth)
        """
        n_frames = x.shape[2]
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embeddings — each processed independently as a list of single-sample tensors
        x       = [self.patch_embedding(u.unsqueeze(0))       for u in x]
        rays    = [self.patch_embedding_ray(u.unsqueeze(0))   for u in rays]
        depth   = [self.patch_embedding_depth(u.unsqueeze(0)) for u in depth]

        # original_grid_sizes: per-modality (T_patch, H_patch, W_patch) — used for RoPE
        original_grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )

        # Concatenate along temporal dimension: [rgb | ray | depth] = 3*T_patch
        x = [torch.cat([u, r, d], dim=2) for u, r, d in zip(x, rays, depth)]

        # grid_sizes for unpatchify must reflect the 4*T_patch output layout
        # (we duplicate ray tokens into ray_d and ray_m via two heads after the
        # transformer). Internally the transformer processes only 3*T_patch.
        grid_sizes_internal = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        grid_sizes = grid_sizes_internal.clone()
        T_patch = original_grid_sizes[0, 0].item()
        grid_sizes[:, 0] = grid_sizes_internal[:, 0] + T_patch  # 3T → 4T frames for unpatchify

        # Expand modality_mask [B, 3*T_lat] → [B, 3*T_lat*H_patch*W_patch]
        batch_size = modality_mask.shape[0]
        h, w = x[0].shape[3], x[0].shape[4]
        temporal_mask = repeat(modality_mask, "b f -> b (f h w)", h=h, w=w)

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        x = [
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ]
        x = torch.cat(x)

        # Pad temporal_mask to seq_len
        padding_size = seq_len - temporal_mask.size(1)
        if padding_size > 0:
            temporal_mask = torch.cat([
                temporal_mask,
                torch.zeros(batch_size, padding_size, dtype=temporal_mask.dtype, device=temporal_mask.device),
            ], dim=1)

        # Time embeddings
        t_shape = tuple(t.shape)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        if t.ndim == 2:
            e = e.unflatten(dim=0, sizes=t_shape)
        else:
            e = repeat(e, "b c -> b f c", f=n_frames)
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        # Text context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # L = RGB token count (= ray = depth). Static across blocks/steps; compute ONCE
        # (single .item()) and pass as a python int so each block is .item()-free -> compile-able.
        L = int((temporal_mask[0] == 0).sum().item())
        kwargs = dict(
            e=e0,
            L=L,
            seq_lens=seq_lens,
            grid_sizes=original_grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        for i, block in enumerate(self.blocks):
            block = partial(block, **kwargs)
            if i in self.gradient_checkpointing_indices:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Internal layout is 3*L tokens (rgb | ray | depth). head_ray and head_mod
        # both operate on the same fused ray tokens, producing ray_d and ray_m
        # outputs; concatenated temporally to give the 4*L token layout that
        # unpatchify (and downstream loss/split logic) expects.
        # L already computed above (RGB-token count); reused here for the output heads.
        #
        # Per-modality timestep modulation. `e` holds 3*T_patch per-frame timesteps in
        # rgb|ray|depth order (t_model = cat([t_rgb, t_ray, t_depth]) in the algorithm),
        # but each head processes only ONE modality block of L = T_patch*h*w tokens. The
        # Head reconstructs L via tokens_per_frame = L // e.shape[1], which requires
        # e.shape[1] == T_patch (frame-major token order). Passing the full 3*T_patch e
        # only divides L evenly when h*w is a multiple of 3 (true for the 1.3B 30x40 ->
        # h*w=300 latent, FALSE for the 5B 15x20 -> h*w=70 latent: 980//42=23 -> 966!=980).
        # Slice e to each modality's own T_patch frames so the modulation is both correct
        # (frame-major) and shape-consistent for any latent grid.
        e_rgb, e_ray, e_depth = _split_modality_time_embeddings(
            e, per_modality_frames=int(T_patch)
        )
        x_rgb_out   = self.head_rgb(x[:, :L,    :], e_rgb)
        x_ray_d_out = self.head_ray(x[:, L:2*L, :], e_ray)
        x_ray_m_out = self.head_mod(x[:, L:2*L, :], e_ray)
        x_depth_out = self.head_depth(x[:, 2*L:3*L, :], e_depth)
        x = torch.cat([x_rgb_out, x_ray_d_out, x_ray_m_out, x_depth_out], dim=1)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        # patch_embedding_ray: trained from scratch with the same Xavier init the RGB
        # patch embedding uses (no tiling/copy of RGB weights). It takes the real
        # ray_in_dim channels directly (no zero padding).
        nn.init.xavier_uniform_(self.patch_embedding_ray.weight.flatten(1))

        with torch.no_grad():
            self.patch_embedding_depth.weight.copy_(self.patch_embedding.weight)
            if self.patch_embedding_depth.bias is not None and self.patch_embedding.bias is not None:
                self.patch_embedding_depth.bias.copy_(self.patch_embedding.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.head_rgb.head.weight)
        nn.init.zeros_(self.head_ray.head.weight)
        nn.init.zeros_(self.head_mod.head.weight)
        nn.init.zeros_(self.head_depth.head.weight)

    @staticmethod
    @torch.no_grad()
    def _copy_self_attn_from_base(target_attn, base_attn):
        """Copy single-branch weights into both modality slots of a MoT attention."""
        for name in ["q", "k", "v", "o"]:
            base_linear = getattr(base_attn, name)
            module_list = getattr(target_attn, name)
            for i in range(len(module_list)):
                module_list[i].load_state_dict(base_linear.state_dict())
        for name in ["norm_q", "norm_k"]:
            base_norm = getattr(base_attn, name)
            module_list = getattr(target_attn, name)
            for i in range(len(module_list)):
                module_list[i].load_state_dict(base_norm.state_dict())

    @torch.no_grad()
    def load_from_t2v_pretrained(self, base_model: "BaseWanModel"):
        """Initialize this MoT backbone from a pretrained RGB-only Wan T2V model."""
        assert (
            self.patch_size == base_model.patch_size
            and self.dim == base_model.dim
            and self.ffn_dim == base_model.ffn_dim
            and self.num_heads == base_model.num_heads
            and self.num_layers == base_model.num_layers
        ), "MoT backbone is not config-compatible with base T2V model."

        self.patch_embedding.load_state_dict(base_model.patch_embedding.state_dict())
        self.text_embedding.load_state_dict(base_model.text_embedding.state_dict())
        self.time_embedding.load_state_dict(base_model.time_embedding.state_dict())
        self.time_projection.load_state_dict(base_model.time_projection.state_dict())
        self.head_rgb.load_state_dict(base_model.head.state_dict())
        self.head_ray.load_state_dict(base_model.head.state_dict())
        self.head_mod.load_state_dict(base_model.head.state_dict())
        self.head_depth.load_state_dict(base_model.head.state_dict())

        if hasattr(base_model, "img_emb") and hasattr(self, "img_emb"):
            self.img_emb.load_state_dict(base_model.img_emb.state_dict())

        self.init_ray_embedding_from_scratch()
        self.init_depth_embedding_from_rgb()

        if hasattr(self, "freqs") and hasattr(base_model, "freqs"):
            self.freqs.copy_(base_model.freqs)

        assert len(self.blocks) == len(base_model.blocks)
        for mot_block, base_block in zip(self.blocks, base_model.blocks):
            mot_block.modulation.copy_(base_block.modulation)
            for i in range(len(mot_block.norm1)):
                mot_block.norm1[i].load_state_dict(base_block.norm1.state_dict())
                mot_block.norm2[i].load_state_dict(base_block.norm2.state_dict())
                mot_block.ffn[i].load_state_dict(base_block.ffn.state_dict())
            mot_block.norm3.load_state_dict(base_block.norm3.state_dict())
            self._copy_self_attn_from_base(mot_block.self_attn, base_block.self_attn)
            self._copy_self_attn_from_base(mot_block.cross_attn, base_block.cross_attn)
            # Zero out modality embeddings so initial bias is zero
            if hasattr(mot_block.self_attn, "ray_embedding"):
                mot_block.self_attn.ray_embedding.zero_()
            if hasattr(mot_block.cross_attn, "ray_embedding"):
                mot_block.cross_attn.ray_embedding.zero_()

    
    @torch.no_grad()
    def init_depth_embedding_from_rgb(self):
        """Initialize patch_embedding_depth from patch_embedding weights."""
        if self.patch_embedding_depth.weight.is_meta:
            self.patch_embedding_depth.weight = nn.Parameter(
                self.patch_embedding.weight.clone()
            )
        else:
            self.patch_embedding_depth.weight.copy_(self.patch_embedding.weight)

        if self.patch_embedding_depth.bias is not None and self.patch_embedding.bias is not None:
            if self.patch_embedding_depth.bias.is_meta:
                self.patch_embedding_depth.bias = nn.Parameter(
                    self.patch_embedding.bias.clone()
                )
            else:
                self.patch_embedding_depth.bias.copy_(self.patch_embedding.bias)
    
    
    @classmethod
    def from_t2v_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "WanModel":
        """Build a MoT WanModel from a pretrained RGB-only Wan T2V checkpoint."""
        base_model = BaseWanModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        mot_model = cls.from_config(base_model.config)
        mot_model.load_from_t2v_pretrained(base_model)
        return mot_model

    @torch.no_grad()
    def init_ray_embedding_from_scratch(self):
        """Initialize patch_embedding_ray (in-ch ray_in_dim) from scratch with the
        same Xavier init the RGB patch embedding uses. No tiling/copy of RGB weights;
        the ray stream is trained from scratch. Bias is left at its default Conv3d
        init (matching RGB, whose bias is not re-initialized)."""
        w = self.patch_embedding_ray.weight
        if w.is_meta:
            new_w = torch.empty(
                w.shape, dtype=self.patch_embedding.weight.dtype, device="cpu"
            )
            nn.init.xavier_uniform_(new_w.flatten(1))
            self.patch_embedding_ray.weight = nn.Parameter(new_w)
        else:
            nn.init.xavier_uniform_(w.flatten(1))
