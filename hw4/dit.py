# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb

class QueryEmbedder(nn.Module):
    def __init__(
        self,
        transformer_hidden_size,
        conditioning_hidden_size,
        num_queries,
        num_dit_layers,
        num_layers=2,
        n_heads=8,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.transformer_hidden_size = transformer_hidden_size
        self.conditioning_hidden_size = conditioning_hidden_size
        self.num_queries = num_queries
        self.num_dit_layers = num_dit_layers
        self.n_heads = n_heads

        self.query_table = nn.Embedding(num_queries, transformer_hidden_size)

        # Stacked blocks: Self-Attn -> Cross-Attn -> FFN (pre-norm)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict(
                {
                    "attn_norm": nn.LayerNorm(transformer_hidden_size, eps=norm_eps),
                    "self_attn": Attention(
                        transformer_hidden_size,
                        n_heads,
                        cross_attention_dim=None,
                        apply_rotary_emb=False,
                    ),
                    "cross_norm": nn.LayerNorm(transformer_hidden_size, eps=norm_eps),
                    "cross_attn": Attention(
                        transformer_hidden_size,
                        n_heads,
                        cross_attention_dim=transformer_hidden_size,
                        apply_rotary_emb=False,
                    ),
                    "ffn_norm": nn.LayerNorm(transformer_hidden_size, eps=norm_eps),
                    "ffn": FeedForward(
                        dim=transformer_hidden_size,
                        hidden_dim=4 * transformer_hidden_size,
                        multiple_of=multiple_of,
                        ffn_dim_multiplier=ffn_dim_multiplier,
                    ),
                }
            )
            self.blocks.append(block)

        # Projection from transformer_hidden_size to conditioning_hidden_size
        self.output_proj = nn.Linear(transformer_hidden_size, conditioning_hidden_size, bias=True)

        # Mapping from num_queries -> num_dit_layers (always per-layer output) without bias
        # and an explicit learned bias in the output space (per layer, per channel)
        self.query_to_layer = nn.Linear(num_queries, num_dit_layers, bias=False)

    def forward(self, cross_attention_states, batch_size=1, cross_attention_mask=None):
        # cross_attention_states: (batch_size, sequence_length, transformer_hidden_size)
        
        # TODO: implement this function
        #
        # First, get the set of learnable query embeddings from the query table using indices 0 to num_queries - 1
        #  Put the queries on the same device as the cross_attention_states, and expand them to shape (batch_size, num_queries, transformer_hidden_size)
        # loop through each block, and do the following:
        #     1. residually apply the self-attention block to the queries with attention layernorm (i.e. x -> x + self_attn(attn_norm(x)))
        #     2. residually apply the cross-attention block to the queries with cross attention layernorm (i.e. x -> x + cross_attn(cross_norm(x)))
                # NOTE: Make sure to pass in the text cross attention states and mask to the cross attention block.
        #     3. residually apply the feed-forward block to the queries with ffn layernorm (i.e. x -> x + ffn(ffn_norm(x)))
        # After this loop, your queries should have shape (batch_size, num_queries, transformer_hidden_size), with absorbed information from the text states.
        # Finally, project the queries to the conditioning hidden size using the output projection layer, yielding a tensor of shape (batch_size, num_queries, conditioning_hidden_size)
        #  Then, map the queries to the per-layer features using the query to layer linear layer, yielding a tensor of shape (batch_size, num_dit_layers, conditioning_hidden_size)
        #      NOTE: for this step, you need to change the queries to shape (batch_size, conditioning_hidden_size, num_queries) first, then apply the linear layer, and then change the shape back to shape (batch_size, num_dit_layers, conditioning_hidden_size)
        #  Finally, return the per-layer features
        #
        # ====== BEGIN STUDENT SOLUTION ===========================================
        indices = torch.arange(
    self.num_queries,
    device=cross_attention_states.device
)
        queries = self.query_table(indices)
        queries = queries.unsqueeze(0)
        queries = queries.expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = queries + block["self_attn"](block["attn_norm"](queries))
            normalized_queries = block["cross_norm"](queries)
            attention_output = block["cross_attn"](
        normalized_queries,
        cross_attention_states=cross_attention_states,
        cross_attention_mask=cross_attention_mask
    )
            queries = queries + attention_output
            queries = queries + block["ffn"](block["ffn_norm"](queries))
        projected_queries = self.output_proj(queries)
        x = projected_queries.transpose(1,2)
        x = self.query_to_layer(x)
        x = x.transpose(1,2)
        return x


        # ====== END STUDENT SOLUTION ===========================================


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class Attention(nn.Module):
    def __init__(self, dim, n_heads, cross_attention_dim=None, apply_rotary_emb=True):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        if cross_attention_dim is None:
            cross_attention_dim = dim

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cross_attention_dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cross_attention_dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

        self.use_rotary = apply_rotary_emb

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis=None, cross_attention_states=None, cross_attention_mask=None):
        bsz, seqlen_q, _ = x.shape

        xq = self.wq(x)

        if cross_attention_states is not None:
            xk_in = cross_attention_states
            xv_in = cross_attention_states
        else:
            xk_in = x
            xv_in = x

        xk = self.wk(xk_in)
        xv = self.wv(xv_in)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        seqlen_kv = xk_in.shape[1]

        xq = xq.view(bsz, seqlen_q, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen_kv, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_kv, self.n_heads, self.head_dim)

        if self.use_rotary:
            xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        attn_mask = None
        if cross_attention_states is not None and cross_attention_mask is not None:
            # cross_attention_mask: (B, K) with True for valid tokens; SDPA expects True for masked positions
            mask_bool = (cross_attention_mask.to(torch.bool)).to(xq.device)
            attn_mask = mask_bool[:, None, None, :]  # (B,1,1,K)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        use_rotary_emb=True,
        use_cross_attention=False,
        cross_attention_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.use_cross_attention = use_cross_attention

        self.attention = Attention(dim, n_heads, apply_rotary_emb=use_rotary_emb)
        if use_cross_attention:
            self.cross_attention = Attention(
                dim, n_heads, cross_attention_dim=cross_attention_dim, apply_rotary_emb=use_rotary_emb
            )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        if use_cross_attention:
            self.cross_attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None, cross_attention_states=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )

            if self.use_cross_attention and cross_attention_states is not None:
                x = x + gate_msa.unsqueeze(1) * self.cross_attention(
                    modulate(self.cross_attention_norm(x), shift_msa, scale_msa),
                    freqs_cis,
                    cross_attention_states=cross_attention_states,
                )

            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            if self.use_cross_attention and cross_attention_states is not None:
                x = x + self.cross_attention(self.cross_attention_norm(x), freqs_cis, cross_attention_states=cross_attention_states)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
        # Optional text conditioning via QueryEmbedder
        use_text_conditioning=False,
        transformer_hidden_size=None,
        num_query_tokens=None,
        qe_num_layers=2,
        qe_n_heads=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.conditioning_hidden_size = min(dim, 1024)
        self.t_embedder = TimestepEmbedder(self.conditioning_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.conditioning_hidden_size, class_dropout_prob)

        # Optional QueryEmbedder for text conditioning
        self.use_text_conditioning = bool(use_text_conditioning)
        if self.use_text_conditioning:
            if transformer_hidden_size is None or num_query_tokens is None:
                raise ValueError("When use_text_conditioning=True, provide transformer_hidden_size and num_query_tokens")
            qe_n_heads = qe_n_heads if qe_n_heads is not None else max(1, min(n_heads, 8))
            self.query_embedder = QueryEmbedder(
                transformer_hidden_size=transformer_hidden_size,
                conditioning_hidden_size=self.conditioning_hidden_size,
                num_queries=num_query_tokens,
                num_dit_layers=n_layers,
                num_layers=qe_num_layers,
                n_heads=qe_n_heads,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps,
            )
        else:
            self.query_embedder = None

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y, text_hidden_states=None, text_attention_mask=None):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)

        if self.use_text_conditioning and self.query_embedder is not None and text_hidden_states is not None:
            # Always per-layer features from QueryEmbedder
            q_feat = self.query_embedder(text_hidden_states, batch_size=x.size(0), cross_attention_mask=text_attention_mask)  # (N, n_layers, D)
            per_layer_t = t.to(x.dtype).unsqueeze(1).expand(-1, q_feat.size(1), -1)
            adaln_inputs_per_layer = (per_layer_t + q_feat.to(x.dtype))  # (N, n_layers, D)
            use_per_layer = True
        else:
            y = self.y_embedder(y, self.training)  # (N, D)
            adaln_input = t.to(x.dtype) + y.to(x.dtype)
            use_per_layer = False

        for idx, layer in enumerate(self.layers):
            if use_per_layer:
                x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_inputs_per_layer[:, idx, :])
            else:
                x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        # Final AdaLN uses the same conditioning as the last layer's used conditioning
        final_c = adaln_inputs_per_layer[:, -1, :] if use_per_layer else adaln_input
        x = self.final_layer(x, final_c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, text_hidden_states=None, text_attention_mask=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, text_hidden_states=text_hidden_states, text_attention_mask=text_attention_mask)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

