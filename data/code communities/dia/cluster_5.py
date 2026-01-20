# Cluster 5

class CrossAttention(nn.Module):
    """Cross-Attention using DenseGeneral."""

    def __init__(self, config: EncoderConfig | DecoderConfig, q_embed_dim: int, kv_embed_dim: int, num_query_heads: int, num_kv_heads: int, head_dim: int, compute_dtype: torch.dtype, out_embed_dim: int | None=None):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f'num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})')
        self.num_gqa_groups = num_query_heads // num_kv_heads
        self.q_proj = DenseGeneral(in_shapes=(q_embed_dim,), out_features=(num_query_heads, head_dim), axis=(-1,), weight_dtype=compute_dtype)
        self.k_proj = DenseGeneral(in_shapes=(kv_embed_dim,), out_features=(num_kv_heads, head_dim), axis=(-1,), weight_dtype=compute_dtype)
        self.v_proj = DenseGeneral(in_shapes=(kv_embed_dim,), out_features=(num_kv_heads, head_dim), axis=(-1,), weight_dtype=compute_dtype)
        self.o_proj = DenseGeneral(in_shapes=(num_query_heads, head_dim), out_features=(self.output_dim,), axis=(-2, -1), weight_dtype=compute_dtype)
        self.rotary_emb = RotaryEmbedding(embedding_dims=self.head_dim, max_timescale=config.rope_theta, dtype=compute_dtype)

    def forward(self, Xq: torch.Tensor, q_positions: torch.Tensor, kv_positions: torch.Tensor | None=None, attn_mask: torch.Tensor | None=None, cache: KVCache | None=None, is_causal: bool=False) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Performs attention calculation with optional KV caching.

        Args:
            Xq: Query tensor (B, T, D). T=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, S, E). S=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, T).
            kv_positions: Positions for keys/values (B, S). If None, uses q_positions.
            attn_mask: Attention mask.
            cache: KVCache.

        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, output_dim).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)). For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype
        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)
        attn_k: torch.Tensor | None = cache.k if cache is not None else None
        attn_v: torch.Tensor | None = cache.v if cache is not None else None
        is_mps = Xq.device.type == 'mps' and torch.backends.mps.is_available()
        if is_mps:
            attn_output = custom_scaled_dot_product_attention(query=Xq_BxNxTxH, key=attn_k, value=attn_v, attn_mask=attn_mask if not is_causal else None, scale=1.0, is_causal=is_causal, num_gqa_groups=self.num_gqa_groups)
        else:
            attn_output = F.scaled_dot_product_attention(Xq_BxNxTxH, attn_k, attn_v, attn_mask=attn_mask if not is_causal else None, scale=1.0, enable_gqa=self.num_gqa_groups > 1, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous()
        output = self.o_proj(attn_output)
        return output.to(original_dtype)

def custom_scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None=None, scale: float=1.0, is_causal: bool=False, num_gqa_groups: int=1) -> torch.Tensor:
    """
    Custom scaled dot-product attention with GQA support for MPS compatibility.

    Args:
        query: (B, N_q, T, H) - Query tensor, N_q = num_query_heads
        key: (B, N_kv, S, H) - Key tensor, N_kv = num_kv_heads
        value: (B, N_kv, S, H) - Value tensor
        attn_mask: (B, 1, T, S) - Attention mask, optional
        scale: Scaling factor for attention scores
        is_causal: If True, apply causal masking
        num_gqa_groups: Number of query groups per KV head (N_q / N_kv)

    Returns:
        output: (B, N_q, T, H) - Attention output
    """
    B, N_q, T, H = query.shape
    _, N_kv, S, _ = key.shape
    if num_gqa_groups > 1:
        key = key.repeat_interleave(num_gqa_groups, dim=1)
        value = value.repeat_interleave(num_gqa_groups, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if is_causal:
        causal_mask = torch.tril(torch.ones(T, S, dtype=torch.bool, device=query.device))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output

class SelfAttention(nn.Module):
    """Attention using DenseGeneral."""

    def __init__(self, config: EncoderConfig | DecoderConfig, q_embed_dim: int, kv_embed_dim: int, num_query_heads: int, num_kv_heads: int, head_dim: int, compute_dtype: torch.dtype, out_embed_dim: int | None=None):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f'num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})')
        self.num_gqa_groups = num_query_heads // num_kv_heads
        self.kv_embed_dim = kv_embed_dim
        self.q_embed_dim = q_embed_dim
        self.q_proj = DenseGeneral(in_shapes=(q_embed_dim,), out_features=(num_query_heads, head_dim), axis=(-1,), weight_dtype=compute_dtype)
        self.k_proj = DenseGeneral(in_shapes=(kv_embed_dim,), out_features=(num_kv_heads, head_dim), axis=(-1,), weight_dtype=compute_dtype)
        self.v_proj = DenseGeneral(in_shapes=(kv_embed_dim,), out_features=(num_kv_heads, head_dim), axis=(-1,), weight_dtype=compute_dtype)
        self.o_proj = DenseGeneral(in_shapes=(num_query_heads, head_dim), out_features=(self.output_dim,), axis=(-2, -1), weight_dtype=compute_dtype)
        self.rotary_emb = RotaryEmbedding(embedding_dims=self.head_dim, max_timescale=config.rope_theta, dtype=compute_dtype)
        self.is_fused_qkv = False

    def get_linear_weight(self, dense: DenseGeneral):
        W_dg = dense.weight.data
        out_features = 1
        input_features = 1
        for dim in dense.out_features:
            out_features *= dim
        for dim in dense.in_shapes:
            input_features *= dim
        W_dg_reshaped_for_linear_T = W_dg.reshape(input_features, out_features)
        linear_weight = W_dg_reshaped_for_linear_T.transpose(0, 1).contiguous()
        return linear_weight

    def patch_fused_qkv(self):
        q_proj_weight = self.get_linear_weight(self.q_proj)
        k_proj_weight = self.get_linear_weight(self.k_proj)
        v_proj_weight = self.get_linear_weight(self.v_proj)
        self.qkv = FusedQKV(self.kv_embed_dim, self.num_query_heads * self.head_dim + 2 * (self.num_kv_heads * self.head_dim), bias=False, num_q_heads=self.num_query_heads, q_head_dim=self.head_dim, num_kv_heads=self.num_kv_heads, kv_head_dim=self.head_dim)
        self.qkv.linear.weight.data = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        self.is_fused_qkv = True

    def forward(self, X: torch.Tensor, q_positions: torch.Tensor, kv_positions: torch.Tensor | None=None, attn_mask: torch.Tensor | None=None, cache: KVCache | None=None, prefill: bool=False, is_causal: bool=False, current_idx: torch.Tensor | None=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Performs attention calculation with optional KV caching.
        Args:
            Xq: Query tensor (B, T, D). T=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, S, E). S=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, T).
            kv_positions: Positions for keys/values (B, S). If None, uses q_positions.
            attn_mask: Attention mask.
            cache: KVCache.
            prefill: If True, use prefill mode.
        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, output_dim).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)). For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = X.dtype
        if self.is_fused_qkv:
            Xq_BxTxNxH, Xk_BxSxKxH, Xv_BxSxKxH = self.qkv(X)
        else:
            Xq_BxTxNxH = self.q_proj(X)
            Xk_BxSxKxH = self.k_proj(X)
            Xv_BxSxKxH = self.v_proj(X)
        position = q_positions.unsqueeze(-1).unsqueeze(-1)
        sinusoid_inp = position / self.rotary_emb.timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        Xq_BxTxNxH = self.rotary_emb.apply_rope(Xq_BxTxNxH, sin, cos)
        Xk_BxSxKxH = self.rotary_emb.apply_rope(Xk_BxSxKxH, sin, cos)
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)
        attn_k: torch.Tensor | None = cache.k if cache is not None else None
        attn_v: torch.Tensor | None = cache.v if cache is not None else None
        Xk_BxKxSxH = Xk_BxSxKxH.transpose(1, 2)
        Xv_BxKxSxH = Xv_BxSxKxH.transpose(1, 2)
        if cache is None:
            attn_k = Xk_BxKxSxH
            attn_v = Xv_BxKxSxH
        elif prefill:
            attn_k, attn_v = (Xk_BxKxSxH, Xv_BxKxSxH)
            cache.prefill(attn_k, attn_v)
        else:
            attn_k, attn_v = cache.update(Xk_BxKxSxH, Xv_BxKxSxH, current_idx)
        is_mps = Xv_BxSxKxH.device.type == 'mps' and torch.backends.mps.is_available()
        if is_mps:
            attn_output = custom_scaled_dot_product_attention(query=Xq_BxNxTxH, key=attn_k, value=attn_v, attn_mask=attn_mask if not is_causal else None, scale=1.0, is_causal=is_causal, num_gqa_groups=self.num_gqa_groups)
        else:
            attn_output = F.scaled_dot_product_attention(Xq_BxNxTxH, attn_k, attn_v, attn_mask=attn_mask if not is_causal else None, scale=1.0, enable_gqa=self.num_gqa_groups > 1, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous()
        output = self.o_proj(attn_output)
        return output.to(original_dtype)

