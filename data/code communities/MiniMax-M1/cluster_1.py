# Cluster 1

class MiniMaxM1FlashAttention2(MiniMaxM1Attention):
    """
    MiniMaxM1 flash attention module. This module inherits from `MiniMaxM1Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Union[Cache, Tuple[torch.Tensor]]]=None, output_attentions: bool=False, use_cache: bool=False, **kwargs):
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
            attention_mask = kwargs.pop('padding_mask')
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        use_sliding_windows = _flash_supports_window_size and getattr(self.config, 'sliding_window', None) is not None and (kv_seq_len > self.config.sliding_window)
        if not _flash_supports_window_size:
            logger.warning_once('The current flash attention version does not support sliding window attention, for a more memory efficient implementation make sure to upgrade flash-attn library.')
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(f'The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.')
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-3)
            value_states = torch.cat([past_key_value[1], value_states], dim=-3)
        past_key_value = (key_states, value_states) if use_cache else None
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, use_sliding_windows=use_sliding_windows)
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, use_sliding_windows=False):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal)
            else:
                attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal, window_size=(self.config.sliding_window, self.config.sliding_window))
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif not use_sliding_windows:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, window_size=(self.config.sliding_window, self.config.sliding_window))
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len:]
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        return (query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k))

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)

