# Cluster 3

@dataclass
class EncoderInferenceState:
    """Parameters specifically for encoder inference."""
    max_seq_len: int
    device: torch.device
    positions: torch.Tensor
    padding_mask: torch.Tensor
    attn_mask: torch.Tensor

    @classmethod
    def new(cls, config: DiaConfig, cond_src: torch.Tensor) -> 'EncoderInferenceState':
        """Creates EtorchrInferenceParams from DiaConfig and a device."""
        device = cond_src.device
        positions = torch.arange(config.encoder_config.max_position_embeddings, dtype=torch.float32, device=device).unsqueeze(0)
        padding_mask = (cond_src.squeeze(1) != 0).to(device).repeat_interleave(2, dim=0)
        attn_mask = create_attn_mask(padding_mask, padding_mask, device, is_causal=False)
        return cls(max_seq_len=config.encoder_config.max_position_embeddings, device=device, positions=positions, padding_mask=padding_mask, attn_mask=attn_mask)

def create_attn_mask(q_padding_mask_1d: torch.Tensor, k_padding_mask_1d: torch.Tensor, device: torch.device, is_causal: bool=False) -> torch.Tensor:
    """
    Creates the attention mask (self or cross) mimicking JAX segment ID logic.
    """
    p_mask_q = q_padding_mask_1d.unsqueeze(2)
    p_mask_k = k_padding_mask_1d.unsqueeze(1)
    non_pad_attends_non_pad = p_mask_q & p_mask_k
    pad_attends_pad = ~p_mask_q & ~p_mask_k
    mask = non_pad_attends_non_pad | pad_attends_pad
    if is_causal:
        causal_mask_2d = torch.tril(torch.ones_like(mask[0], dtype=torch.bool, device=device))
        causal_mask = mask & causal_mask_2d
        return causal_mask.unsqueeze(1)
    else:
        return mask.unsqueeze(1)

@dataclass
class DecoderInferenceState:
    """Parameters specifically for decoder inference."""
    device: torch.device
    dtype: torch.dtype
    enc_out: torch.Tensor
    enc_positions: torch.Tensor
    dec_positions: torch.Tensor
    self_attn_cache: list[KVCache]
    cross_attn_cache: list[KVCache]
    casual_attn_mask: torch.Tensor
    cross_attn_mask: torch.Tensor

    @classmethod
    def new(cls, config: DiaConfig, enc_state: EncoderInferenceState, enc_out: torch.Tensor, dec_cross_attn_cache: list[KVCache], compute_dtype: torch.dtype, max_generation_length: Optional[int]=None) -> 'DecoderInferenceState':
        """Creates DecoderInferenceParams from DiaConfig and a device."""
        device = enc_out.device
        max_audio_len = max_generation_length or config.decoder_config.max_position_embeddings
        batch_size = enc_out.shape[0] // 2
        dec_positions = torch.full((2 * batch_size, 1), fill_value=0, dtype=torch.int32, device=device)
        causal_mask = torch.tril(torch.ones(max_audio_len, max_audio_len, dtype=torch.bool, device=device))
        dec_mask = torch.ones((2 * batch_size, 1), dtype=torch.bool, device=device)
        cross_attn_mask = create_attn_mask(dec_mask, enc_state.padding_mask, device, is_causal=False)
        self_attn_cache = [KVCache(batch_size, config.decoder_config.num_key_value_heads, max_audio_len, config.decoder_config.head_dim, compute_dtype, device) for _ in range(config.decoder_config.num_hidden_layers)]
        return cls(device=device, dtype=compute_dtype, enc_out=enc_out, enc_positions=enc_state.positions, dec_positions=dec_positions, self_attn_cache=self_attn_cache, cross_attn_cache=dec_cross_attn_cache, casual_attn_mask=causal_mask, cross_attn_mask=cross_attn_mask)

    def prepare_step(self, step_from: int, step_to: int | None=None) -> None:
        if step_to is None:
            step_to = step_from + 1
        self.dec_positions = torch.arange(step_from, step_to, dtype=torch.int32, device=self.device).unsqueeze(0)

