# Cluster 17

class MultiHeadAttention(nn.Module):
    """
    Overview:
        For each entry embedding, compute individual attention across all entries, add them up to get output attention
    """

    def __init__(self, n_heads: int=None, dim: int=None, dropout: float=0):
        """
        Overview:
            Init attention
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): head num for multihead attention
            - dropout (:obj:`nn.Module`): dropout layer
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def split(self, x, T=False):
        """
        Overview:
            Split input to get multihead queries, keys, values
        Arguments:
            - x (:obj:`tensor`): query or key or value
            - T (:obj:`bool`): whether to transpose output
        Returns:
            - x (:obj:`list`): list of output tensors for each head
        """
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor]=None, value: Optional[torch.Tensor]=None, mask: torch.Tensor=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(batch_size * n_heads, seq_len, dim_per_head)
            return tensor
        if key is None and value is None:
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            value = key
        assert key is not None
        _, _key_len, dim = key.size()
        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))
        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        attn_mask = (mask == 0).view(batch_size, 1, -1, full_key_len).repeat(1, n_heads, 1, 1).expand(batch_size, n_heads, query_len, full_key_len).view(batch_size * n_heads, query_len, full_key_len)
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))
        attn_weights = F.softmax(dot_prod, dim=-1, dtype=torch.float).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)
        attentioned = attn_weights.bmm(v)
        attentioned = attentioned.type_as(query).view(batch_size, n_heads, query_len, dim_per_head).transpose(1, 2).contiguous().view(batch_size, query_len, dim)
        out = self.out_lin(attentioned)
        return (out, dot_prod)

def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

