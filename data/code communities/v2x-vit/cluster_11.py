# Cluster 11

class PyramidWindowAttention(nn.Module):

    def __init__(self, dim, heads, dim_heads, drop_out, window_size, relative_pos_embedding, fuse_method='naive'):
        super().__init__()
        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)
        self.pwmsa = nn.ModuleList([])
        for head, dim_head, ws in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim, head, dim_head, drop_out, ws, relative_pos_embedding))
        self.fuse_mehod = fuse_method
        if fuse_method == 'split_attn':
            self.split_attn = SplitAttn(256)

    def forward(self, x):
        output = None
        if self.fuse_mehod == 'naive':
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            return output / len(self.pwmsa)
        elif self.fuse_mehod == 'split_attn':
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            return self.split_attn(window_list)

