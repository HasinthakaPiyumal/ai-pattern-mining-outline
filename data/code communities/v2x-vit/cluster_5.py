# Cluster 5

class BaseWindowAttention(nn.Module):

    def __init__(self, dim, heads, dim_head, drop_out, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(drop_out))

    def forward(self, x):
        b, l, h, w, c, m = (*x.shape, self.heads)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        new_h = h // self.window_size
        new_w = w // self.window_size
        q, k, v = map(lambda t: rearrange(t, 'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c', m=m, w_h=self.window_size, w_w=self.window_size), qkv)
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        out = rearrange(out, 'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)', m=self.heads, w_h=self.window_size, w_w=self.window_size, new_w=new_w, new_h=new_h)
        out = self.to_out(out)
        return out

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

