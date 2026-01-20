# Cluster 17

def test():
    torch.manual_seed(0)
    loss = PixorLoss(None)
    pred = torch.sigmoid(torch.randn(1, 7, 2, 3))
    label = torch.zeros(1, 7, 2, 3)
    loss = loss(pred, label)
    print(loss)

