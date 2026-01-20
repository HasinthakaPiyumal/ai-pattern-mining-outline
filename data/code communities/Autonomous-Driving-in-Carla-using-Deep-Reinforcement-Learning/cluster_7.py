# Cluster 7

def main():
    data_dir = 'autoencoder/dataset/'
    writer = SummaryWriter(f'runs/' + 'auto-encoder')
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.ImageFolder(data_dir + 'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + 'test', transform=test_transforms)
    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f'Selected device :) :) :) {device}')
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optim)
        writer.add_scalar('Training Loss/epoch', train_loss, epoch + 1)
        val_loss = test(model, validloader)
        writer.add_scalar('Validation Loss/epoch', val_loss, epoch + 1)
        print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS, train_loss, val_loss))
    model.save()

def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for x, _ in trainloader:
        x = x.to(device)
        x_hat = model(x)
        loss = ((x - x_hat) ** 2).sum() + model.encoder.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(trainloader.dataset)

def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            encoded_data = model.encoder(x)
            x_hat = model(x)
            loss = ((x - x_hat) ** 2).sum() + model.encoder.kl
            val_loss += loss.item()
    return val_loss / len(testloader.dataset)

