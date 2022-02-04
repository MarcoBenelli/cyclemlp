import torch
import torchvision

import cycle_mlp
import engine


# Download training data from open datasets.
training_data = torchvision.datasets.STL10(
    root='data',
    split='train',
    # train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Download test data from open datasets.
test_data = torchvision.datasets.STL10(
    root='data',
    split='test',
    # train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

batch_size = 256

# Create data loaders.
train_dataloader = torch.utils.data.DataLoader(training_data,
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print('Shape of X [N, C, H, W]: ', x.shape)
    in_chans = x.shape[1]
    print('Shape of y: ', y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = cycle_mlp.CycleMLP_B1(in_chans=in_chans, num_classes=10).to(device)
print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)

epochs = 300
engine_ = engine.Engine(device)
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    engine_.train(train_dataloader, model, loss_fn, optimizer)
    engine_.test(test_dataloader, model, loss_fn)
    print()
print('Done!')

torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model State to model.pth')
