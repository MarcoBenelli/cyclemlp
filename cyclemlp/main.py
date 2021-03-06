import csv

import torch
import torchvision
import timm

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
    img_size = x.shape[2]
    print('Shape of y: ', y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = cycle_mlp.CycleMLP_B1(in_chans=in_chans, num_classes=10)
# model = torchvision.models.resnet18(num_classes=10)
# model = timm.models.vit_tiny_patch16_224(num_classes=10, img_size=img_size)
# model = timm.models.mixer_s16_224(num_classes=10, img_size=img_size)
model = model.to(device)
print(model)

epochs = 100

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

engine_ = engine.Engine(device)
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train loss', 'test loss',
                     'train accuracy', 'test accuracy'])
for t in range(epochs):
    print(f'Epoch {t}\n-------------------------------')
    train_loss, train_accuracy = engine_.train(
        train_dataloader, model, loss_fn, optimizer)
    test_loss, test_accuracy = engine_.test(test_dataloader, model, loss_fn)
    scheduler.step()
    print()
    with open('output.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([t, f'{train_loss:f}', f'{test_loss:f}',
                         f'{train_accuracy:f}', f'{test_accuracy:f}'])
print('Done!')

torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model State to model.pth')
