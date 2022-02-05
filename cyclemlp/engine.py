import torch


class Engine:
    def __init__(self, device):
        self._device = device

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        mean_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            current = batch * len(x) if batch < len(dataloader)-1 else size
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

            mean_loss += loss / len(x)
        return mean_loss

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self._device), y.to(self._device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print('Test Error:')
        print(f' Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}')
        return test_loss, correct
