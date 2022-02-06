import torch


class Engine:
    def __init__(self, device):
        self._device = device

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        mean_loss = 0
        correct = 0
        current = 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = model(x)
            correct += (pred.argmax(dim=1) == y).sum().item()
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            print('loss: {:7f}  [{:{width}}/{:{width}}]'.format(
                loss, current, size, width=len(str(size))))
            current += len(x)

            mean_loss += loss
        return mean_loss/len(dataloader), correct/size

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
        print(f' Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:8f}')
        return test_loss, correct
