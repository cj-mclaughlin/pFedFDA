from utils.util import AverageMeter
import torch 
from clients.client_base import Client

class ClientLocal(Client):
    def __init__(self, args, client_idx):
        super().__init__(args, client_idx)

    def train(self):
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        self.model = self.model.to(self.device)
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        for e in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs.update(acc, x.size(0))
                losses.update(loss.item(), x.size(0))

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model = self.model.to("cpu")
        return accs.avg, losses.avg