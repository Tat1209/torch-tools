import torch

class Trainer:
    def __init__(self, network, criterion, optimizer, scheduler=None, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_1epoch(self, dl):
        total_loss = 0.0
        total_corr = 0

        self.network.train()
        for inputs, labels in dl:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, loss = self.forward_flow(inputs, labels)
            preds, corr = self.eval_flow(outputs, labels)

            total_loss += loss.item() * len(inputs)
            total_corr += corr

            self.update_grad(loss)

        if self.scheduler is not None:
            self.scheduler.step()

        train_loss = total_loss / len(dl.dataset)
        train_acc = total_corr / len(dl.dataset)

        return train_loss, train_acc
    
    def val_1epoch(self, dl):
        if dl is None:
            return None, None

        self.network.eval()
        total_loss = 0.0
        total_corr = 0

        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, loss = self.forward_flow(inputs, labels)
                preds, corr = self.eval_flow(outputs, labels)

                total_loss += loss.item() * len(inputs)
                total_corr += corr

        val_loss = total_loss / len(dl.dataset)
        val_acc = total_corr / len(dl.dataset)

        return val_loss, val_acc

    def forward_flow(self, inputs, labels):
        outputs = self.network(inputs)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def eval_flow(self, outputs, labels):
        preds = torch.argmax(outputs.detach(), dim=1)
        corr = torch.sum(preds == labels.data).item()

        return preds, corr

    def update_grad(self, loss):
        # 通常は zero_grad -> bachward -> step
        # 効率を考慮するなら backward -> step -> zero_grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
