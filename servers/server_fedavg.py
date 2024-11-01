import time
from copy import deepcopy
import torch
from servers.server_base import Server
from clients.client_fedavg import ClientFedAvg

class ServerFedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = []
        for client_idx in range(self.num_clients):
            c = ClientFedAvg(args, client_idx)
            self.clients.append(c)

    def send_models(self):
        for c in self.active_clients:
            c.set_model(self.model)

    def train(self):
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            if r == (self.global_rounds): # full participation on last round
                self.sampling_prob = 1.0
            self.sample_active_clients()
            self.send_models()

            # train clients
            train_acc, train_loss = self.train_clients()
            train_time = time.time() - start_time

            # average clients
            self.aggregate_models()

            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

            # logging
            if r % self.eval_gap == 0 or r == self.global_rounds:
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()  
                test_acc, test_loss, test_acc_std = self.evaluate() 
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Test Loss [{test_loss:.4f}|{ptest_loss:.4f}]\t Test Acc [{test_acc:.2f}({test_acc_std:.2f})|{ptest_acc:.2f}({ptest_acc_std:.2f})]\t Train Time [{train_time:.2f}]")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Train Time [{train_time:.2f}]")
                
    def evaluate(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
            c.model = old_model
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std

    def evaluate_personalized(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            c.train()
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
            c.model = old_model
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std
    