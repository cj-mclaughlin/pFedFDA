import time
from servers.server_base import Server
from clients.client_local import ClientLocal

class ServerLocal(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = [
            ClientLocal(args, i) for i in range(self.num_clients)
        ]

    def train(self):
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            if r == (self.global_rounds): # full participation on last round
                self.sampling_prob = 1.0
            self.sample_active_clients()
            
            # train clients
            train_acc, train_loss = self.train_clients()
            
            train_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(train_time)

            # logging
            if r % self.eval_gap == 0 or r == self.global_rounds:
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate() 
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Test Loss [{ptest_loss:.4f}]\t Test Acc [{ptest_acc:.2f}({ptest_acc_std:.2f})]\t Train Time [{train_time:.2f}]")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Train Time [{train_time:.2f}]")
                
    def evaluate_personalized(self):
        ptest_acc, ptest_loss, ptest_acc_std = self.evaluate()
        return ptest_acc, ptest_loss, ptest_acc_std