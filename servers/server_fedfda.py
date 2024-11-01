import time
from copy import deepcopy
import torch
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from servers.server_base import Server
from clients.client_fedfda import ClientFedFDA

class ServerFedFDA(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = [
            ClientFedFDA(args, i) for i in range(self.num_clients)
        ]
        self.global_means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.global_covariance = torch.Tensor(torch.eye(self.D))
        self.global_priors = torch.ones(self.num_classes) / self.num_classes
        self.r = 0

    def train(self):
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            self.r = r
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
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Test Loss [{ptest_loss:.4f}]\t Test Acc [{ptest_acc:.2f}({ptest_acc_std:.2f})]\t Train Time [{train_time:.2f}]")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Train Time [{train_time:.2f}]")

    def aggregate_models(self):
        # aggregate base model
        super().aggregate_models()

        # aggregate gaussian estimates
        total_samples = sum(c.num_train for c in self.active_clients)
        self.global_means.data = torch.zeros_like(self.clients[0].means)
        self.global_covariance.data = torch.zeros_like(self.clients[0].covariance)
        
        for c in self.active_clients:
            self.global_means.data = self.global_means.data + (c.num_train / total_samples)*c.adaptive_means.data
            self.global_covariance.data = self.global_covariance.data + (c.num_train / total_samples)*c.adaptive_covariance.data
    
    def send_models(self):
        # send base model
        super().send_models()
        # send global gaussian estiamtes
        for c in self.active_clients:
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            if self.r == 1:
                c.means.data = self.global_means.data
                c.covariance.data = self.global_covariance.data
                c.adaptive_means.data = self.global_means.data
                c.adaptive_covariance.data = self.global_covariance.data

    def evaluate_personalized(self):
        """
        Update global \beta and compute local interpolated (\mu, \Sigma) 
        Evaluate using interpolated (\mu, \Sigma) and local prior
        """
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            c.global_means = c.global_means.to(self.device)
            c.global_covariance = c.global_covariance.to(self.device)
            c.model.eval()
            # solve for beta and use adaptive statistics for classifier
            c_feats, c_labels = c.compute_feats(split="train")
            c.solve_beta(feats=c_feats, labels=c_labels)
            means_mle, scatter_mle, priors, counts = c.compute_mle_statistics(feats=c_feats, labels=c_labels)
            means_mle = torch.stack([means_mle[i] if means_mle[i] is not None and counts[i] > c.min_samples else c.global_means[i] for i in range(self.num_classes)])
            cov_mle = (scatter_mle / (np.sum(counts)-1)) + 1e-4 + torch.eye(self.D).to(self.device)
            cov_psd = cov_nearest(cov_mle.cpu().numpy(), method="clipped")
            cov_psd = torch.Tensor(cov_psd).to(self.device)
            c.update(means_mle, cov_psd)
            c.set_lda_weights(c.adaptive_means, c.adaptive_covariance)
            with torch.no_grad():
                acc, loss = c.evaluate()
                accs.append(acc)
                weighted_loss += (c.num_test / total_samples) * loss.detach()
                weighted_acc += (c.num_test / total_samples) * acc
                c.model = old_model
            c.model = c.model.to("cpu")
            c.global_means = c.global_means.to("cpu")
            c.global_covariance = c.global_covariance.to("cpu")
            c.adaptive_means = c.adaptive_means.to("cpu")
            c.adaptive_covariance = c.adaptive_covariance.to("cpu")
            c.means = c.means.to("cpu")
            c.covariance = c.covariance.to("cpu")
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std