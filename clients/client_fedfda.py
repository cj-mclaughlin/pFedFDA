import torch 
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.correlation_tools import cov_nearest
import torchmin
from clients.client_base import Client
from utils.util import AverageMeter

class ClientFedFDA(Client):
    def __init__(self, args, client_idx, corrupted=False):
        super().__init__(args, client_idx, corrupted)
        self.eps = 1e-4
        self.means_beta = torch.ones(size=(self.num_classes,)) * 0.5
        self.cov_beta = torch.Tensor([0.5])
        
        # Local statistics
        self.means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.covariance = torch.Tensor(torch.eye(self.D))
        counts, priors = self.get_label_distribution("train")
        self.priors = priors.cpu() + self.eps
        self.priors = self.priors / self.priors.sum()
        self.class_counts = counts.cpu()
        
        # Global statistics
        self.global_means = deepcopy(self.means)
        self.global_covariance = deepcopy(self.covariance)
        
        # Interpolated statistics
        self.adaptive_means = deepcopy(self.means)
        self.adaptive_covariance = deepcopy(self.covariance)
        
        # Interpolation term solver configuration
        self.single_beta = args.single_beta
        self.local_beta = args.local_beta
        self.num_cv_folds = 2
        self.min_samples = self.num_cv_folds
        if self.local_beta:
            self.single_beta = True

    def train(self):
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        
        self.model = self.model.to(self.device)
        self.means = self.means.to(self.device)
        self.covariance = self.covariance.to(self.device)
        self.adaptive_means = self.adaptive_means.to(self.device)
        self.adaptive_covariance = self.adaptive_covariance.to(self.device)
        self.global_means = self.global_means.to(self.device)
        self.global_covariance = self.global_covariance.to(self.device)
        
        self.model.train()
        self.set_lda_weights(self.global_means, self.global_covariance, self.priors)

        # Freeze FC head for FDA parameterization
        for param in self.model.fc.parameters():
            param.requires_grad_(False)

        if self.use_probvlm:
            # Train the ProbVLM adapter, then fit statistics on features from the updated adapter
            train_acc, train_loss = self.train_probvlm()
            feats_ep, labels_ep = self.compute_feats(split="train")
        else:
            losses = AverageMeter()
            accs = AverageMeter()
            feats_ep, labels_ep = [], []

            for e in range(1, self.local_epochs + 1):
                for i, (x, y) in enumerate(trainloader):
                    x, y = x.to(self.device), y.to(self.device)

                    feats, output = self.model(x, return_feat=True)
                    loss = self.loss(output, y)
                    acc = (output.argmax(1) == y).float().mean() * 100.0
                    
                    accs.update(acc.item(), x.size(0))
                    losses.update(loss.item(), x.size(0))

                    feats_ep.append(feats.detach())
                    labels_ep.append(y.cpu().numpy())

                    # Only run backward pass if parameters are trainable (handles frozen CLIP)
                    if any(p.requires_grad for p in self.model.parameters()):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            feats_ep = torch.cat(feats_ep, dim=0)
            labels_ep = np.concatenate(labels_ep, axis=0)
            train_acc, train_loss = accs.avg, losses.avg

        # Solve for local interpolation beta
        self.solve_beta(feats_ep, labels_ep)
        
        # Estimate local Gaussian MLE parameters
        means_mle, scatter_mle, priors, counts = self.compute_mle_statistics(feats=feats_ep, labels=labels_ep)
        cov_mle = (scatter_mle / (np.sum(counts) - 1)) + self.eps + torch.eye(self.D).to(self.device)
        cov_psd = cov_nearest(cov_mle.cpu().numpy(), method="clipped")
        cov_psd = torch.Tensor(cov_psd).to(self.device)
        means_mle = torch.stack([means_mle[i] if means_mle[i] is not None else self.global_means[i] for i in range(self.num_classes)])

        # Update interpolated adaptive parameters
        self.update(means_mle, cov_psd)
        
        # Move tensors back to CPU to conserve GPU memory
        self._to_cpu()
        return train_acc, train_loss

    def evaluate_zero_shot(self):
        """Evaluates client test data using CLIP text similarity logits."""
        testloader = self.load_test_data()
        self.model = self.model.to(self.device)
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                # Compute cosine similarities scaled by temperature
                logits = self.model.get_text_similarities(x)
                loss = self.loss(logits, y)
                
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item() * y.size(0)

        self._to_cpu()
        return (correct / total) * 100.0, total_loss / total

    def evaluate_goodness_of_fit(self):
        """
        Computes Gaussian log-likelihood and Mardia's Skewness/Kurtosis
        on local class feature distributions.
        """
        self.model = self.model.to(self.device)
        feats, labels = self.compute_feats(split="train")
        
        class_ll, mardia_skew, mardia_kurt = [], [], []

        for c in range(self.num_classes):
            idx = (labels == c)
            N = idx.sum().item()
            if N < self.D + 1: # Require enough samples for non-singular covariance
                continue

            zc = feats[idx] # (N, D)
            mu_c = zc.mean(dim=0, keepdim=True)
            zc_centered = zc - mu_c
            
            # Sample covariance
            cov_c = (zc_centered.T @ zc_centered) / (N - 1) + self.eps * torch.eye(self.D, device=self.device)
            
            # 1. Log-Likelihood under estimated Gaussian
            try:
                dist = torch.distributions.MultivariateNormal(mu_c.squeeze(0), cov_c)
                ll = dist.log_prob(zc).mean().item()
                class_ll.append(ll)
            except Exception:
                pass

            # 2. Mardia's Multivariate Normality Metrics
            try:
                inv_cov = torch.linalg.pinv(cov_c)
                # Mahalanobis distance matrix M_ij = (x_i - mu)^T Sigma^-1 (x_j - mu)
                M = (zc_centered @ inv_cov @ zc_centered.T)
                
                skew = (M ** 3).sum().item() / (6.0 * N)
                kurt = torch.diag(M ** 2).mean().item() # Ideal value under Normality ~ D*(D+2)
                
                mardia_skew.append(skew)
                mardia_kurt.append(kurt)
            except Exception:
                pass

        self._to_cpu()
        return {
            "avg_log_likelihood": np.mean(class_ll) if class_ll else float("nan"),
            "mardia_skewness": np.mean(mardia_skew) if mardia_skew else float("nan"),
            "mardia_kurtosis": np.mean(mardia_kurt) if mardia_kurt else float("nan"),
            "expected_kurtosis": self.D * (self.D + 2)
        }

    def beta_classifier(self, beta, means_local, cov_local, feats, labels):
        if self.single_beta:
            means = beta * means_local + (1 - beta) * self.global_means
            cov = beta * cov_local + (1 - beta) * self.global_covariance
        else:
            means = beta[0] * means_local + (1 - beta[0]) * self.global_means
            cov = beta[-1] * cov_local + (1 - beta[-1]) * self.global_covariance
        y_pred = self.lda_classify(feats, means=means, covariance=cov, priors=self.priors, use_lstsq=True)
        return torch.nn.functional.cross_entropy(y_pred, torch.LongTensor(labels).to(self.device))

    def solve_beta(self, feats, labels, seed=0):
        if self.local_beta:
            self.means_beta = torch.ones(1).to(self.device)
            self.cov_beta = torch.ones(1).to(self.device)
            return       
            
        vals, counts = np.unique(labels, return_counts=True)
        pruned_feats = deepcopy(feats)
        pruned_labels = deepcopy(labels)
        
        for v, c in zip(vals, counts):
            if c < self.num_cv_folds:
                pruned_feats = pruned_feats[pruned_labels != v]
                pruned_labels = pruned_labels[pruned_labels != v]
                
        try:
            skf = StratifiedKFold(n_splits=self.num_cv_folds, random_state=seed, shuffle=True)
            l_feats_te, l_labels_te = [], []
            l_means, l_covs = [], []
            for i, (train_index, test_index) in enumerate(skf.split(pruned_feats, pruned_labels)):
                feats_tr, labels_tr = pruned_feats[train_index], pruned_labels[train_index] 
                feats_te, labels_te = pruned_feats[test_index], pruned_labels[test_index] 
                means_tr, scatter_tr, priors, counts = self.compute_mle_statistics(feats=feats_tr, labels=labels_tr)
                cov_tr = (scatter_tr / (np.sum(counts) - 1)) + self.eps + torch.eye(self.D).to(self.device)
                cov_psd = cov_nearest(cov_tr.cpu().numpy(), method="clipped")
                cov_psd = torch.Tensor(cov_psd).to(self.device)
                means = torch.stack([means_tr[i] if means_tr[i] is not None else self.global_means[i] for i in range(self.num_classes)])
                l_means.append(means)
                l_covs.append(cov_psd)
                l_feats_te.append(feats_te)
                l_labels_te.append(labels_te)
        except Exception:
            self.means_beta = torch.zeros(1).to(self.device)
            self.cov_beta = torch.zeros(1).to(self.device)
            return

        loss_torch = lambda a: torch.sum(torch.stack([self.beta_classifier(a.clip(0, 1), 
                                                    l_means[i], 
                                                    l_covs[i], 
                                                    l_feats_te[i], 
                                                    l_labels_te[i])
                                                    for i in range(self.num_cv_folds)]))

        try: 
            if self.single_beta:
                x = torchmin.minimize(loss_torch, x0=0.5 * torch.ones(size=(1,)).to(self.device), method="l-bfgs", max_iter=10, options={"gtol": 1e-3}).x.cpu().clip(0, 1)
                self.means_beta = torch.ones_like(self.means_beta) * x[0]
                self.cov_beta = x[0]
            else:
                x = torchmin.minimize(loss_torch, x0=0.5 * torch.ones(size=(2,)).to(self.device), method="l-bfgs", max_iter=10, options={"gtol": 1e-3}).x.cpu().clip(0, 1)
                self.means_beta = torch.ones_like(self.means_beta) * x[0]
                self.cov_beta = x[1]
        except Exception:
            pass

    def update(self, means_mle, cov_mle):
        self.means_beta = self.means_beta.to(self.device)
        self.cov_beta = self.cov_beta.to(self.device)
        self.means.data = means_mle.data
        self.covariance.data = cov_mle.data
        self.adaptive_means.data = self.means_beta.unsqueeze(1) * means_mle.data + (1 - self.means_beta.unsqueeze(1)) * self.global_means.data
        self.adaptive_covariance.data = self.cov_beta * cov_mle.data + (1 - self.cov_beta) * self.global_covariance.data
        self.means_beta = self.means_beta.cpu()
        self.cov_beta = self.cov_beta.cpu()
    
    def compute_mle_statistics(self, split="train", feats=None, labels=None):
        self.model = self.model.to(self.device)
        means = [None] * self.num_classes
        if feats is None:
            feats, labels = self.compute_feats(split=split)
        counts = np.bincount(labels, minlength=self.num_classes)
        priors = np.bincount(labels, minlength=self.num_classes) / float(len(labels))
        priors = torch.Tensor(priors).float().to(self.device)
        
        for i, y in enumerate(np.unique(labels)):
            if (labels == y).sum() > self.min_samples:
                means[y] = torch.mean(feats[labels == y], axis=0)

        feats_centered = []
        for i, y in enumerate(np.unique(labels)):
            means_y = means[y] if means[y] is not None else self.global_means[y]
            f = feats[labels == y] - means_y
            feats_centered.append(f)
            
        feats_centered = torch.cat(feats_centered, dim=0)
        scatter = torch.mm(feats_centered.t(), feats_centered)
        return means, scatter, priors, counts
    
    def lda_classify(self, Z, means=None, covariance=None, priors=None, use_lstsq=True):
        if priors is None:
            priors = self.priors
        covariance = (1 - self.eps) * covariance + self.eps * torch.trace(covariance) / self.D * torch.eye(self.D).to(self.device)
        if use_lstsq:
            coefs = torch.linalg.lstsq(covariance, means.T)[0].T
        else:
            coefs = torch.matmul(torch.linalg.inv(covariance), means.T).T
        intercepts = -0.5 * torch.diag(torch.matmul(means, coefs.T)) + torch.log(priors).to(self.device)
        return Z @ coefs.T + intercepts

    def set_lda_weights(self, means=None, covariance=None, priors=None, use_lstsq=True):
        if means is None:
            means = self.means
        if covariance is None:
            covariance = self.covariance
        if priors is None:
            priors = self.priors
        with torch.no_grad():
            means = means.to(self.device)
            covariance = covariance.to(self.device)
            priors = priors.to(self.device)
            covariance = (1 - self.eps) * covariance + self.eps * torch.trace(covariance) / self.D * torch.eye(self.D).to(self.device)
            if use_lstsq:
                coefs = torch.linalg.lstsq(covariance, means.T)[0].T
            else:
                coefs = torch.matmul(torch.linalg.inv(covariance), means.T).T
            intercepts = -0.5 * torch.diag(torch.matmul(means, coefs.T)) + torch.log(priors).to(self.device)
            self.model.fc.weight.data = coefs.detach()
            self.model.fc.bias.data = intercepts.detach()

    _stat_keys = ["means", "covariance", "global_means", "global_covariance",
                  "adaptive_means", "adaptive_covariance", "means_beta", "cov_beta",
                  "priors", "class_counts"]

    def stats_state(self):
        """All Gaussian distribution estimates held by this client (CPU copies)."""
        return {k: getattr(self, k).detach().cpu().clone() for k in self._stat_keys}

    def load_stats_state(self, state):
        for k in self._stat_keys:
            setattr(self, k, state[k].clone())

    def _to_cpu(self):
        """Helper to safely offload client tensors to CPU memory."""
        self.model = self.model.to("cpu")
        self.means = self.means.to("cpu")
        self.covariance = self.covariance.to("cpu")
        self.adaptive_means = self.adaptive_means.to("cpu")
        self.adaptive_covariance = self.adaptive_covariance.to("cpu")
        self.global_means = self.global_means.to("cpu")
        self.global_covariance = self.global_covariance.to("cpu")