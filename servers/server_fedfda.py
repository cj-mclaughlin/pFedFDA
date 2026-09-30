import os
import csv
import time
from copy import deepcopy
import torch
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from servers.server_base import Server
from clients.client_fedfda import ClientFedFDA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

class ServerFedFDA(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = [
            ClientFedFDA(args, i) for i in range(self.num_clients)
        ]
        self.args = args  # kept for constructing new (OOD) clients after training
        self.global_means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.global_covariance = torch.Tensor(torch.eye(self.D))
        self.global_priors = torch.ones(self.num_classes) / self.num_classes
        self.r = 0

    def train(self):
        for r in range(1, self.global_rounds + 1):
            start_time = time.time()
            self.r = r
            if r == self.global_rounds: # Full participation on final round
                self.sampling_prob = 1.0
            self.sample_active_clients()
            self.send_models()

            # Train active clients
            train_acc, train_loss = self.train_clients()
            train_time = time.time() - start_time
            
            # Aggregate global Gaussian distribution statistics
            self.aggregate_models()
            
            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

            # Periodic logging
            if r % self.eval_gap == 0 or r == self.global_rounds:
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()  
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Test Loss [{ptest_loss:.4f}]\t Test Acc [{ptest_acc:.2f} (±{ptest_acc_std:.2f})]\t Train Time [{train_time:.2f}s]")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Train Time [{train_time:.2f}s]")

            # Overwrite latest checkpoint (after eval, which also updates client statistics)
            self.save_checkpoint(r)

    def checkpoint_state(self, round):
        state = super().checkpoint_state(round)
        state["global_means"] = self.global_means.detach().cpu().clone()
        state["global_covariance"] = self.global_covariance.detach().cpu().clone()
        state["global_priors"] = self.global_priors.detach().cpu().clone()
        state["clients"] = [c.stats_state() for c in self.clients]
        return state

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        self.global_means = state["global_means"]
        self.global_covariance = state["global_covariance"]
        self.global_priors = state["global_priors"]
        for c, c_state in zip(self.clients, state["clients"]):
            c.load_stats_state(c_state)
        self.r = state["round"]

    def aggregate_models(self):
        # Aggregate base feature extractor weights
        super().aggregate_models()

        # Aggregate class Gaussian estimates weighted by client sample counts
        total_samples = sum(c.num_train for c in self.active_clients)
        self.global_means.data = torch.zeros_like(self.clients[0].means)
        self.global_covariance.data = torch.zeros_like(self.clients[0].covariance)
        
        for c in self.active_clients:
            weight = c.num_train / total_samples
            self.global_means.data += weight * c.adaptive_means.data
            self.global_covariance.data += weight * c.adaptive_covariance.data
    
    def send_models(self):
        super().send_models()
        # Broadcast global Gaussian statistics to active clients
        for c in self.active_clients:
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            if self.r == 1:
                c.means.data = self.global_means.data
                c.covariance.data = self.global_covariance.data
                c.adaptive_means.data = self.global_means.data
                c.adaptive_covariance.data = self.global_covariance.data

    def fit_client_statistics(self, c):
        """Solves the client's interpolation beta and updates its adaptive statistics from its training split."""
        c_feats, c_labels = c.compute_feats(split="train")
        c.solve_beta(feats=c_feats, labels=c_labels)

        means_mle, scatter_mle, priors, counts = c.compute_mle_statistics(feats=c_feats, labels=c_labels)
        means_mle = torch.stack([means_mle[i] if means_mle[i] is not None and counts[i] > c.min_samples else c.global_means[i] for i in range(self.num_classes)])
        cov_mle = (scatter_mle / (np.sum(counts) - 1)) + 1e-4 + torch.eye(self.D).to(self.device)
        cov_psd = cov_nearest(cov_mle.cpu().numpy(), method="clipped")
        cov_psd = torch.Tensor(cov_psd).to(self.device)

        c.update(means_mle, cov_psd)

    def evaluate_personalized(self):
        """
        Evaluates personalized FDA classifiers across clients.
        Interpolates local and global statistics via cross-validated \beta.
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

            # Solve local beta on client training split
            self.fit_client_statistics(c)
            c.set_lda_weights(c.adaptive_means, c.adaptive_covariance)
            
            with torch.no_grad():
                acc, loss = c.evaluate()
                accs.append(acc)
                weighted_loss += (c.num_test / total_samples) * loss.detach()
                weighted_acc += (c.num_test / total_samples) * acc
                c.model = old_model
            c._to_cpu()
            
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std

    def evaluate_checkpoint(self):
        """
        Evaluates each client's personalized FDA classifier directly from its stored
        adaptive statistics (no re-fitting), e.g. to sanity-check a loaded checkpoint.
        """
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []

        for c in self.clients:
            c.model = deepcopy(self.model)
            c.set_lda_weights(c.adaptive_means, c.adaptive_covariance)
            with torch.no_grad():
                acc, loss = c.evaluate()
                accs.append(acc)
                weighted_loss += (c.num_test / total_samples) * loss.detach()
                weighted_acc += (c.num_test / total_samples) * acc
            c._to_cpu()

        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std

    def evaluate_zero_shot(self):
        """Phase 3: Zero-shot CLIP evaluation across all clients."""
        print("\n=== Running Zero-Shot CLIP Evaluation ===")
        total_samples = sum(c.num_test for c in self.clients)
        weighted_acc = 0.0
        weighted_loss = 0.0

        for c in self.clients:
            c.model = deepcopy(self.model)
            acc, loss = c.evaluate_zero_shot()
            weighted_acc += (c.num_test / total_samples) * acc
            weighted_loss += (c.num_test / total_samples) * loss

        print(f"Zero-Shot Test Accuracy: {weighted_acc:.2f}% | Test Loss: {weighted_loss:.4f}\n")
        return weighted_acc, weighted_loss

    def compute_goodness_of_fit(self, save_path="gaussian_fit_2d.png"):
        """Phase 1: Diagnostic suite measuring Gaussian fit across client feature sets."""
        print("\n=== Evaluating Local Feature Distribution Goodness of Fit ===")
        all_metrics = []
        vis_features = []
        vis_labels = []

        for i, c in enumerate(self.clients):
            c.model = deepcopy(self.model)
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            
            metrics = c.evaluate_goodness_of_fit()
            all_metrics.append(metrics)

            # Collect features from the first few clients to avoid massive visual clutter
            if i < 5: 
                c.model = c.model.to(c.device)
                with torch.no_grad():
                    feats, labels = c.compute_feats(split="train")
                    vis_features.append(feats.cpu())
                    vis_labels.append(labels)
                c._to_cpu()

        avg_ll = np.nanmean([m["avg_log_likelihood"] for m in all_metrics])
        avg_skew = np.nanmean([m["mardia_skewness"] for m in all_metrics])
        avg_kurt = np.nanmean([m["mardia_kurtosis"] for m in all_metrics])
        expected_kurt = all_metrics[0]["expected_kurtosis"]

        print(f"Average Gaussian Log-Likelihood: {avg_ll:.4f}")
        print(f"Mardia's Skewness (Ideal = 0): {avg_skew:.4f}")
        print(f"Mardia's Kurtosis (Observed vs Expected): {avg_kurt:.2f} vs {expected_kurt:.2f}")

        # --- 2D PCA Projection and Visualization ---
        print(f"Generating 2D PCA visualization... saving to '{save_path}'\n")
        
        vis_features = torch.cat(vis_features, dim=0).numpy()
        vis_labels = np.concatenate(vis_labels, axis=0)

        # Fit PCA on the aggregated features
        pca = PCA(n_components=2)
        feats_2d = pca.fit_transform(vis_features)
        
        # Get the PCA projection matrix P (shape: 2 x D)
        P = pca.components_

        # Project global means and covariance to 2D
        global_means_np = self.global_means.cpu().numpy()
        global_cov_np = self.global_covariance.cpu().numpy()

        # Transform means using the fitted PCA (handles mean centering automatically)
        means_2d = pca.transform(global_means_np) 
        
        # Project the shared covariance matrix: Sigma_2D = P * Sigma_D * P^T
        cov_2d = P @ global_cov_np @ P.T

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("tab10")

        for cls in range(self.num_classes):
            idx = vis_labels == cls
            if not idx.any(): continue
            
            # Plot the projected client data points
            ax.scatter(feats_2d[idx, 0], feats_2d[idx, 1], alpha=0.3, s=15, color=cmap(cls), label=f"Class {cls}")
            
            # In pFedFDA, covariance is shared across all classes, so we reuse cov_2d
            mu = means_2d[cls]
            
            # Calculate eigenvalues and eigenvectors to draw the ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            # Calculate the angle of the ellipse
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # Width and height for 2 standard deviations (~86% of mass in 2D)
            # 2 * (number of std devs) * sqrt(eigenvalue)
            width = 4 * np.sqrt(eigenvalues[0])
            height = 4 * np.sqrt(eigenvalues[1])
            
            # Plot the 2-std-dev contour
            ell = Ellipse(xy=mu, width=width, height=height, angle=angle, 
                          edgecolor=cmap(cls), facecolor='none', linewidth=2, linestyle='--')
            ax.add_patch(ell)
            
            # Plot the class mean as a large cross
            ax.scatter(mu[0], mu[1], marker='X', s=100, color=cmap(cls), edgecolor='black', zorder=5)

        ax.set_title("2D PCA Projection of Features and Global Gaussian Estimates (2 Std Dev)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        # Put legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return {"log_likelihood": avg_ll, "skewness": avg_skew, "kurtosis": avg_kurt}

    def corrupted_clients(self, client_ids):
        """Lazily builds CIFAR-C corrupted versions of the given client partitions (corruption/severity assigned as with --augmented)."""
        for idx in client_ids:
            yield ClientFedFDA(self.args, idx, corrupted=True)

    def evaluate_clients(self, clients, population):
        """
        Evaluates each client with the final global model, returning one result row per client.
          global: LDA from the global Gaussian statistics (no adaptation, uniform priors)
          local:  personalized LDA from beta-interpolated local/global statistics fit on the client's train split
        """
        rows = []
        for c in clients:
            c.model = deepcopy(self.model)
            c.global_means = self.global_means.clone().to(self.device)
            c.global_covariance = self.global_covariance.clone().to(self.device)
            c.model.eval()

            with torch.no_grad():
                c.set_lda_weights(c.global_means, c.global_covariance, self.global_priors)
                global_acc, global_loss = c.evaluate()

            self.fit_client_statistics(c)
            with torch.no_grad():
                c.set_lda_weights(c.adaptive_means, c.adaptive_covariance)
                local_acc, local_loss = c.evaluate()
            c._to_cpu()

            rows.append({
                "population": population,
                "client_idx": c.client_idx,
                "augmentation": c.augmentation,
                "severity": c.severity,
                "num_test": c.num_test,
                "global_acc": global_acc.item(),
                "global_loss": global_loss.item(),
                "local_acc": local_acc.item(),
                "local_loss": local_loss.item(),
            })
        return rows

    def evaluate_populations(self, populations, out_dir):
        """
        Evaluates each named population of clients ({name: iterable of clients}) and writes
        per-client results (eval_clients.csv) and per-population summaries (eval_summary.csv) to out_dir.
        """
        all_rows, summary_rows = [], []
        for population, clients in populations.items():
            print(f"\n=== Evaluating population: {population} ===")
            rows = self.evaluate_clients(clients, population)
            all_rows += rows

            weights = np.array([r["num_test"] for r in rows]) / sum(r["num_test"] for r in rows)
            summary = {"population": population, "num_clients": len(rows)}
            for k in ["global_acc", "local_acc", "global_loss", "local_loss"]:
                vals = np.array([r[k] for r in rows])
                summary[k] = float(np.sum(weights * vals))
                summary[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary_rows.append(summary)
            print(f"[{population}] Test Acc [global|local]: [{summary['global_acc']:.2f} (±{summary['global_acc_std']:.2f}) | {summary['local_acc']:.2f} (±{summary['local_acc_std']:.2f})]")

        os.makedirs(out_dir, exist_ok=True)
        for fname, rows in [("eval_clients.csv", all_rows), ("eval_summary.csv", summary_rows)]:
            path = os.path.join(out_dir, fname)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {path}")
        return summary_rows
