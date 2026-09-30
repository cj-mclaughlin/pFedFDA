"""
ProbVLM probabilistic adapter for frozen CLIP (Upadhyay et al., ICCV 2023).
Adapter and losses migrated from https://github.com/ExplainableML/ProbVLM (src/networks.py, src/losses.py).
"""
import torch
import torch.nn as nn
from torch import Tensor
from .clip_wrapper import CLIPWrapper


class BayesCap_MLP(nn.Module):
    '''
    Residual MLP predicting the (mu, 1/alpha, beta) parameters of a generalized Gaussian
        inp_dim: int, Input dimension
        out_dim: int, Output dimension (must equal inp_dim for the residual connection)
        hid_dim: int, hidden dimension (trunk and mu/alpha/beta heads)
        num_layers: Number of hidden layers
        p_drop: dropout probability
    '''
    def __init__(self, inp_dim, out_dim, hid_dim=512, num_layers=1, p_drop=0):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            if layer == 0:
                mod.append(nn.Linear(inp_dim, hid_dim))
                mod.append(nn.ReLU())
            elif layer == num_layers // 2:
                mod.append(nn.Linear(hid_dim, hid_dim))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
            elif layer == num_layers - 1:
                mod.append(nn.Linear(hid_dim, out_dim))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.ReLU(),
        )
        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x_intr = self.mod(x) + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta


class BayesCap_for_CLIP(nn.Module):
    def __init__(self, inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.1):
        super(BayesCap_for_CLIP, self).__init__()
        self.img_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)

    def forward(self, i_features, t_features):
        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)
        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta)


class GenGaussLoss(nn.Module):
    """Negative log-likelihood of the target under the predicted generalized Gaussian."""
    def __init__(self, reduction='mean', alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3):
        super(GenGaussLoss, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def forward(self, mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor):
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps

        resi = torch.abs(mean - target)
        resi = (resi * one_over_alpha1 * beta1).clamp(min=self.resi_min, max=self.resi_max)

        log_one_over_alpha = torch.log(one_over_alpha1)
        log_beta = torch.log(beta1)
        lgamma_beta = torch.lgamma(torch.pow(beta1, -1))

        l = resi - log_one_over_alpha + lgamma_beta - log_beta
        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()
        raise ValueError(f"Reduction {self.reduction} not supported")


class TempCombLoss(nn.Module):
    """T1 * L1(mean, target) + T2 * generalized Gaussian NLL."""
    def __init__(self, reduction='mean', alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3):
        super(TempCombLoss, self).__init__()
        self.L_GenGauss = GenGaussLoss(
            reduction=reduction,
            alpha_eps=alpha_eps, beta_eps=beta_eps,
            resi_min=resi_min, resi_max=resi_max
        )
        self.L_l1 = nn.L1Loss(reduction=reduction)

    def forward(self, mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor, T1: float, T2: float):
        l1 = self.L_l1(mean, target)
        l2 = self.L_GenGauss(mean, one_over_alpha, beta, target)
        return T1 * l1 + T2 * l2


# adapter hyperparameters from the ProbVLM CLIP notebook
PROBVLM_KWARGS = dict(hid_dim=256, num_layers=3, p_drop=0.05)


class ProbVLMWrapper(CLIPWrapper):
    """
    Frozen CLIP + ProbVLM adapter. The adapter's predicted image mean (mu) is used as the
    feature for the classification head / FDA statistics.
    """
    def __init__(self, num_classes=10, args=None):
        super(ProbVLMWrapper, self).__init__(num_classes=num_classes, args=args)
        self.probvlm = BayesCap_for_CLIP(inp_dim=self.D, out_dim=self.D, **PROBVLM_KWARGS)

    def encode_dist(self, x):
        """Returns the frozen CLIP image features and the adapter's (mu, 1/alpha, beta)."""
        feat = self._get_image_features(x)
        return feat, self.probvlm.img_BayesCap(feat)

    def forward(self, x, return_feat=False):
        _, (mu, _, _) = self.encode_dist(x)
        out = self.fc(mu)
        if return_feat:
            return mu, out
        return out
