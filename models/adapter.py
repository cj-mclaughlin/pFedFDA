import torch.nn as nn
from .clip_wrapper import CLIPWrapper
from .probvlm import PROBVLM_KWARGS


class AdapterWrapper(CLIPWrapper):
    """
    Frozen CLIP + residual MLP adapter on the final image features (feat + MLP(feat)), trained with the standard
    local objective. Uses the same hidden width as the ProbVLM adapter; the output matches the CLIP feature dim.
    """
    def __init__(self, num_classes=10, args=None):
        super(AdapterWrapper, self).__init__(num_classes=num_classes, args=args)
        hid_dim = PROBVLM_KWARGS["hid_dim"]
        self.adapter = nn.Sequential(
            nn.Linear(self.D, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, self.D),
        )

    def forward(self, x, return_feat=False):
        feat = self._get_image_features(x)
        feat = feat + self.adapter(feat)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out
