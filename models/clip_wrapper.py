import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPWrapper(nn.Module):
    def __init__(self, num_classes=10, args=None):
        super(CLIPWrapper, self).__init__()
        
        # Load hyperparams from args
        clip_arch = args.clip_arch if args and hasattr(args, 'clip_arch') else "ViT-B/32"
        self.device = args.device if args and hasattr(args, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize_features = args.normalize_features if args and hasattr(args, 'normalize_features') else False
        
        # Load pre-trained CLIP model
        self.encoder, self.preprocess = clip.load(clip_arch, device=self.device)
        
        # Freeze the entire model
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        
        self.D = self.encoder.visual.output_dim
        self.cls = num_classes
        
        # Classification head for pFedFDA
        self.fc = nn.Linear(self.D, num_classes)

        # Placeholder for cached text features
        self.text_features = None

        # hardcode here for now
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.set_text_prompts(cifar10_classes)

    def train(self, mode=True):
        # The CLIP encoder is frozen and always kept in eval mode
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_state_dict(self):
        """State of the trainable head (fc + any adapter); the frozen CLIP encoder is never saved."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}

    def load_trainable_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("encoder.") for k in missing), \
            f"Head state mismatch (missing: {missing}, unexpected: {unexpected})"

    def _get_image_features(self, x):
        """Internal helper to handle upsampling and visual encoding."""
        if x.shape[-1] < 224 or x.shape[-2] < 224:
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            
        with torch.no_grad():
            feat = self.encoder.encode_image(x).float()
            if self.normalize_features:
                feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def forward(self, x, return_feat=False):
        """Standard forward pass for FDA (using the learned linear head)."""
        feat = self._get_image_features(x)
        out = self.fc(feat)
        
        if return_feat:
            return feat, out
        return out

    def set_text_prompts(self, class_names, template="a photo of a {}"):
        """
        Encodes and caches the text embeddings for the dataset classes.
        Call this once before evaluation or zero-shot tasks.
        """
        prompts = [template.format(c) for c in class_names]
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.encoder.encode_text(text_tokens).float()
            # Text features are strictly L2 normalized for cosine similarity
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
    def get_text_similarities(self, x):
        """
        Returns the (Batch, Num_Classes) similarity scores.
        This forms the basis for both Zero-Shot classification and Cross-Modal Density Estimation.
        """
        assert self.text_features is not None, "You must call set_text_prompts() before getting similarities."
        
        # Always normalize image features when comparing to text, regardless of FDA setting
        img_feat = self._get_image_features(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        # Cosine similarity scaled by CLIP's learned temperature
        logit_scale = self.encoder.logit_scale.exp()
        similarities = logit_scale * img_feat @ self.text_features.T
        
        # If doing Cross-Modal Density Estimation, you can return img_feat and self.text_features here too
        return similarities