from .resnet_gn import resnet8, resnet18, resnet50
from .cnn import CIFARNet, EMNISTNet
from .clip_wrapper import CLIPWrapper
from .probvlm import ProbVLMWrapper
from .adapter import AdapterWrapper

# frozen CLIP backbones: only the head (fc + adapter) is trained, aggregated, and saved
_clip_model_dict = {
    "clip": CLIPWrapper,
    "probvlm": ProbVLMWrapper,
    "adapter": AdapterWrapper,
}
CLIP_MODELS = list(_clip_model_dict.keys())

_base_model_dict = {
    "cifarnet": CIFARNet,
    "emnistnet": EMNISTNet,
    "resnet8": resnet8,
    "resnet18": resnet18,
    "resnet50": resnet50,
}

class ModelFactory:
    """
    A factory dictionary to handle instantiating models.
    Intercepts the 'args' parameter so legacy architectures don't throw unexpected keyword errors.
    """
    def __getitem__(self, key):
        if key in _clip_model_dict:
            # CLIP models need args for 'clip_arch' and 'normalize_features'
            return lambda num_classes, in_channels, args: _clip_model_dict[key](num_classes=num_classes, args=args)
        else:
            # Legacy models ignore args
            return lambda num_classes, in_channels, args: _base_model_dict[key](num_classes=num_classes, in_channels=in_channels)
            
    def __contains__(self, key):
        return key in _base_model_dict or key in _clip_model_dict

model_dict = ModelFactory()