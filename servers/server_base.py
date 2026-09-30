from abc import ABC
import os
import numpy as np
import torch
from copy import deepcopy
from models.clip_wrapper import CLIPWrapper

class Server(ABC):
    def __init__(self, args):
        self.num_clients = args.num_clients
        self.model = deepcopy(args.model)
        self.global_rounds = args.global_rounds
        self.clients = []
        self.D = self.model.D
        self.num_classes = args.num_classes
        self.device = args.device
        self.active_clients = []
        self.sampling_prob = args.sampling_prob
        self.active_client_ids = []
        self.eval_gap = args.eval_gap
        self.train_times = []
        self.round_times = []
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_meta = {
            "method": args.method,
            "dataset": args.dataset,
            "partition_path": args.partition_path,
            "model_name": args.model_name,
            "num_clients": args.num_clients,
            "num_classes": args.num_classes,
            "D": self.D,
        }
        # CLIP backbone is frozen and reloaded from pretrained weights, so only the head (fc + adapter) is checkpointed
        self.save_head_only = isinstance(self.model, CLIPWrapper)
        if self.save_head_only:
            self.checkpoint_meta["clip_arch"] = args.clip_arch
            self.checkpoint_meta["normalize_features"] = args.normalize_features

    def checkpoint_state(self, round):
        state = {
            "meta": self.checkpoint_meta,
            "round": round,
            "model": self.model.trainable_state_dict() if self.save_head_only else self.model.state_dict(),
            "train_times": self.train_times,
            "round_times": self.round_times,
        }
        return state

    def load_checkpoint_state(self, state):
        if self.save_head_only:
            model_state = state["model"]
            if "weight" in model_state:
                # legacy checkpoints stored fc.state_dict() (+ a separate "probvlm" entry)
                model_state = {f"fc.{k}": v for k, v in model_state.items()}
                model_state.update({f"probvlm.{k}": v for k, v in state.get("probvlm", {}).items()})
            self.model.load_trainable_state_dict(model_state)
        else:
            self.model.load_state_dict(state["model"])
        self.train_times = state["train_times"]
        self.round_times = state["round_times"]

    def save_checkpoint(self, round, path=None):
        path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # write to a temp file first so an interrupted save never corrupts the latest checkpoint
        tmp_path = path + ".tmp"
        torch.save(self.checkpoint_state(round), tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path=None):
        path = path or self.checkpoint_path
        state = torch.load(path, map_location="cpu", weights_only=True)
        mismatched = {k: (state["meta"].get(k), v) for k, v in self.checkpoint_meta.items() if state["meta"].get(k) != v}
        if mismatched:
            raise ValueError(f"Checkpoint {path} does not match current config (checkpoint, current): {mismatched}")
        self.load_checkpoint_state(state)
        print(f"Loaded checkpoint from {path} (round {state['round']})")
        return state["round"]

    def send_models(self):
        for c in self.active_clients:
            c.set_model(self.model)

    def sample_active_clients(self):
        self.active_clients = []
        self.active_client_ids = []
        sampling_prob_tensor = torch.ones(self.num_clients)*self.sampling_prob
        selected_indices = (torch.bernoulli(sampling_prob_tensor).numpy()).astype(int)
        selected_indices = np.where(selected_indices==1)[0]
        for idx in np.unique(selected_indices):
            self.active_clients.append(self.clients[idx])
            self.active_client_ids.append(idx)

    def aggregate_models(self):
        total_samples = sum(c.num_train for c in self.active_clients)
    
        # only aggregate trainable params (fc + adapters); never the frozen CLIP encoder
        aggregated = lambda name, param: param.requires_grad and not name.startswith("encoder.")
        for name, param in self.model.named_parameters():
            if aggregated(name, param):
                param.data.zero_()

        for c in self.active_clients:
            for (name, global_param), client_param in zip(self.model.named_parameters(), c.model.parameters()):
                if aggregated(name, global_param):
                    global_param.data = global_param.data + (c.num_train / total_samples)*client_param.data

    def evaluate(self, only_active=False):
        clients = self.active_clients if only_active else self.active_clients
        total_samples = sum(c.num_test for c in clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in clients:
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std
    
    def evaluate_personalized(self):
        pass

    def train_clients(self):
        train_acc, train_loss = 0, 0
        num_total = sum(c.num_train for c in self.active_clients)
        for i, c in enumerate(self.active_clients):
            client_acc, client_loss = c.train()
            train_acc += (c.num_train / num_total) * client_acc
            train_loss += (c.num_train / num_total) * client_loss
        return train_acc, train_loss