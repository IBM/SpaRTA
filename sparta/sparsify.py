import os
import torch


class SparsifyModel:
    def __init__(self, model, sparsity, frozen_modules=['embed_tokens']):

        assert (0 < sparsity < 1)

        self.model = model
        self.sparsity = sparsity
        self.frozen_modules = frozen_modules

        nonfrozen_keep_prob = self.compute_keep_prob(sparsity)

        indices, deltas = {}, {}
        for name, param in model.named_parameters():

            param.requires_grad = False

            if name == 'score.weight':
                keep_prob = 1.0
            elif any(k in name for k in self.frozen_modules):
                keep_prob = 0.0
            else:
                keep_prob = nonfrozen_keep_prob
            ind_dtype = torch.int32  # if max(param.shape) > 32767 else torch.int16
            indices[name] = param.bernoulli(keep_prob).nonzero().to(ind_dtype)
            deltas[name] = torch.nn.Parameter(torch.zeros(indices[name].shape[0],
                                                          dtype=param.dtype,
                                                          device=param.device))

        self.train_params = {'indices': indices, 'deltas': deltas}

        self.param_names = list(indices.keys())

        self.merged = False

        self.device = model.device
        if hasattr(model, 'hf_device_map'):
            self.hf_device_map = model.hf_device_map

        self.train()

        if hasattr(model, 'num_labels'):
            self.num_labels = model.num_labels

    def __repr__(self):
        return (f"SparseModel(sparsity={self.sparsity},"
                f" frozen_modules={self.frozen_modules},"
                f"\n  {self.model}\n)")

    def compute_keep_prob(self, sparsity):
        total = sum(p.numel() for p in self.model.base_model.parameters())
        frozen = sum(p.numel() for name, p in self.model.base_model.named_parameters()
                     if any(k in name for k in self.frozen_modules))
        keep = (1 - sparsity) * total
        keep_prob = keep / (total - frozen)
        if keep_prob >= 1.:
            msg = f"{sparsity=} is too low\n"
            msg += f"Base model has {total} params, and {frozen} of those are frozen.\n"
            msg += f"This leaves {total-frozen} non-frozen params for sparsification.\n"
            msg += f"But you expect to keep {keep:.0f} of the base model params as trainable.\n"
            msg += f"This is more than available non-frozen params. Choose: sparsity > {frozen/total:.6f}"
            raise ValueError(msg)
        return keep_prob

    def parameters(self):
        return iter(self.train_params['deltas'].values())

    def num_parameters(self):
        return sum(param.numel() for param in self.parameters())

    def print_trainable_parameters(self):
        n_trainable = self.num_parameters()
        pct = n_trainable/self.model.num_parameters()*100
        print(f"Num trainable parameters: {n_trainable:,d} ({pct:.0f}%)")

    def merge_deltas(self):
        # add sparse deltas
        for name, param in self.model.named_parameters():
            indices = self.train_params['indices'][name].int().unbind(1)
            param[indices] += self.train_params['deltas'][name]
        self.merged = True

    @torch.no_grad()
    def unmerge_deltas(self):
        # subtract deltas
        for name, param in self.model.named_parameters():
            param.detach_()
            indices = self.train_params['indices'][name].int().unbind(1)
            param[indices] -= self.train_params['deltas'][name]
        self.merged = False

    def __call__(self, *args, **kwargs):  # forward

        if self.training and self.merged:
            self.unmerge_deltas()

        if not self.merged:
            self.merge_deltas()

        out = self.model(*args, **kwargs)

        return out

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        if not self.merged:
            self.merge_deltas()
        return self.model.generate(*args, **kwargs)

    @torch.no_grad()
    def save(self, save_dir, merged=True):
        if merged:
            if not self.merged:
                self.merge_deltas()

            self.model.save_pretrained(save_dir)
            print('\nSparse model saved as a fully merged model in: %s' % save_dir)

        else:
            if self.merged:
                self.unmerge_deltas()

            torch.save(self.train_params,
                       os.path.join(save_dir, 'sparse_deltas.pt'))
            torch.save(self.model.score.weight.data,
                       os.path.join(save_dir, 'head_init.pt'))
            self.model.config.save_pretrained(save_dir)
            print('\nModel sparse delta parameters saved in: %s' % save_dir)

    def train(self):
        self.model.train()
        if self.merged:
            self.unmerge_deltas()
        self.training = True

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        if not self.merged:
            self.merge_deltas()
        self.training = False

    @torch.no_grad()
    def to(self, device):
        if hasattr(self, 'hf_device_map'):
            print(f"Model wasn't moved to {device=} because is distributed over a 'device_map'")
            return None
        device = torch.device(device)
        if self.device == device:
            return None
        self.model.to(device)
        for k in self.param_names:
            self.train_params['indices'][k] = self.train_params['indices'][k].to(device)
            self.train_params['deltas'][k] = torch.nn.Parameter(self.train_params['deltas'][k].to(device))
        self.device = device
