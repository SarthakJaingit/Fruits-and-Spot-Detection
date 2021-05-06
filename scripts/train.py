import dataset
import fruit_dataframe
import warnings
import torch

#git clone this repo from https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from ranger import Ranger


def create_dataloaders(prep_noiseloader = True, train_batch_size = 2, test_batch_size = 2):

    if prep_noiseloader:
        noise_path = "ScriptDataset/noisy_dataset"
        noise_dataset = dataset.NoiseDataset(noise_path, 100, 512)
        noise_loader = torch.utils.data.DataLoader(noise_dataset, batch_size = test_batch_size, shuffle = True)

    else:
        warnings.warn("Not loading with noise loader, hence built in Model function will error")
        noise_path = None

    bounding_box_dict, labels_dict = fruit_dataframe.get_dict(["Placeholder", "Apples", "Strawberry", "Apple_Bad_Spot", "Strawberry_Bad_Spot"])
    train_dataset = dataset.FruitDetectDataset(labels_dict, bounding_box_dict, dataset.get_transforms(mode = "train"),
                                        mode = "train", noisy_dataset_path=noise_path)
    test_dataset = dataset.FruitDetectDataset(labels_dict, bounding_box_dict, dataset.get_transforms(mode = "test"), mode = "test",
                                        noisy_dataset_path=noise_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, collate_fn= collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True, collate_fn= collate_fn)

    print("Created Data Loader generators with specified batch size")
    if prep_noiseloader:
        return train_loader, test_loader, noise_loader
    else:
        return train_loader, test_loader

def collate_fn(batch):
  return tuple([list(a) for a in zip(*batch)])

''' Foret, Pierre, et al. “Sharpness-Aware Minimization for Efficiently Improving Generalization.”
ArXiv.org, 29 Apr. 2021, arxiv.org/abs/2010.01412. '''

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


if __name__ == "__main__":
    train_loader, test_loader, noise_loader = create_dataloaders()
