import sys, os

# Get the absolute path of the project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
if root_path not in sys.path:
    sys.path.append(root_path)

from base_resnet import ResNet, BasicBlock
from git_rebasin.weight_matching import (apply_permutation, resnet18_permutation_spec, weight_matching, reverse_permutation_model)
from git_rebasin.utils import (flatten_params, unflatten_params)

from net2net import widen_resnet18, unwiden_resnet18

import torch.nn as nn
import numpy as np
from jax import random
import torch

def get_number_permutations_differences(perm1, perm2):
    differences = {}
    for key in perm1.keys():
        differences[key] = int(np.sum(perm1[key][:len(perm1[key])//2] != perm2[key][:len(perm1[key])//2]))
    return differences

def permute_resnet18(ref_model, model_to_permute):
    model_a_dict = ref_model.state_dict()
    model_b_dict = model_to_permute.state_dict()
    model_a_dict = {key: value.cpu().numpy() for key, value in model_a_dict.items()}
    model_b_dict = {key: value.cpu().numpy() for key, value in model_b_dict.items()}

    permutation_spec = resnet18_permutation_spec()
    final_permutation = weight_matching(random.PRNGKey(42), permutation_spec,
                                        flatten_params(model_a_dict), flatten_params(model_b_dict), silent=True)

    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation, flatten_params(model_b_dict)))

    model_b_clever = {key: torch.from_numpy(np.array(value)) for key, value in model_b_clever.items()}

    model_to_permute.load_state_dict(model_b_clever)

    return model_to_permute, permutation_spec, final_permutation

def main():
    model_a = ResNet(block=BasicBlock, num_blocks=[2,2,2,2], norm_layer=nn.BatchNorm2d, hidden_sizes=[32, 64, 128, 256], n_class=100)
    model_b = ResNet(block=BasicBlock, num_blocks=[2,2,2,2], norm_layer=nn.BatchNorm2d, hidden_sizes=[32, 64, 128, 256], n_class=100)
    model_c = ResNet(block=BasicBlock, num_blocks=[2,2,2,2], norm_layer=nn.BatchNorm2d, hidden_sizes=[16, 32, 64, 128], n_class=100)
    model_c_widen = widen_resnet18(model_c, [32, 64, 128, 256])[0]

    # original_model_c_dict = unwiden_resnet18(model_c, model_c_widen)

    # tot=0
    # for key in model_c.state_dict().keys():
    #     share = (torch.sum(model_c.state_dict()[key] == original_model_c_dict[key])) / (torch.sum(model_c.state_dict()[key] == original_model_c_dict[key]) + torch.sum(model_c.state_dict()[key] != original_model_c_dict[key]))
    #     tot += share
    # tot /=len(model_c.state_dict().keys())

    ignored_keys = {"norm4.weight", "norm4.bias"}
    checkpoint_a = torch.load('C:/Users/boite/Desktop/small_to_big/FIARSE_FedRolex/benchmark_logs/fiarse_cifar100_100_500_1_[1.0]_[100]_[32, 64, 128, 256]_1.0_epoch_500.pt', map_location="cpu")
    filtered_state_dict_a = {k: v for k, v in checkpoint_a.items() if k not in ignored_keys}
    # checkpoint_b = torch.load('C:/Users/boite/Desktop/small_to_big/FIARSE_FedRolex/benchmark_logs/fedrolex_cifar100_100_500_1_[1.0]_[100]_[32, 64, 128, 256]_1.0_epoch_500.pt', map_location="cpu")
    # filtered_state_dict_b = {k: v for k, v in checkpoint_b.items() if k not in ignored_keys}
    model_a.load_state_dict(filtered_state_dict_a)
    # model_b.load_state_dict(filtered_state_dict_b)

    model_a_dict = model_a.state_dict()
    model_b_dict = model_b.state_dict()
    model_a_dict = {key: value.cpu().numpy() for key, value in model_a_dict.items()}
    model_b_dict = {key: value.cpu().numpy() for key, value in model_b_dict.items()}

    permutation_spec = resnet18_permutation_spec()
    final_permutation = weight_matching(random.PRNGKey(42), permutation_spec,
                                        flatten_params(model_a_dict), flatten_params(model_b_dict), silent=True)

    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation, flatten_params(model_b_dict)))

    model_b_clever = {key: torch.from_numpy(np.array(value)) for key, value in model_b_clever.items()}
    
    model_b_unpermuted = reverse_permutation_model(permutation_spec, final_permutation, flatten_params(model_b_clever))

    tot = 0
    for key in model_b_dict.keys():
        share = (np.sum(model_b_dict[key] == model_b_unpermuted[key])) / (np.sum(model_b_dict[key] == model_b_unpermuted[key]) + np.sum(model_b_dict[key] != model_b_unpermuted[key]))
        tot += share
    tot /= len(model_b_dict.keys())

    model_b.load_state_dict(model_b_clever)

    # torch.save(model_b.state_dict(), 'C:/Users/boite/Desktop/FIARSE_FedRolex/model_fedrolex_2_permuted.pt')

if __name__ == "__main__":
    main()