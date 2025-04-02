import sys, os

# Get the absolute path of the project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
if root_path not in sys.path:
    sys.path.append(root_path)

from git_rebasin.weight_matching import *
from git_rebasin.utils import (flatten_params, unflatten_params)
from utils import average_models, evaluate_model, compare_model_performances

import numpy as np
from jax import random
import torch

def get_share_permutations_differences(perm1, perm2, factor):
    differences = {}
    for key in perm1.keys():
        differences[key] = int(np.sum(perm1[key][:len(perm1[key])//factor] != perm2[key][:len(perm1[key])//factor])) / len(perm1[key][:len(perm1[key])//factor])
    return differences

def permute_resnet(ref_model, model_to_permute, resnet_model: str):
    model_a_dict = ref_model.state_dict()
    model_b_dict = model_to_permute.state_dict()
    model_a_dict = {key: value.cpu().numpy() for key, value in model_a_dict.items()}
    model_b_dict = {key: value.cpu().numpy() for key, value in model_b_dict.items()}

    match resnet_model:
        case '18':
            permutation_spec = resnet18_permutation_spec()
        case '34':
            permutation_spec = resnet34_permutation_spec()        
        case '50':
            permutation_spec = resnet50_permutation_spec()        
        case '101':
            permutation_spec = resnet101_permutation_spec()
        case '152':
            permutation_spec = resnet152_permutation_spec()       
        case _:
            raise Exception(f'The following ResNet architecture is not supported for permutation: {resnet_model}')

    final_permutation = weight_matching(random.PRNGKey(42), permutation_spec,
                                        flatten_params(model_a_dict), flatten_params(model_b_dict), silent=True)

    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation, flatten_params(model_b_dict)))

    model_b_clever = {key: torch.from_numpy(np.array(value)) for key, value in model_b_clever.items()}

    model_to_permute.load_state_dict(model_b_clever)

    return model_to_permute, permutation_spec, final_permutation

def reverse_permutation_model(ps: PermutationSpec, perm, params):
  return {k: torch.from_numpy(np.array(get_reverse_permuted_param(ps, perm, k, params))) for k in params.keys()}

def test_permutation(model1, model2, dataloader, resnet_model: str, nb_batches=100):
    averaged_model = average_models([model1, model2])
    print('Averaged Model Before Permutation:')
    evaluate_model(dataloader, averaged_model, nb_batches, print_=True)

    acc1, acc2 = compare_model_performances(dataloader, model1, model2, nb_batches, print_=False)
    if acc1 >= acc2:
        model2, _, _ = permute_resnet(model1, model2, resnet_model)
    else:
        model1, _, _ = permute_resnet(model2, model1, resnet_model)

    averaged_model = average_models([model1, model2])

    print('Averaged Model After Permutation:')
    
    evaluate_model(dataloader, averaged_model, nb_batches, print_=True)