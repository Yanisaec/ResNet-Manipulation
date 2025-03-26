import sys, os

# Get the absolute path of the project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
if root_path not in sys.path:
    sys.path.append(root_path)

from collections import defaultdict
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment
import torch
from git_rebasin.utils import rngmix

class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def resnet18_permutation_spec() -> PermutationSpec:
    # conv = lambda name, p_in, p_out: {f"{name}.weight": (None, None, p_in, p_out)}
    # norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    # dense = lambda name, p_in, p_out: {f"{name}.weight": (p_in, p_out), f"{name}.bias": (p_out,)}
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
    norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
    
    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.norm1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.norm2", p),
    }

    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.norm1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.norm2", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),  # Mapping shortcut conv layer
        **norm(f"{name}.shortcut.1", p_out),  # Mapping shortcut BN layer
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("norm1", "P_bg0"),
        
        **easyblock("layer1.0", "P_bg0"),
        **easyblock("layer1.1", "P_bg0"),
        
        **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
        **easyblock("layer2.1", "P_bg1"),
        
        **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
        **easyblock("layer3.1", "P_bg2"),
        
        **shortcutblock("layer4.0", "P_bg2", "P_bg3"),
        **easyblock("layer4.1", "P_bg3"),
        
        **dense("linear", "P_bg3", None),
    })

def reverse_permutation_list(permutation):
  inverse_perm = [0] * len(permutation)
  for i, p in enumerate(permutation):
      inverse_perm[p] = i
  inverse_perm = jnp.array(inverse_perm)
  return inverse_perm

# def reverse_permutation(vector, permutation, axis=0):
#   inverse_perm = [0] * len(permutation)
#   for i, p in enumerate(permutation):
#       inverse_perm[p] = i
  
#   inverse_perm = jnp.array(inverse_perm)
  
#   ndim = vector.ndim
#   idx = tuple(inverse_perm if i == axis else jnp.arange(vector.shape[i]) for i in range(ndim))
  
#   grid = jnp.meshgrid(*idx, indexing='ij')
#   try:
#     result = vector[tuple(grid)]
#   except:
#     breakpoint()
#   return result
  
def get_reverse_permuted_param(ps: PermutationSpec, perm, k: str, params):
  w = params[k]
  w = jnp.array(w)
  for axis, p in enumerate(ps.axes_to_perm[k]):
    if p is not None:
      # w = reverse_permutation(w, perm[p], axis)
      w = jnp.take(w, reverse_permutation_list(perm[p]), axis=axis)
  return w

def reverse_permutation_model(ps: PermutationSpec, perm, params):
  return {k: torch.from_numpy(np.array(get_reverse_permuted_param(ps, perm, k, params))) for k in params.keys()}

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      w = jnp.take(w, perm[p], axis=axis)

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(rng,
                    ps: PermutationSpec,
                    params_a,
                    params_b,
                    max_iter=100,
                    init_perm=None,
                    silent=False):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())

  for iteration in range(max_iter):
    progress = False
    for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = jnp.zeros((n, n))
      for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a[wk]
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
        A += w_a @ w_b.T

      ri, ci = linear_sum_assignment(A, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()

      oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
      newL = jnp.vdot(A, jnp.eye(n)[ci, :])
      if not silent: print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = jnp.array(ci)

    if not progress:
      break
  return perm

if __name__ == "__main__":
  def test_reverse_permutation():
      shape = (4, 3, 2, 5)
      test_array = jnp.arange(np.prod(shape)).reshape(shape)
      
      permutation = [1, 0, 3, 2]
      manual_permuted = jnp.stack([test_array[p] for p in permutation])
      restored_array = reverse_permutation(manual_permuted, permutation, axis=0)
      is_equal = jnp.array_equal(test_array, restored_array)
      print(f"Test for axis 0: {'Passed' if is_equal else 'Failed'}")

      permutation = [1, 0, 2]
      manual_permuted = jnp.stack([test_array[:,p] for p in permutation], axis=1)
      restored_array = reverse_permutation(manual_permuted, permutation, axis=1)
      is_equal = jnp.array_equal(test_array, restored_array)
      print(f"Test for axis 1: {'Passed' if is_equal else 'Failed'}")

      permutation = [1, 0]
      manual_permuted = jnp.stack([test_array[:,:,p] for p in permutation], axis=2)
      restored_array = reverse_permutation(manual_permuted, permutation, axis=2)
      is_equal = jnp.array_equal(test_array, restored_array)
      print(f"Test for axis 2: {'Passed' if is_equal else 'Failed'}")

  # Run the test
  test_reverse_permutation()