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
        **conv(f"{name}.shortcut.0", p_in, p_out),  
        **norm(f"{name}.shortcut.1", p_out),  
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

def resnet34_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
  norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
  
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
      **conv(f"{name}.shortcut.0", p_in, p_out),
      **norm(f"{name}.shortcut.1", p_out),
  }
  
  return permutation_spec_from_axes_to_perm({
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      
      # Layer 1 (3 blocks)
      **easyblock("layer1.0", "P_bg0"),
      **easyblock("layer1.1", "P_bg0"),
      **easyblock("layer1.2", "P_bg0"),
      
      # Layer 2 (4 blocks)
      **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
      **easyblock("layer2.1", "P_bg1"),
      **easyblock("layer2.2", "P_bg1"),
      **easyblock("layer2.3", "P_bg1"),
      
      # Layer 3 (6 blocks)
      **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
      **easyblock("layer3.1", "P_bg2"),
      **easyblock("layer3.2", "P_bg2"),
      **easyblock("layer3.3", "P_bg2"),
      **easyblock("layer3.4", "P_bg2"),
      **easyblock("layer3.5", "P_bg2"),
      
      # Layer 4 (3 blocks)
      **shortcutblock("layer4.0", "P_bg2", "P_bg3"),
      **easyblock("layer4.1", "P_bg3"),
      **easyblock("layer4.2", "P_bg3"),
      
      **dense("linear", "P_bg3", None),
  })

def resnet50_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
  norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
  
  # Bottleneck block without shortcut
  bottleneck = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}.norm1", f"P_{name}_inner1"),
      **conv(f"{name}.conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}.norm2", f"P_{name}_inner2"),
      **conv(f"{name}.conv3", f"P_{name}_inner2", p_out),
      **norm(f"{name}.norm3", p_out),
  }
  
  # Bottleneck block with shortcut
  shortcut_bottleneck = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}.norm1", f"P_{name}_inner1"),
      **conv(f"{name}.conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}.norm2", f"P_{name}_inner2"),
      **conv(f"{name}.conv3", f"P_{name}_inner2", p_out),
      **norm(f"{name}.norm3", p_out),
      **conv(f"{name}.shortcut.0", p_in, p_out),
      **norm(f"{name}.shortcut.1", p_out),
  }
  
  return permutation_spec_from_axes_to_perm({
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      
      # Layer 1 (3 blocks)
      **shortcut_bottleneck("layer1.0", "P_bg0", "P_bg0_x4"),
      **bottleneck("layer1.1", "P_bg0_x4", "P_bg0_x4"),
      **bottleneck("layer1.2", "P_bg0_x4", "P_bg0_x4"),
      
      # Layer 2 (4 blocks)
      **shortcut_bottleneck("layer2.0", "P_bg0_x4", "P_bg1_x4"),
      **bottleneck("layer2.1", "P_bg1_x4", "P_bg1_x4"),
      **bottleneck("layer2.2", "P_bg1_x4", "P_bg1_x4"),
      **bottleneck("layer2.3", "P_bg1_x4", "P_bg1_x4"),
      
      # Layer 3 (6 blocks)
      **shortcut_bottleneck("layer3.0", "P_bg1_x4", "P_bg2_x4"),
      **bottleneck("layer3.1", "P_bg2_x4", "P_bg2_x4"),
      **bottleneck("layer3.2", "P_bg2_x4", "P_bg2_x4"),
      **bottleneck("layer3.3", "P_bg2_x4", "P_bg2_x4"),
      **bottleneck("layer3.4", "P_bg2_x4", "P_bg2_x4"),
      **bottleneck("layer3.5", "P_bg2_x4", "P_bg2_x4"),
      
      # Layer 4 (3 blocks)
      **shortcut_bottleneck("layer4.0", "P_bg2_x4", "P_bg3_x4"),
      **bottleneck("layer4.1", "P_bg3_x4", "P_bg3_x4"),
      **bottleneck("layer4.2", "P_bg3_x4", "P_bg3_x4"),
      
      **dense("linear", "P_bg3_x4", None),
  })

def resnet101_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
  norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
  
  bottleneck = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}.norm1", f"P_{name}_inner1"),
      **conv(f"{name}.conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}.norm2", f"P_{name}_inner2"),
      **conv(f"{name}.conv3", f"P_{name}_inner2", p_out),
      **norm(f"{name}.norm3", p_out),
  }
  
  shortcut_bottleneck = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}.norm1", f"P_{name}_inner1"),
      **conv(f"{name}.conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}.norm2", f"P_{name}_inner2"),
      **conv(f"{name}.conv3", f"P_{name}_inner2", p_out),
      **norm(f"{name}.norm3", p_out),
      **conv(f"{name}.shortcut.0", p_in, p_out),
      **norm(f"{name}.shortcut.1", p_out),
  }
  
  spec_dict = {
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      
      # Layer 1 (3 blocks)
      **shortcut_bottleneck("layer1.0", "P_bg0", "P_bg0_x4"),
      **bottleneck("layer1.1", "P_bg0_x4", "P_bg0_x4"),
      **bottleneck("layer1.2", "P_bg0_x4", "P_bg0_x4"),
      
      # Layer 2 (4 blocks)
      **shortcut_bottleneck("layer2.0", "P_bg0_x4", "P_bg1_x4"),
      **bottleneck("layer2.1", "P_bg1_x4", "P_bg1_x4"),
      **bottleneck("layer2.2", "P_bg1_x4", "P_bg1_x4"),
      **bottleneck("layer2.3", "P_bg1_x4", "P_bg1_x4"),
  }
  
  # Layer 3 (23 blocks)
  spec_dict.update(shortcut_bottleneck("layer3.0", "P_bg1_x4", "P_bg2_x4"))
  for i in range(1, 23):
      spec_dict.update(bottleneck(f"layer3.{i}", "P_bg2_x4", "P_bg2_x4"))
  
  # Layer 4 (3 blocks)
  spec_dict.update(shortcut_bottleneck("layer4.0", "P_bg2_x4", "P_bg3_x4"))
  spec_dict.update(bottleneck("layer4.1", "P_bg3_x4", "P_bg3_x4"))
  spec_dict.update(bottleneck("layer4.2", "P_bg3_x4", "P_bg3_x4"))
  
  spec_dict.update(dense("linear", "P_bg3_x4", None))
  
  return permutation_spec_from_axes_to_perm(spec_dict)

def resnet152_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
  norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
  
  bottleneck = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}.norm1", f"P_{name}_inner1"),
      **conv(f"{name}.conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}.norm2", f"P_{name}_inner2"),
      **conv(f"{name}.conv3", f"P_{name}_inner2", p_out),
      **norm(f"{name}.norm3", p_out),
  }
  
  shortcut_bottleneck = lambda name, p_in, p_out: {
      **conv(f"{name}.conv1", p_in, f"P_{name}_inner1"),
      **norm(f"{name}.norm1", f"P_{name}_inner1"),
      **conv(f"{name}.conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
      **norm(f"{name}.norm2", f"P_{name}_inner2"),
      **conv(f"{name}.conv3", f"P_{name}_inner2", p_out),
      **norm(f"{name}.norm3", p_out),
      **conv(f"{name}.shortcut.0", p_in, p_out),
      **norm(f"{name}.shortcut.1", p_out),
  }
  
  spec_dict = {
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      
      # Layer 1 (3 blocks)
      **shortcut_bottleneck("layer1.0", "P_bg0", "P_bg0_x4"),
      **bottleneck("layer1.1", "P_bg0_x4", "P_bg0_x4"),
      **bottleneck("layer1.2", "P_bg0_x4", "P_bg0_x4"),
  }
  
  # Layer 2 (8 blocks)
  spec_dict.update(shortcut_bottleneck("layer2.0", "P_bg0_x4", "P_bg1_x4"))
  for i in range(1, 8):
      spec_dict.update(bottleneck(f"layer2.{i}", "P_bg1_x4", "P_bg1_x4"))
  
  # Layer 3 (36 blocks)
  spec_dict.update(shortcut_bottleneck("layer3.0", "P_bg1_x4", "P_bg2_x4"))
  for i in range(1, 36):
      spec_dict.update(bottleneck(f"layer3.{i}", "P_bg2_x4", "P_bg2_x4"))
  
  # Layer 4 (3 blocks)
  spec_dict.update(shortcut_bottleneck("layer4.0", "P_bg2_x4", "P_bg3_x4"))
  spec_dict.update(bottleneck("layer4.1", "P_bg3_x4", "P_bg3_x4"))
  spec_dict.update(bottleneck("layer4.2", "P_bg3_x4", "P_bg3_x4"))
  
  spec_dict.update(dense("linear", "P_bg3_x4", None))
  
  return permutation_spec_from_axes_to_perm(spec_dict)

def reverse_permutation_list(permutation):
  inverse_perm = [0] * len(permutation)
  for i, p in enumerate(permutation):
      inverse_perm[p] = i
  inverse_perm = jnp.array(inverse_perm)
  return inverse_perm

def get_reverse_permuted_param(ps: PermutationSpec, perm, k: str, params):
  w = params[k]
  w = jnp.array(w)
  for axis, p in enumerate(ps.axes_to_perm[k]):
    if p is not None:
      # w = reverse_permutation(w, perm[p], axis)
      w = jnp.take(w, reverse_permutation_list(perm[p]), axis=axis)
  return w

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