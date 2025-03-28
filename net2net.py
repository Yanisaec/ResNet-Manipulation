import torch as th
import numpy as np
from collections import Counter
import torch.nn as nn
import copy 
import torch_pruning as tp

def wider(m1, m2, new_width, bnorm=None, out_size=None, noise=False,
          random_init=False, weight_norm=False, no_running=False, affine=True, random_mapping=None, divide=True):
    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data if m1.bias is not None else None

    if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1)//factor**2, factor, factor)
            elif w1.dim() == 5:
                assert out_size is not None,\
                       "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(w2.size(0), w2.size(1)//factor, out_size[0],
                             out_size[1], out_size[2])
        else:
            assert w1.size(0) == w2.size(1), "Module weights are not compatible"
        assert new_width > w1.size(0), "New size should be larger"

        old_width = w1.size(0)
        nw1 = m1.weight.data.clone()
        nw2 = w2.clone()

        if nw1.dim() == 4:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))
        elif nw1.dim() == 5:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3), nw1.size(4))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3), nw2.size(4))
        else:
            nw1.resize_(new_width, nw1.size(1))
            nw2.resize_(nw2.size(0), new_width)

        if b1 is not None:
            nb1 = m1.bias.data.clone()
            nb1.resize_(new_width)

        if bnorm is not None:
            if not no_running:
                if bnorm.running_mean is not None:
                    nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
                    nrunning_var = bnorm.running_var.clone().resize_(new_width)
            if affine:
                nweight = bnorm.weight.data.clone().resize_(new_width)
                nbias = bnorm.bias.data.clone().resize_(new_width)

        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)

        nw1.narrow(0, 0, old_width).copy_(w1)
        nw2.narrow(0, 0, old_width).copy_(w2)
        if b1 is not None:
            nb1.narrow(0, 0, old_width).copy_(b1)

        if bnorm is not None:
            if not no_running:
                if bnorm.running_mean is not None:
                    nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
                    nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
            if affine:
                nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

        # TEST:normalize weights
        if weight_norm:
            for i in range(old_width):
                norm = w1.select(0, i).norm()
                w1.select(0, i).div_(norm)

        # select weights randomly
        tracking = dict()
        for i in range(old_width, new_width):
            if random_mapping is None:
                idx = np.random.randint(0, old_width)
                try:
                    tracking[idx].append(i)
                except:
                    tracking[idx] = [idx]
                    tracking[idx].append(i)
            else:
                 for val, d in random_mapping.items():
                    for item in d:
                        if item == i:  
                            idx = val
            # TEST:random init for new units
            if random_init:
                n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                if m2.weight.dim() == 4:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                elif m2.weight.dim() == 5:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                elif m2.weight.dim() == 2:
                    n2 = m2.out_features * m2.in_features
                nw1.select(0, i).normal_(0, np.sqrt(2./n))
                nw2.select(0, i).normal_(0, np.sqrt(2./n2))
            else:
                nw1.select(0, i).copy_(w1.select(0, idx).clone())
                nw2.select(0, i).copy_(w2.select(0, idx).clone())
            if b1 is not None:
                nb1[i] = b1[idx]

            if bnorm is not None:
                if not no_running:
                    if bnorm.running_mean is not None:
                        nrunning_mean[i] = bnorm.running_mean[idx]
                        nrunning_var[i] = bnorm.running_var[idx]
                if affine:
                    nweight[i] = bnorm.weight.data[idx]
                    nbias[i] = bnorm.bias.data[idx]
                bnorm.num_features = new_width

        if divide:
            if not random_init:
                if random_mapping is None:
                    for idx, d in tracking.items():
                        for item in d:
                            nw2[item].div_(len(d))
                else:
                    for idx, d in random_mapping.items():
                        for item in d:
                            nw2[item].div_(len(d))

        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m1.out_channels = new_width
        m2.in_channels = new_width

        if noise:
            noise = np.random.normal(scale=5e-2 * nw1.std(),
                                     size=list(nw1.size()))
            nw1 += th.FloatTensor(noise).type_as(nw1)

        m1.weight.data = nw1

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            if w1.dim() == 4:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor**2)
                m2.in_features = new_width*factor**2
            elif w2.dim() == 5:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor)
                m2.in_features = new_width*factor
        else:
            m2.weight.data = nw2

        if b1 is not None:
            m1.bias.data = nb1

        if bnorm is not None:
            if not no_running:
                if bnorm.running_mean is not None:
                    bnorm.running_var = nrunning_var
                    bnorm.running_mean = nrunning_mean
            if affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias

        return m1, m2, bnorm, tracking

def widen_resnet18(model, new_hidden_sizes=[64, 128, 256, 512], noise=False, random_init=False, weight_norm=False, forced_mapping=None, divide=True):
    """
    Widen a ResNet model to have new hidden sizes.
    
    Args:
        model: The ResNet model to widen
        new_hidden_sizes: List of integers representing the new widths for each layer
        noise: Whether to add noise to new weights
        random_init: Whether to initialize new weights randomly
        weight_norm: Whether to normalize weights
        
    Returns:
        Widened model
    """
    if len(new_hidden_sizes) != 4:
        raise ValueError("new_hidden_sizes must have 4 elements for ResNet architecture")
    
    # Create a deep copy of the model to avoid modifying the original
    model = copy.deepcopy(model)
    
    # Store all modules that need to be updated
    modules = {}
    
    # First convolutional layer
    modules['conv1'] = model.conv1
    modules['norm1'] = model.norm1
    
    # Extract all layer modules
    for layer_idx in range(1, 5):  # Layers 1-4
        layer = getattr(model, f'layer{layer_idx}')
        for block_idx in range(len(layer)):  # Usually 2 blocks per layer in ResNet18
            block = layer[block_idx]
            
            # Main path modules
            modules[f'l{layer_idx}_b{block_idx}_conv1'] = copy.deepcopy(block.conv1)
            modules[f'l{layer_idx}_b{block_idx}_norm1'] = copy.deepcopy(block.norm1)
            modules[f'l{layer_idx}_b{block_idx}_conv2'] = copy.deepcopy(block.conv2)
            modules[f'l{layer_idx}_b{block_idx}_norm2'] = copy.deepcopy(block.norm2)
            
            # Shortcut modules (if exists)
            if len(block.shortcut) > 0:
                modules[f'l{layer_idx-1}_b1_conv2_for_sc'] = copy.deepcopy(getattr(model, f'layer{layer_idx-1}')[block_idx+1].conv2)
                modules[f'l{layer_idx-1}_b1_norm2_for_sc'] = copy.deepcopy(getattr(model, f'layer{layer_idx-1}')[block_idx+1].norm2)
                modules[f'l{layer_idx}_b{block_idx}_shortcut_conv'] = copy.deepcopy(block.shortcut[0])
                modules[f'l{layer_idx}_b{block_idx}_shortcut_norm'] = copy.deepcopy(block.shortcut[1])
                modules[f'l{layer_idx}_b1_conv1_for_sc'] = copy.deepcopy(layer[block_idx+1].conv1)

    # Final linear layer
    modules['linear'] = model.linear
    
    # Define the connections between modules for widening
    connections = [
        # Input layer to first block
        ('conv1', 'l1_b0_conv1', 'norm1', new_hidden_sizes[0]),
        
        # Layer 1
        ('l1_b0_conv1', 'l1_b0_conv2', 'l1_b0_norm1', new_hidden_sizes[0]),
        ('l1_b0_conv2', 'l1_b1_conv1', 'l1_b0_norm2', new_hidden_sizes[0]),
        ('l1_b1_conv1', 'l1_b1_conv2', 'l1_b1_norm1', new_hidden_sizes[0]),
        ('l1_b1_conv2', 'l2_b0_conv1', 'l1_b1_norm2', new_hidden_sizes[0]),
        
        # Layer 2
        ('l2_b0_conv1', 'l2_b0_conv2', 'l2_b0_norm1', new_hidden_sizes[1]),
        ('l2_b0_conv2', 'l2_b1_conv1', 'l2_b0_norm2', new_hidden_sizes[1]),
        # Layer 2 shortcut
        ('l1_b1_conv2_for_sc', 'l2_b0_shortcut_conv', 'l1_b1_norm2_for_sc', new_hidden_sizes[0]),
        ('l2_b0_shortcut_conv', 'l2_b1_conv1_for_sc', 'l2_b0_shortcut_norm', new_hidden_sizes[1]),
        
        ('l2_b1_conv1', 'l2_b1_conv2', 'l2_b1_norm1', new_hidden_sizes[1]),
        ('l2_b1_conv2', 'l3_b0_conv1', 'l2_b1_norm2', new_hidden_sizes[1]),
        
        # Layer 3
        ('l3_b0_conv1', 'l3_b0_conv2', 'l3_b0_norm1', new_hidden_sizes[2]),
        ('l3_b0_conv2', 'l3_b1_conv1', 'l3_b0_norm2', new_hidden_sizes[2]),
        # Layer 3 shortcut
        ('l2_b1_conv2_for_sc', 'l3_b0_shortcut_conv', 'l2_b1_norm2_for_sc', new_hidden_sizes[1]),
        ('l3_b0_shortcut_conv', 'l3_b1_conv1_for_sc', 'l3_b0_shortcut_norm', new_hidden_sizes[2]),
        
        ('l3_b1_conv1', 'l3_b1_conv2', 'l3_b1_norm1', new_hidden_sizes[2]),
        ('l3_b1_conv2', 'l4_b0_conv1', 'l3_b1_norm2', new_hidden_sizes[2]),
        
        # Layer 4
        ('l4_b0_conv1', 'l4_b0_conv2', 'l4_b0_norm1', new_hidden_sizes[3]),
        ('l4_b0_conv2', 'l4_b1_conv1', 'l4_b0_norm2', new_hidden_sizes[3]),
        # Layer 4 shortcut
        ('l3_b1_conv2_for_sc', 'l4_b0_shortcut_conv', 'l3_b1_norm2_for_sc', new_hidden_sizes[2]),
        ('l4_b0_shortcut_conv', 'l4_b1_conv1_for_sc', 'l4_b0_shortcut_norm', new_hidden_sizes[3]),
        
        ('l4_b1_conv1', 'l4_b1_conv2', 'l4_b1_norm1', new_hidden_sizes[3]),
        ('l4_b1_conv2', 'linear', 'l4_b1_norm2', new_hidden_sizes[3]),
    ]
    
    # Apply wider operation to each connection
    mapping = {}
    for m1_name, m2_name, bn_name, new_width in connections:
        m1 = modules[m1_name]
        m2 = modules[m2_name]
        bn = modules[bn_name] if bn_name else None
        
        # Apply wider operation
        if m1_name == 'conv1' and forced_mapping is not None:
            rm = forced_mapping['conv1']
        elif m1_name != 'conv1' and 'l1' in m1_name:
            if forced_mapping is not None:
                rm = forced_mapping['conv1']
            else:
                rm = mapping['conv1']
        elif m1_name != 'l2_b0_conv1' and 'l2' in m1_name:
            if forced_mapping is not None:
                rm = forced_mapping['l2_b0_conv1']
            else:
                rm = mapping['l2_b0_conv1']
        elif m1_name != 'l3_b0_conv1' and 'l3' in m1_name:
            if forced_mapping is not None:
                rm = forced_mapping['l3_b0_conv1']
            else:
                rm = mapping['l3_b0_conv1']
        elif m1_name != 'l4_b0_conv1' and 'l4' in m1_name:
            if forced_mapping is not None:
                rm = forced_mapping['l4_b0_conv1']
            else:
                rm = mapping['l4_b0_conv1']
        else:
            rm = None
            
        m1, m2, bn, random_mapping = wider(m1=m1, m2=m2, new_width=new_width, bnorm=bn, noise=noise, random_init=random_init, weight_norm=weight_norm, random_mapping=rm, divide=divide)

        mapping[m1_name] = random_mapping

        # Update modules dict with the widened modules
        modules[m1_name] = m1
        modules[m2_name] = m2
        if bn_name:
            modules[bn_name] = bn
    
    # Update model with widened modules
    model.conv1 = modules['conv1']
    model.norm1 = modules['norm1']
    
    # Update model's in_planes attribute to reflect the new sizes
    model.in_planes = new_hidden_sizes[3] * model.layer4[0].expansion
    
    # Update all layers
    for layer_idx in range(1, 5):  # Layers 1-4
        layer = getattr(model, f'layer{layer_idx}')
        for block_idx in range(len(layer)):  # Usually 2 blocks per layer
            block = layer[block_idx]
            
            # Update main path modules
            block.conv1 = modules[f'l{layer_idx}_b{block_idx}_conv1']
            block.norm1 = modules[f'l{layer_idx}_b{block_idx}_norm1']
            block.conv2 = modules[f'l{layer_idx}_b{block_idx}_conv2']
            block.norm2 = modules[f'l{layer_idx}_b{block_idx}_norm2']
            
            # Update shortcut modules (if exists)
            if len(block.shortcut) > 0:
                block.shortcut[0] = modules[f'l{layer_idx}_b{block_idx}_shortcut_conv']
                block.shortcut[1] = modules[f'l{layer_idx}_b{block_idx}_shortcut_norm']
    
    # Update linear layer
    model.linear = modules['linear']
    
    return model, mapping

def unwiden_resnet18(model_original_dict, model_widened_dict):
    unwidened_model_dict = {}
    for key in model_original_dict:
        w_original = model_original_dict[key]
        w = model_widened_dict[key]
        
        if len(w.shape) == 1: # norm/bias layer
            new_w = w[:w_original.shape[0]]
        elif len(w.shape) == 2: # linear layer
            new_w = w[:,:w_original.shape[1]]
        elif len(w.shape) == 4: # conv layer
            new_w = w[:w_original.shape[0],:w_original.shape[1]]
        
        unwidened_model_dict[key] = new_w
    
    return unwidened_model_dict

def get_layer_by_name(model, layer_name):
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    return None  # Return None if layer is not found
    
def compare_model_architecture(model1: nn.Module, model2: nn.Module) -> bool:
    model1_layers = list(model1.named_modules())
    model2_layers = list(model2.named_modules())

    if len(model1_layers) != len(model2_layers):
        print("Mismatch in the number of layers")
        return False

    for i, (a, b) in enumerate(zip(model1_layers, model2_layers)):
        (name1, layer1) = a
        (name2, layer2) = b
        if name1 != name2:
            print(f"Layer name mismatch: {name1} != {name2}")
            return False
        if type(layer1) is not type(layer2) and i > 0:
            print(f"Layer type mismatch at {name1}: {type(layer1)} != {type(layer2)}")
            return False
        if isinstance(layer1, nn.Conv2d):
            if (layer1.in_channels != layer2.in_channels or 
                layer1.out_channels != layer2.out_channels or
                layer1.kernel_size != layer2.kernel_size or
                layer1.stride != layer2.stride or
                layer1.padding != layer2.padding):
                print(f"Conv2d mismatch at {name1}")
                return False
        if isinstance(layer1, nn.Linear):
            if layer1.in_features != layer2.in_features or layer1.out_features != layer2.out_features:
                print(f"Linear layer mismatch at {name1}")
                return False
        if isinstance(layer1, nn.BatchNorm2d):
            if layer1.num_features != layer2.num_features:
                print(f"BatchNorm2d mismatch at {name1}")
                return False 

    return True  

def average_models(models):
    """
    Given a list of N PyTorch models with the same architecture,
    returns a new model with the averaged weights.
    """
    assert len(models) > 0, "Model list is empty"

    # Deep copy the first model to use as the base
    averaged_model = copy.deepcopy(models[0])
    averaged_state_dict = averaged_model.state_dict()

    # Initialize an empty dictionary to accumulate weights
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = th.zeros_like(averaged_state_dict[key])

    # Accumulate weights from all models
    for model in models:
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += model.state_dict()[key]

    # Compute the mean
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] /= len(models)

    # Load the averaged weights into the new model
    averaged_model.load_state_dict(averaged_state_dict)

    return averaged_model

def prune_model(model, pruning_ratio=0.5, out_features=10):
    global_model = copy.deepcopy(model)
    global_model = global_model.cpu().eval()
    example_inputs = th.randn(1, 3, 32, 32)
    imp = tp.importance.GroupNormImportance(p=2) 
    ignored_layers = []
    for m in global_model.modules():
        if isinstance(m, th.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        global_model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=8, 
    )
    pruner.step()

    return global_model