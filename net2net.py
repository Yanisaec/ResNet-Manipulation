import torch as th
import numpy as np
import copy 

def wider(om1, om2, new_width, bnorm=None, out_size=None, noise=False,
          random_init=False, weight_norm=False, no_running=False, affine=True, random_mapping=None, divide=True):
    m1 = copy.deepcopy(om1)
    m2 = copy.deepcopy(om2)
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

def widen_resnet(model, new_hidden_sizes=[64, 128, 256, 512], model_type=None, noise=False, random_init=False, weight_norm=False, forced_mapping=None, divide=True):
    """
    Widen any ResNet model to have new hidden sizes.
    
    Args:
        model: The ResNet model to widen
        new_hidden_sizes: List of integers representing the new widths for each layer
        model_type: String indicating the ResNet type ('18', '34', '50', '101', '152'). 
                    If None, it will be inferred from the model structure.
        noise: Whether to add noise to new weights
        random_init: Whether to initialize new weights randomly
        weight_norm: Whether to normalize weights
        forced_mapping: Dictionary that contains the mapping of the new channels
        divide: Whether to divide weights when mapping to new channels
        
    Returns:
        Widened model, mapping dictionary
    """
    import copy
    
    if len(new_hidden_sizes) != 4:
        raise ValueError("new_hidden_sizes must have 4 elements for ResNet architecture")
    
    # Create a deep copy of the model to avoid modifying the original
    model = copy.deepcopy(model)
    
    # Determine model type if not provided
    if model_type is None:
        # Check if it's a bottleneck architecture (ResNet50, 101, 152)
        is_bottleneck = hasattr(model.layer1[0], 'conv3')
        # Count the number of blocks to determine the specific model
        num_blocks = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
        
        if is_bottleneck:
            if num_blocks == [3, 4, 6, 3]:
                model_type = '50'
            elif num_blocks == [3, 4, 23, 3]:
                model_type = '101'
            elif num_blocks == [3, 8, 36, 3]:
                model_type = '152'
        else:
            if num_blocks == [2, 2, 2, 2]:
                model_type = '18'
            elif num_blocks == [3, 4, 6, 3]:
                model_type = '34'
    
    # Check if we got a valid model type
    if model_type not in ['18', '34', '50', '101', '152']:
        raise ValueError(f"Unknown or unsupported ResNet model type: {model_type}")
    
    # Determine if we're dealing with a bottleneck architecture
    is_bottleneck = model_type in ['50', '101', '152']
    expansion = 4 if is_bottleneck else 1
    
    # Store all modules that need to be updated
    modules = {}
    
    # First convolutional layer
    modules['conv1'] = copy.deepcopy(model.conv1)
    modules['norm1'] = copy.deepcopy(model.norm1)
    
    # Extract all layer modules
    for layer_idx in range(1, 5):  # Layers 1-4
        layer = getattr(model, f'layer{layer_idx}')
        for block_idx in range(len(layer)):
            if layer_idx > 1:
                prev_block = copy.deepcopy(block)
            block = layer[block_idx]
            if block_idx < len(layer)-1:
                future_block = layer[block_idx+1]
            
            # Main path modules
            modules[f'l{layer_idx}_b{block_idx}_conv1'] = copy.deepcopy(block.conv1)
            modules[f'l{layer_idx}_b{block_idx}_norm1'] = copy.deepcopy(block.norm1)
            modules[f'l{layer_idx}_b{block_idx}_conv2'] = copy.deepcopy(block.conv2)
            modules[f'l{layer_idx}_b{block_idx}_norm2'] = copy.deepcopy(block.norm2)
            
            # Bottleneck architecture has an additional conv3 layer
            if is_bottleneck:
                modules[f'l{layer_idx}_b{block_idx}_conv3'] = copy.deepcopy(block.conv3)
                modules[f'l{layer_idx}_b{block_idx}_norm3'] = copy.deepcopy(block.norm3)
            
            # Shortcut modules (if exists)
            if len(block.shortcut) > 0:
                modules[f'l{layer_idx}_b{block_idx}_shortcut_conv'] = copy.deepcopy(block.shortcut[0])
                modules[f'l{layer_idx}_b{block_idx}_shortcut_norm'] = copy.deepcopy(block.shortcut[1])
                modules[f'l{layer_idx}_b{block_idx+1}_conv1_for_sc'] = copy.deepcopy(future_block.conv1)
                if layer_idx > 1:
                    if not is_bottleneck:
                        modules[f'l{layer_idx-1}_b{prev_block_idx}_conv2_for_sc'] = copy.deepcopy(prev_block.conv2)
                        modules[f'l{layer_idx-1}_b{prev_block_idx}_norm2_for_sc'] = copy.deepcopy(prev_block.norm2)
                    else:
                        modules[f'l{layer_idx-1}_b{prev_block_idx}_conv3_for_sc'] = copy.deepcopy(prev_block.conv3)
                        modules[f'l{layer_idx-1}_b{prev_block_idx}_norm3_for_sc'] = copy.deepcopy(prev_block.norm3)
                else:
                    modules[f'conv1_for_sc'] = copy.deepcopy(model.conv1)
                    modules[f'norm1_for_sc'] = copy.deepcopy(model.norm1)
                    
            prev_block_idx = block_idx
    
    # Final linear layer
    modules['linear'] = copy.deepcopy(model.linear)
    
    # Define the connections between modules for widening
    connections = []
    
    # Input to first block connection
    connections.append(('conv1', f'l1_b0_conv1', 'norm1', new_hidden_sizes[0]))
    
    # Process each layer
    for layer_idx in range(1, 5):
        layer = getattr(model, f'layer{layer_idx}')
        current_width = new_hidden_sizes[layer_idx-1]
        
        for block_idx in range(len(layer)):
            # Determine input and output widths for this block
            # For layer transitions or stride=2 blocks, we need to handle different widths
            prev_layer_idx = layer_idx
            prev_block_idx = block_idx - 1
            
            # Handle cases where we're at the first block of a layer
            if block_idx == 0 and layer_idx > 1:
                prev_layer_idx = layer_idx - 1
                prev_block_idx = len(getattr(model, f'layer{prev_layer_idx}')) - 1
            
            # For bottleneck architecture
            if is_bottleneck:
                # conv1: in_planes -> planes
                modules_in_width = current_width * expansion if block_idx == 0 and layer_idx > 1 else current_width * expansion if block_idx > 0 else new_hidden_sizes[layer_idx-2] * expansion if layer_idx > 1 else new_hidden_sizes[0]
                connections.append((
                    f'l{layer_idx}_b{block_idx}_conv1',
                    f'l{layer_idx}_b{block_idx}_conv2',
                    f'l{layer_idx}_b{block_idx}_norm1',
                    current_width
                ))
                
                # conv2: planes -> planes
                connections.append((
                    f'l{layer_idx}_b{block_idx}_conv2',
                    f'l{layer_idx}_b{block_idx}_conv3',
                    f'l{layer_idx}_b{block_idx}_norm2',
                    current_width
                ))
                
                # conv3: planes -> planes * expansion
                if block_idx < len(layer) - 1:
                    next_module = f'l{layer_idx}_b{block_idx+1}_conv1'
                elif layer_idx < 4:
                    next_module = f'l{layer_idx+1}_b0_conv1'
                else:
                    next_module = 'linear'
                
                connections.append((
                    f'l{layer_idx}_b{block_idx}_conv3',
                    next_module,
                    f'l{layer_idx}_b{block_idx}_norm3',
                    current_width * expansion
                ))
                
                # Shortcut connection if needed
                if len(layer[block_idx].shortcut) > 0:
                    shortcut_in_width = new_hidden_sizes[layer_idx-2] * expansion if layer_idx > 1 else new_hidden_sizes[0]
                    connections.append((
                        f'l{prev_layer_idx}_b{prev_block_idx}_conv3_for_sc' if prev_block_idx >= 0 else 'conv1_for_sc',
                        f'l{layer_idx}_b{block_idx}_shortcut_conv',
                        f'l{prev_layer_idx}_b{prev_block_idx}_norm3_for_sc' if prev_block_idx >= 0 else 'norm1_for_sc',
                        shortcut_in_width
                    ))
                    
                    connections.append((
                        f'l{layer_idx}_b{block_idx}_shortcut_conv',
                        f'l{layer_idx}_b{block_idx+1}_conv1_for_sc' if block_idx < len(layer) - 1 else f'l{layer_idx+1}_b0_conv1' if layer_idx < 4 else 'linear',
                        f'l{layer_idx}_b{block_idx}_shortcut_norm',
                        current_width * expansion
                    ))
            
            # For basic block architecture (ResNet18, 34)
            else:
                # conv1: in_planes -> planes
                modules_in_width = current_width if block_idx == 0 and layer_idx > 1 else current_width if block_idx > 0 else new_hidden_sizes[layer_idx-2] if layer_idx > 1 else new_hidden_sizes[0]
                connections.append((
                    f'l{layer_idx}_b{block_idx}_conv1',
                    f'l{layer_idx}_b{block_idx}_conv2',
                    f'l{layer_idx}_b{block_idx}_norm1',
                    current_width
                ))
                
                # conv2: planes -> planes
                if block_idx < len(layer) - 1:
                    next_module = f'l{layer_idx}_b{block_idx+1}_conv1'
                elif layer_idx < 4:
                    next_module = f'l{layer_idx+1}_b0_conv1'
                else:
                    next_module = 'linear'
                
                connections.append((
                    f'l{layer_idx}_b{block_idx}_conv2',
                    next_module,
                    f'l{layer_idx}_b{block_idx}_norm2',
                    current_width
                ))
                
                # Shortcut connection if needed
                if len(layer[block_idx].shortcut) > 0:
                    shortcut_in_width = new_hidden_sizes[layer_idx-2] if layer_idx > 1 else new_hidden_sizes[0]
                    connections.append((
                        f'l{prev_layer_idx}_b{prev_block_idx}_conv2_for_sc' if prev_block_idx >= 0 else 'conv1',
                        f'l{layer_idx}_b{block_idx}_shortcut_conv',
                        f'l{prev_layer_idx}_b{prev_block_idx}_norm2_for_sc' if prev_block_idx >= 0 else 'norm1',
                        shortcut_in_width
                    ))
                    
                    connections.append((
                        f'l{layer_idx}_b{block_idx}_shortcut_conv',
                        f'l{layer_idx}_b{block_idx+1}_conv1_for_sc' if block_idx < len(layer) - 1 else f'l{layer_idx+1}_b0_conv1' if layer_idx < 4 else 'linear',
                        f'l{layer_idx}_b{block_idx}_shortcut_norm',
                        current_width
                    ))
    
    # Apply wider operation to each connection
    mapping = {}
    for m1_name, m2_name, bn_name, new_width in connections:
        if not m1_name in modules or not m2_name in modules:
            continue
        
        # if 'l2_b0_conv1' in m1_name or 'l2_b0_conv1' in m2_name:
        #     breakpoint()

        m1 = modules[m1_name]
        m2 = modules[m2_name]
        bn = modules[bn_name] if bn_name in modules else None
        
        # Apply wider operation
        if m1_name == 'conv1' and forced_mapping is not None:
            rm = forced_mapping['conv1']
        elif (m1_name != 'conv1' and 'l1' in m1_name) or m1_name == 'conv1_for_sc':
            if not is_bottleneck:
                if forced_mapping is not None:
                    rm = forced_mapping['conv1']
                else:
                    rm = mapping['conv1']
            else:
                if 'conv3' not in m1_name and 'shortcut' not in m1_name:
                    if forced_mapping is not None:
                        rm = forced_mapping['conv1']
                    else:
                        rm = mapping['conv1']
                else:
                    if 'b0' not in m1_name or 'shortcut' in m1_name:
                        if forced_mapping is not None:
                            rm = forced_mapping['l1_b0_conv3']
                        else:
                            rm = mapping['l1_b0_conv3']
                    else:
                        rm = None
        elif m1_name != 'l2_b0_conv1' and 'l2' in m1_name:
            if not is_bottleneck:
                if forced_mapping is not None:
                    rm = forced_mapping['l2_b0_conv1']
                else:
                    rm = mapping['l2_b0_conv1']
            else:
                if 'conv3' not in m1_name and 'shortcut' not in m1_name:
                    if forced_mapping is not None:
                        rm = forced_mapping['l2_b0_conv1']
                    else:
                        rm = mapping['l2_b0_conv1']
                else:
                    if 'b0' not in m1_name or 'shortcut' in m1_name:
                        if forced_mapping is not None:
                            rm = forced_mapping['l2_b0_conv3']
                        else:
                            rm = mapping['l2_b0_conv3']
                    else:
                        rm = None
        elif m1_name != 'l3_b0_conv1' and 'l3' in m1_name:
            if not is_bottleneck:
                if forced_mapping is not None:
                    rm = forced_mapping['l3_b0_conv1']
                else:
                    rm = mapping['l3_b0_conv1']
            else:
                if 'conv3' not in m1_name and 'shortcut' not in m1_name:
                    if forced_mapping is not None:
                        rm = forced_mapping['l3_b0_conv1']
                    else:
                        rm = mapping['l3_b0_conv1']
                else:
                    if 'b0' not in m1_name or 'shortcut' in m1_name:
                        if forced_mapping is not None:
                            rm = forced_mapping['l3_b0_conv3']
                        else:
                            rm = mapping['l3_b0_conv3']
                    else:
                        rm = None
        elif m1_name != 'l4_b0_conv1' and 'l4' in m1_name:
            if not is_bottleneck:
                if forced_mapping is not None:
                    rm = forced_mapping['l4_b0_conv1']
                else:
                    rm = mapping['l4_b0_conv1']
            else:
                if 'conv3' not in m1_name and 'shortcut' not in m1_name:
                    if forced_mapping is not None:
                        rm = forced_mapping['l4_b0_conv1']
                    else:
                        rm = mapping['l4_b0_conv1']
                else:
                    if 'b0' not in m1_name or 'shortcut' in m1_name:
                        if forced_mapping is not None:
                            rm = forced_mapping['l4_b0_conv3']
                        else:
                            rm = mapping['l4_b0_conv3']
                    else:
                        rm = None
                        
        elif forced_mapping is not None:
            rm = forced_mapping[m1_name]
        else:
            rm = None

        m1, m2, bn, random_mapping = wider(om1=m1, om2=m2, new_width=new_width, bnorm=bn, noise=noise, random_init=random_init, weight_norm=weight_norm, random_mapping=rm, divide=divide)

        mapping[m1_name] = random_mapping

        # Update modules dict with the widened modules
        modules[m1_name] = m1
        modules[m2_name] = m2
        if bn_name and bn_name in modules:
            modules[bn_name] = bn

    # Update model with widened modules
    model.conv1 = modules['conv1']
    model.norm1 = modules['norm1']
    
    # Update model's in_planes attribute to reflect the new sizes
    model.in_planes = new_hidden_sizes[3] * expansion
    
    # Update all layers
    for layer_idx in range(1, 5):  # Layers 1-4
        layer = getattr(model, f'layer{layer_idx}')
        for block_idx in range(len(layer)):
            block = layer[block_idx]
            
            # Update main path modules
            block.conv1 = modules[f'l{layer_idx}_b{block_idx}_conv1']
            block.norm1 = modules[f'l{layer_idx}_b{block_idx}_norm1']
            block.conv2 = modules[f'l{layer_idx}_b{block_idx}_conv2']
            block.norm2 = modules[f'l{layer_idx}_b{block_idx}_norm2']
            
            # Update bottleneck-specific modules
            if is_bottleneck:
                block.conv3 = modules[f'l{layer_idx}_b{block_idx}_conv3']
                block.norm3 = modules[f'l{layer_idx}_b{block_idx}_norm3']
            
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

def test_widen(model, new_hidden_sizes=[64, 128, 256, 512], model_type=None, divide=True):
    dummy = th.randn(1, 3, 32, 32)
    output_original = model(dummy)
    model_widened, _ = widen_resnet(model, new_hidden_sizes=new_hidden_sizes, model_type=model_type, divide=divide)
    new_output = model_widened(dummy)
    print(f'Norm difference between before and after widening the model: {th.norm(output_original-new_output)}')