import copy 
import torch as th
import torch_pruning as tp

def prune_model(model, pruning_ratio=0.5, out_features=10, input_shape=(1, 3, 32, 32)):
    global_model = copy.deepcopy(model)
    global_model = global_model.cpu().eval()
    example_inputs = th.randn(input_shape)
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