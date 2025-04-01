import torch as th
import torch.nn as nn
import copy 
import torch

from metric import Metric, to_device

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

    models = [model.cuda() for model in models]
    
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

def train_model(dataloader, model, epochs, save_path=None, lr=0.05):
    model.cuda()
    model.train()
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_iter = 0

    for epoch in range(epochs):
        tot_loss = 0
        for i, step_input in enumerate(dataloader):
            total_iter += 1

            inputs, targets = step_input['img'], step_input['label']
            inputs, targets = inputs.to('cuda'), targets.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            tot_loss += loss.data
            loss.backward(retain_graph=True)
            optimizer.step()

        print(f'Epoch: {epoch} - Loss: {tot_loss/total_iter}.')
    
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        
    return model

def evaluate_model(dataloader, model, nb_batch, print_=False):
    assert nb_batch > 0

    tot_loss1 = 0
    tot_acc1 = 0

    model.cuda()
    model.eval()

    for i, batch in enumerate(dataloader):
        if i == nb_batch:
            break

        inputs, targets = batch['img'], batch['label']
                
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        output1 = model(inputs)

        dict1 = {'loss': th.nn.CrossEntropyLoss()(output1, targets), 'score': output1}

        eval1 = Metric().evaluate(['Local-Loss', 'Local-Accuracy'], to_device(batch, 'cuda'), dict1)

        tot_loss1 += eval1['Local-Loss']
        tot_acc1 += eval1['Local-Accuracy']

    if print_:
        print(f'Model performances over {nb_batch} batches. Loss: {tot_loss1/nb_batch}. Accuracy: {tot_acc1/nb_batch}.')

    return tot_loss1, tot_acc1
    
def compare_model_performances(dataloader, model1, model2, nb_batch, print_=False):
    _, acc1 = evaluate_model(dataloader, model1, nb_batch, print_=print_)
    _, acc2 = evaluate_model(dataloader, model2, nb_batch, print_=print_)

    return acc1, acc2