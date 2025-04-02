# ResNet Model Handling Library

## Overview
This library provides tools to manipulate ResNet models of various depths (18, 34, 50, 101, 152). It is built upon the principles of model transformation from [Net2Net](https://github.com/erogol/Net2Net) and [GitReBasin](https://github.com/samuela/git-re-basin), enabling seamless adaptation of these techniques for ResNet architectures.
It also contains handling of the Cifar10 and Cifar100 datasets in order to train and evaluate models and to assess the impact of ResNet architecture modifications on their performances.

### Features
- **Widening**: Expand ResNet models without any modification in the outputs of the models.
- **Unwidening**: Reduce model width while maintaining key properties (reverse operation of the widening process to get the original model back).
- **Permutation Alignment**: Improve accuracy when averaging model weights by aligning permutations.

## Installation
You only need an up to date version of pytorch to run the code.
# ResNet Model Handling Library

## Usage
### Importing the Library
```python
from resnet import *
from net2net import widen_resnet, unwiden_resnet
from git_rebasin.resnet_permutation import permute_resnet

model = ResNet50(hidden_sizes=[32, 64, 128, 256], n_class=10)
```

### Widening a ResNet Model
```python
widened_model = widen_resnet(model, new_hidden_sizes=[64, 128, 256, 512], model_type=None, noise=False, random_init=False, weight_norm=False, forced_mapping=None, divide=True)
```

### Unwidening a ResNet Model
```python
original_model = unwiden_resnet(model_original_dict, model_widened_dict)
```

### Permuting Model Weights for Averaging
```python
aligned_model = permute_resnet(ref_model, model_to_permute, resnet_model='50')
```

## Dependencies
- PyTorch
- NumPy

## Citation
If you use this library, please cite:
- Net2Net: [Paper Link](https://arxiv.org/abs/1511.05641)
- GitReBasin: [Paper Link](https://arxiv.org/abs/2209.04836)

## License
MIT License

## Acknowledgments
This project is inspired by the works of Net2Net and GitReBasin, adapted to provide flexible ResNet handling functionalities.


