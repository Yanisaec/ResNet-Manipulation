import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from collections import defaultdict
import datasets

def fetch_dataset(data_name, subset):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name == 'CIFAR10' or data_name == 'cifar10':
        dataset['train'] = datasets.CIFAR10(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataset['test'] = datasets.CIFAR10(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif data_name == 'CIFAR100' or data_name == 'cifar100':
        dataset['train'] = datasets.CIFAR100(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataset['test'] = datasets.CIFAR100(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    print('data ready')
    return dataset

def split_dataset(dataset, num_users, data_split_mode, data_name, cfg):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], label_split = iid(dataset['train'], num_users, data_name)
        data_split['test'], _ = iid(dataset['test'], num_users, data_name)
    elif data_split_mode == 'dirichlet':
        data_split['train'], label_split = dirichlet_split(dataset['train'], num_users, cfg['shared']['alpha'])
        data_split['test'], _ = dirichlet_split(dataset['test'], num_users, cfg['shared']['alpha'])
    else:
        raise ValueError('Not valid data split mode')
    return data_split, label_split

def iid(dataset, num_users, data_name):
    if data_name in ['MNIST', 'CIFAR10', 'CIFAR100', 'cifar10', 'cifar100']:
        label = torch.tensor(dataset.target)
    else:
        raise ValueError('Not valid data name')
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    label_split = {}
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))

    return data_split, label_split

def dirichlet_split(dataset, num_users, alpha=0.3):
    # Extract labels and get number of classes
    label = torch.tensor(dataset.target)  
    num_classes = len(torch.unique(label))
    data_split = {}
    label_split = {}

    # Group dataset indices by class
    label_indices = {c: np.where(label.numpy() == c)[0] for c in range(num_classes)}

    idx_batch = [[] for _ in range(num_users)]

    # Step 1: Assign indices to users based on Dirichlet distribution
    for c, indices in label_indices.items():
        np.random.shuffle(indices)
        
        # Generate Dirichlet-distributed proportions for this class
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_users))
        
        # Adjust proportions to ensure fair distribution
        proportions = np.array([p * (len(idx_j) < len(dataset) / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

        # Split indices based on computed proportions
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(indices, proportions))]
    
    # Step 2: Assign final indices and labels
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])  # Shuffle assigned indices
        data_split[i] = idx_batch[i]
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
    
    # print(np.max([len(data_split[i]) for i in range(num_users)]))
    # print(np.min([len(data_split[i]) for i in range(num_users)]))
    return data_split, label_split

class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]]

class GenericDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input = self.dataset[index]
        return input

class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0]['label'].size(0)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - index)
        return {'label': self.dataset[:]['label'][:, self.idx[index]:self.idx[index] + seq_length]}