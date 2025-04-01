from torch.utils.data import DataLoader
import random

from resnet import *
from datasets.dataset import *
from git_rebasin.resnet18_permutation import *
from torch_pruning import *
from net2net import *
from utils import *

_batchsize=8
_seed=42
_train=False
_test_resnet18=False

torch.manual_seed(_seed)
torch.cuda.manual_seed(_seed)
np.random.seed(_seed)
torch.cuda.manual_seed_all(_seed)
torch.set_deterministic_debug_mode('default')
torch.backends.cudnn.enabled = False
os.environ['PYTHONHASHSEED'] = str(_seed)

def main():
    dataset = fetch_dataset('cifar10', 'label')
    dataloader_train = DataLoader(dataset['train'], _batchsize)
    dataloader_test = DataLoader(dataset['test'], _batchsize)

    if _test_resnet18:
        resnet18_1 = ResNet18(hidden_sizes=[32, 64, 128, 256], n_class=10)

        if _train:
            train_model(dataloader_train, resnet18_1, 1, 'resnet18.pt')

        resnet18_1.load_state_dict(torch.load('resnet18.pt', weights_only=True))

        resnet18_2 = ResNet18(hidden_sizes=[32, 64, 128, 256], n_class=10)   

        # test_widen(resnet18_1)
        
        test_permutation(resnet18_1, resnet18_2, dataloader_test, '18', 100)
    resnet18_1 = ResNet18(hidden_sizes=[32, 64, 128, 256], n_class=10)
    breakpoint()
    resnet34_1 = ResNet34(hidden_sizes=[32, 64, 128, 256], n_class=10)
    resnet34_2 = ResNet34(hidden_sizes=[32, 64, 128, 256], n_class=10)  

    resnet50_1 = ResNet50(hidden_sizes=[32, 64, 128, 256], n_class=10)
    resnet50_2 = ResNet50(hidden_sizes=[32, 64, 128, 256], n_class=10)
    
    resnet101_1 = ResNet101(hidden_sizes=[32, 64, 128, 256], n_class=10)
    resnet101_2 = ResNet101(hidden_sizes=[32, 64, 128, 256], n_class=10)
        
    resnet152_1 = ResNet152(hidden_sizes=[32, 64, 128, 256], n_class=10)
    resnet152_2 = ResNet152(hidden_sizes=[32, 64, 128, 256], n_class=10)
    

if __name__ == "__main__":
    main()