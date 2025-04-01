from torch.utils.data import DataLoader

from resnet import *
from datasets.dataset import *
from git_rebasin.resnet18_permutation import *
from torch_pruning import *
from net2net import *
from utils import *

_batchsize=8
_seed=42
_train=False
_test_resnet18=True
_test_resnet34=True
_test_resnet50=True
_test_resnet101=False
_test_resnet152=False

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
            train_model(dataloader_train, resnet18_1, 1, 'checkpoints/resnet18.pt')

        resnet18_1.load_state_dict(torch.load('checkpoints/resnet18.pt', weights_only=True))

        resnet18_2 = ResNet18(hidden_sizes=[32, 64, 128, 256], n_class=10)   

        test_widen(resnet18_1, model_type='18')
        
        # test_permutation(resnet18_1, resnet18_2, dataloader_test, '18', 1000)

    if _test_resnet34:
        resnet34_1 = ResNet34(hidden_sizes=[32, 64, 128, 256], n_class=10)

        if _train:
            train_model(dataloader_train, resnet34_1, 1, 'checkpoints/resnet34.pt')

        resnet34_1.load_state_dict(torch.load('checkpoints/resnet34.pt', weights_only=True))

        resnet34_2 = ResNet34(hidden_sizes=[32, 64, 128, 256], n_class=10)  

        test_widen(resnet34_1, model_type='34')

        # test_permutation(resnet34_1, resnet34_2, dataloader_test, '34', 1000) 
        
    if _test_resnet50:
        resnet50_1 = ResNet50(hidden_sizes=[32, 64, 128, 256], n_class=10)

        if _train:
            train_model(dataloader_train, resnet50_1, 1, 'checkpoints/resnet50.pt')

        resnet50_1.load_state_dict(torch.load('checkpoints/resnet50.pt', weights_only=True))

        resnet50_2 = ResNet50(hidden_sizes=[32, 64, 128, 256], n_class=10)

        test_widen(resnet50_1, model_type='50')

        # test_permutation(resnet50_1, resnet50_2, dataloader_test, '50', 1000)      
    
    if _test_resnet101:
        resnet101_1 = ResNet101(hidden_sizes=[32, 64, 128, 256], n_class=10)

        if _train:
            train_model(dataloader_train, resnet101_1, 1, 'checkpoints/resnet101.pt')

        resnet101_1.load_state_dict(torch.load('checkpoints/resnet101.pt', weights_only=True))

        resnet101_2 = ResNet101(hidden_sizes=[32, 64, 128, 256], n_class=10)

        test_permutation(resnet101_1, resnet101_2, dataloader_test, '101', 1000)     

    if _test_resnet152:
        resnet152_1 = ResNet152(hidden_sizes=[32, 64, 128, 256], n_class=10)

        if _train:
            train_model(dataloader_train, resnet152_1, 1, 'checkpoints/resnet152.pt')

        resnet152_1.load_state_dict(torch.load('checkpoints/resnet152.pt', weights_only=True))

        resnet152_2 = ResNet152(hidden_sizes=[32, 64, 128, 256], n_class=10)

        test_permutation(resnet152_1, resnet152_2, dataloader_test, '152', 1000) 

        
if __name__ == "__main__":
    main()