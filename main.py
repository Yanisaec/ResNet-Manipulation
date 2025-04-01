from torch.utils.data import DataLoader

from resnet import *
from datasets.dataset import *
from git_rebasin.resnet_permutation import *
from net2net import *
from utils import *

_batchsize=8
_seed=42
_train=True
_permute=False
_widen=False

torch.manual_seed(_seed)
torch.cuda.manual_seed(_seed)
np.random.seed(_seed)
torch.cuda.manual_seed_all(_seed)
torch.set_deterministic_debug_mode('default')
torch.backends.cudnn.enabled = False
os.environ['PYTHONHASHSEED'] = str(_seed)

def test_resnet(train=False, widen=False, permute=False, model_type=False, dataloader_train=None, dataloader_test=None, hidden_sizes=[32, 64, 128, 256], n_class=10):
    match model_type:
        case '18':
            model1 = ResNet18(hidden_sizes=hidden_sizes, n_class=n_class)
            model2 = ResNet18(hidden_sizes=hidden_sizes, n_class=n_class)
        case '34':
            model1 = ResNet34(hidden_sizes=hidden_sizes, n_class=n_class)
            model2 = ResNet34(hidden_sizes=hidden_sizes, n_class=n_class)
        case '50':
            model1 = ResNet50(hidden_sizes=hidden_sizes, n_class=n_class)
            model2 = ResNet50(hidden_sizes=hidden_sizes, n_class=n_class)
        case '101':
            model1 = ResNet101(hidden_sizes=hidden_sizes, n_class=n_class)
            model2 = ResNet101(hidden_sizes=hidden_sizes, n_class=n_class)
        case '152':
            model1 = ResNet152(hidden_sizes=hidden_sizes, n_class=n_class)
            model2 = ResNet152(hidden_sizes=hidden_sizes, n_class=n_class)
        case _:
            raise Exception(f'The following ResNet architecture is not supported: {model_type}')
        
    cp = f'checkpoints/resnet{model_type}.pt'
    if train:
        train_model(dataloader_train, model1, 2, cp)
    
    model1.load_state_dict(torch.load(cp, weights_only=True))

    if permute:
        test_permutation(model1, model2, dataloader_test, model_type, 1000)
    
    if widen:
        test_widen(model1, model_type=model_type)
        

def main():
    dataset = fetch_dataset('cifar10', 'label')
    dataloader_train = DataLoader(dataset['train'], _batchsize)
    dataloader_test = DataLoader(dataset['test'], _batchsize)

    model_types = ['18', '34', '50', '101', '152']
    for model_type in model_types:
        test_resnet(train=_train, widen=_widen, permute=_permute, model_type=model_type, dataloader_train=dataloader_train, dataloader_test=dataloader_test)

        
if __name__ == "__main__":
    main()