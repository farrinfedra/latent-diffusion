import numpy as np
from six.moves import cPickle as pickle
import os
import platform

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000,data_dir="/datasets/cifar/cifar-10-batches-py"):
    
    # Load the raw CIFAR-10 data
    cifar10_dir = data_dir
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    x_val = X_val.astype('float32')

    x_train = (x_train / 127.5 - 1.0)
    x_test = (x_test / 127.5 - 1.0)
    x_val = (x_val / 127.5 - 1.0)
    
    print(x_train.max())
    print(x_train.min())
    
#     x_train /= 255
#     x_test /= 255
#     x_val /= 255
    
#     x_train = x_train*2 - 1
#     x_test = x_test*2 - 1
#     x_val = x_val*2 - 1
    
    return x_train, y_train, x_val, y_val, x_test, y_test


class CifarBase(Dataset):
    def __init__(self, data, transform=transforms.Compose([transforms.ToTensor()])):
        
        assert data.shape[1] == 3072, 'shape of the cifar dataset is wrong'
        
#         print(data.shape)
        self.data = (data.reshape(data.shape[0],3,32,32)).transpose(0,2,3,1)

    
#         print(self.data.shape)
#         self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index,:]
        
#         print(f"max value of before {x.max()}")
#         print(f"min value of before {x.min()}")

#         y = self.targets[index]
#         print(f"x shape is {x.shape}") 
    
#         truck_img = torch.from_numpy(x).unsqueeze(0)
    
#         if self.transform:
#             x = Image.fromarray(self.data[index,:].astype(np.uint8)) # .transpose(1,2,0)
# #             print(f"x shape just before transform is {x.size}")
#             x = self.transform(x)
#             print(x)
#         print(f"x shape after transform is {x.shape}")
        example = {}
        example["image"] = torch.from_numpy(x) # x.view(32,32,3) * 2 - 1
        
#         print(f"max value of after {x.max()}")
#         print(f"min value of after {x.min()}")

        
        return example
    
    def __len__(self):
        return len(self.data)
    
class CifarTrain(CifarBase):
    def __init__(self, **kwargs):
        x_train, y_train, _, _, _, _ = get_CIFAR10_data()
        super().__init__(data = x_train)

class CifarVal(CifarBase):
    def __init__(self, **kwargs):
        _, _, x_val, _, _, _ = get_CIFAR10_data()
        super().__init__(data = x_val)
        