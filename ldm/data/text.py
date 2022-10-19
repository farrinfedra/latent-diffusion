import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
# import glob
# import time
# import cv2
# import json


class Base(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
            
        self._length = len(self.image_paths)
        
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32) 
        if random.randint(0, 10) > 5:
            example["caption"] = "dummy caption"
        else:
            example["caption"] = "some other dummy caption in different len"
        return example


    def __len__(self):
        return self.size
    
    
class Txttrain(Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="/kuacc/users/bbiner21/Github/lsun/out", **kwargs)
        
class Txtval(Base):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="/kuacc/users/bbiner21/Github/lsun/out",
                         flip_p=flip_p, **kwargs)        
        
