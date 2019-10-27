import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class LFWC(Dataset):
    def __init__(self, blurred_dirs, non_blurred_dir, transform=None):
        self.transform = transform

        self.blurred_dirs = blurred_dirs
        self.non_blurred_dir = non_blurred_dir

        self.non_blurred = [os.path.join(non_blurred_dir, s) for s in list(os.listdir(non_blurred_dir))]


        self.items = []
        for l in blurred_dirs:
            l = [os.path.join(l, s) for s in list(os.listdir(l))]
            zipped = zip(self.non_blurred, l)
            self.items.extend(zipped)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()



        tensor = transforms.ToTensor()

        return {'nonblurred': tensor(Image.open(self.items[item][0])), 'blurred': tensor(Image.open(self.items[item][1]))}
