import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class LFWC(Dataset):
    def __init__(self, blurred_dir, non_blurred_dir, transform=None):
        self.transform = transform
        self.blurred_dir = blurred_dir
        self.non_blurred_dir = non_blurred_dir
        self.non_blurred = list(os.listdir(non_blurred_dir))

        self.blurred = list(os.listdir(blurred_dir))

    def __len__(self):
        return len(self.non_blurred)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()


        im_name = self.non_blurred[item]

        non_blurred = Image.open(os.path.join(self.non_blurred_dir, im_name))
        blurred = Image.open(os.path.join(self.blurred_dir, im_name))

        tensor = transforms.ToTensor()


        return {'nonblurred': tensor(non_blurred), 'blurred': tensor(blurred)}
