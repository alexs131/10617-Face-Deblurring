import torch
from torchvision.transforms import transforms

from models.LFWC import LFWC

dataset =  LFWC(["lfwcrop_color/faces_blurred0.5","lfwcrop_color/faces_blurred1.0","lfwcrop_color/faces_blurred1.5",
"lfwcrop_color/faces_blurred2.0","lfwcrop_color/faces_blurred2.5","lfwcrop_color/faces_blurred3.0",
"lfwcrop_color/faces_pixelated",'lfwcrop_color/faces_linear_motion0','lfwcrop_color/faces_linear_motion90'], "lfwcrop_color/faces")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

count = 0

for im in train_data_loader:
    transform = transforms.ToPILImage()

    transform(im["blurred"][0]).save("train/faces_blurred/" + str(count) + ".ppm")
    transform(im["nonblurred"][0]).save("train/faces/" + str(count) + ".ppm")
    count += 1

for im in test_data_loader:
    transform = transforms.ToPILImage()

    transform(im["blurred"][0]).save("data/test/faces_blurred/" + str(count) + ".ppm")
    transform(im["nonblurred"][0]).save("data/test/faces/" + str(count) + ".ppm")
    count += 1
