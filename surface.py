import os
import torch
import torch.nn.functional as F
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import argparse
import pretrainedmodels
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='')
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--adv_dir', type=str, default='')
parser.add_argument('--output_path', type=str, default='./surfaces-TAI/')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]))
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]))
parser.add_argument("--batch_size", type=int, default=1)
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299),
     T.ToTensor(),
     T.Lambda(lambda x: x.type(torch.FloatTensor))]  # 确保张量类型为FloatTensor
)



class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        return (x - self.mean) / self.std



class ImageNet(Dataset):
    def __init__(self, input_dir, input_csv, transforms):
        self.input_dir = input_dir
        self.transforms = transforms
        self.data_info = pd.read_csv(input_csv, sep=',')  # 读取新格式CSV

    def __getitem__(self, idx):
        filename = self.data_info.iloc[idx]['filename']
        label = self.data_info.iloc[idx]['label']
        img_path = os.path.join(self.input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, filename, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data_info)


def img2torch(image_path):
    pil_img = Image.open(image_path).convert('RGB')
    image = transforms(pil_img)
    image = torch.unsqueeze(image, 0)
    return image


def get_loss_value(image, model, gt):
    model.eval()
    with torch.no_grad():
        if image.dtype != torch.float32:
            image = image.type(torch.float32)
        output = model(image)
        loss_value = F.cross_entropy(output, gt)
    return loss_value.item()


def plot_3d_images(img, model, gt, output_path, image_id, adv_dir):
    os.makedirs(output_path, exist_ok=True)

    image_path = os.path.join(adv_dir, image_id)
    if not os.path.exists(image_path):
        print(f"earn: no {image_path} skip......")
        return


    adv_image = img2torch(image_path).cuda()


    eta = torch.rand_like(img, dtype=torch.float32).cuda()
    delta = torch.rand_like(img, dtype=torch.float32).cuda()


    a = np.arange(-1, 1, 0.05)
    b = np.arange(-1, 1, 0.05)
    map_3d = np.zeros(shape=(a.shape[0], b.shape[0]))
    size = a.shape[0]


    for i in range(size):
        for j in range(size):

            new_image = adv_image + (255.0 / 255) * (a[i] * eta + b[j] * delta)
            new_image = new_image.type(torch.float32)
            new_image = torch.clamp(new_image, 0, 1)
            map_3d[i][j] = get_loss_value(new_image, model, gt)

    X, Y = np.meshgrid(a, b, indexing='ij')
    fig = plt.figure(figsize=(20, 20), facecolor='white')

    sub = fig.add_subplot(111, projection='3d')
    surf = sub.plot_surface(X, Y, map_3d, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    sub.zaxis.set_major_locator(plt.LinearLocator(10))
    sub.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
    sub.set_title(r'TAI', fontsize=40)
    plt.savefig(os.path.join(output_path, image_id) + '.jpg', dpi=300)
    plt.close(fig)


def main():
    model = torch.nn.Sequential(
        Normalize(opt.mean, opt.std),
        pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda()
    ).float()


    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True)


    for images, images_ID, gt_cpu in tqdm(data_loader):
        images = images.cuda().float()
        gt = gt_cpu.cuda()
        plot_3d_images(images, model, gt, output_path=opt.output_path, image_id=images_ID[0], adv_dir=opt.adv_dir)


if __name__ == '__main__':
    main()