import sys
import os

import h5py
import torch
from torchvision.models.resnet import ResNet, resnet101
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device('cuda')

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return x

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.224])])

class CLEVRImg(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.length = len(os.listdir(os.path.join(root,
                                                'images', split)))

    def __getitem__(self, index):
        img = os.path.join(self.root, 'images',
                        self.split,
                        'CLEVR_{}_{}.png'.format(self.split,
                                            str(index).zfill(6)))
        img = Image.open(img).convert('RGB')
        return transform(img)

    def __len__(self):
        return self.length

def create_dataset(split, batch_size, resnet):
    dataloader = DataLoader(CLEVRImg(sys.argv[1], split), batch_size=batch_size,
                            num_workers=6)

    size = len(dataloader)
    print(split, 'total', size * batch_size)
    f = h5py.File('data/{}_features.hdf5'.format(split), 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, 1024, 14, 14),
                            dtype='f4')
    with torch.no_grad():
        for i, image in tqdm(enumerate(dataloader)):
            image = image.to(DEVICE)
            features = resnet(image).detach().cpu().numpy()
            dset[i * batch_size:(i + 1) * batch_size] = features
    f.close()

if __name__ == '__main__':
    resnet = resnet101(True).to(DEVICE)
    resnet.eval()
    resnet.forward = forward.__get__(resnet, ResNet)
    create_dataset('val', 50, resnet)
    create_dataset('train', 50, resnet)
