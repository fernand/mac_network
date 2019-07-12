import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class CLEVRDataset(Dataset):
    def __init__(self, root, split='train'):
        with open(f'{root}{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.root = root
        self.split = split
        self.h = h5py.File(root+'{}_features.hdf5'.format(split), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])
        return img, question, len(question), answer, family

    def __len__(self):
        return len(self.data)

def collate_data(batch):
    images, lengths, answers, families = [], [], [], []
    batch_size = len(batch)
    max_len = max(map(lambda x: len(x[1]), batch))
    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        families.append(family)
    return torch.stack(images), torch.from_numpy(questions), lengths, torch.LongTensor(answers), families
