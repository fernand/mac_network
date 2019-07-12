import os
import shutil
import pickle
from collections import Counter

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVRDataset, collate_data
from mac import MACNetwork

PATH = '/home/fernand/clevr/mac_data/'
DEVICE = torch.device('cuda')
torch.manual_seed(12345)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1:
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def train(epoch, net, net_average, opt, crit, batch_size):
    clevr = CLEVRDataset(PATH, 'train')
    train_set = DataLoader(clevr, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=collate_data)
    pbar = tqdm(iter(train_set))
    moving_acc = 0
    net.train(True)
    for image, question, q_len, answer, _ in pbar:
        image, question, answer = (
            image.to(DEVICE),
            question.to(DEVICE),
            answer.to(DEVICE),
        )
        output = net(image, question, q_len)
        loss = crit(output, answer)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 8)
        opt.step()
        opt.zero_grad()
        correct = output.clone().detach().argmax(1) == answer
        correct = float(correct.sum()) / batch_size
        if moving_acc == 0:
            moving_acc = correct
        else:
            moving_acc = moving_acc * 0.99 + correct * 0.01
        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch + 1, loss.item(), moving_acc
            )
        )
        accumulate(net_average, net)
    clevr.close()

def valid(epoch, net_average, batch_size):
    clevr = CLEVRDataset(PATH, 'val')
    valid_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=6, collate_fn=collate_data
    )

    net_average.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer, family in tqdm(iter(valid_set)):
            image, question = image.to(DEVICE), question.to(DEVICE)

            output = net_average(image, question, q_len)
            correct = output.clone().detach().argmax(1) == answer.to(DEVICE)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        for k, v in family_total.items():
            w.write('{}: {:.5f}\n'.format(k, family_correct[k] / v))

    print(
        'Avg Acc: {:.5f}'.format(
            sum(family_correct.values()) / sum(family_total.values())
        )
    )
    clevr.close()


if __name__ == '__main__':
    shutil.rmtree('log')
    os.mkdir('log')
    shutil.rmtree('checkpoint')
    os.mkdir('checkpoint')
    with open(PATH+'dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words).to(DEVICE)
    net_average = MACNetwork(n_words).to(DEVICE)
    accumulate(net_average, net, 0)
    opt = optim.Adam(net.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    batch_size = 64
    for epoch in range(20):
        train(epoch, net, net_average, opt, crit, batch_size)
        valid(epoch, net, batch_size)
        with open(
            'checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb'
        ) as f:
            torch.save(net.state_dict(), f)
