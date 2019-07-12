import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

def linear(in_dim, out_dim):
    lin = nn.Linear(in_dim, out_dim)
    xavier_uniform_(lin.weight)
    lin.bias.data.zero_()
    return lin

class ControlUnit(nn.Module):
    def __init__(self, d, n_steps):
        super().__init__()
        self.control_question = linear(2*d, d)
        self.attention = linear(d, 1)

    def forward(self, step_i, context, question, control):
        control_question = self.control_question(torch.cat([control, question], 1))
        control_question = control_question.unsqueeze(1)
        attention = self.attention(control_question * context)
        attention = F.softmax(attention, 1)
        return (attention * context).sum(1)

class ReadUnit(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.memory = linear(d, d)
        self.interaction = linear(2*d, d)
        self.attention = linear(d, 1)

    def forward(self, memory, knowledge, control):
        knowledge = knowledge.permute(0, 2, 1)
        interaction = (self.memory(memory).unsqueeze(1) * knowledge)
        interaction = self.interaction(torch.cat([interaction, knowledge], 2))
        attention = self.attention(control.unsqueeze(1) * interaction).squeeze(2)
        attention = F.softmax(attention, 1).unsqueeze(2)
        return (attention * knowledge).sum(1)

class WriteUnit(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m_info = linear(2*d, d)

    def forward(self, information, memory):
        return self.m_info(torch.cat([information, memory], 1))

class MACUnit(nn.Module):
    def __init__(self, d, n_steps):
        super().__init__()
        self.position_aware = nn.ModuleList()
        for i in range(n_steps):
            self.position_aware.append(linear(2*d, d))
        self.cu = ControlUnit(d, n_steps)
        self.ru = ReadUnit(d)
        self.wu = WriteUnit(d)
        self.d = d
        self.n_steps = n_steps

    def forward(self, context, question, knowledge):
        b_size = question.size(0)
        q_pa = self.position_aware[0](question)
        control = torch.randn(1, 1).cuda().expand(b_size, self.d)
        memory = q_pa.clone()
        for i in range(self.n_steps):
            if i > 0:
                q_pa = self.position_aware[i](question)
            control = self.cu(i, context, q_pa, control)
            read = self.ru(memory, knowledge, control)
            memory = self.wu(read, memory)
        return memory

class MACNetwork(nn.Module):
    def __init__(self, n_vocab, d=512, n_embed=300, n_steps=12, n_classes=28):
        super().__init__()
        # Kaiming initialization by default.
        self.conv = nn.Sequential(
            nn.Conv2d(1024, d, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.ELU()
        )
        self.embed = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, d, batch_first=True, bidirectional=True)
        self.lstm_proj = linear(2*d, d)
        self.mac = MACUnit(d, n_steps)
        self.output = nn.Sequential(
            linear(3*d, d),
            nn.ELU(),
            linear(d, n_classes)
        )
        self.d = d

    def forward(self, image, question, question_len):
        b_size = question.size(0)
        img = self.conv(image)
        img = img.view(b_size, self.d, -1)
        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)
        memory = self.mac(lstm_out, h, img)
        return self.output(torch.cat([h, memory], 1))

# For debugging tensor sizes.
if __name__ == '__main__':
    batch_size = 64
    image = torch.randn(batch_size, 1024, 14, 14)
    question = torch.randint(0, 2000, (batch_size, 30), dtype=torch.int64)
    answer = torch.randint(0, 28, (batch_size,), dtype=torch.int64)
    q_len = torch.tensor([30] * batch_size, dtype=torch.int64)
    net = MACNetwork(2000)
    net(image, question, q_len)
