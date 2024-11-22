# model_gat.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch_geometric.nn import GATConv, global_add_pool
import numpy as np

class GAT(Module):
    def __init__(self, opt, n_node):
        super(GAT, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.num_classes = n_node
        self.dropout = 0.6

        self.embedding = nn.Embedding(self.num_classes + 1, self.hidden_size, padding_idx=0)
        self.conv1 = GATConv(self.hidden_size, self.hidden_size, heads=8, dropout=self.dropout)
        self.conv2 = GATConv(self.hidden_size * 8, self.hidden_size, heads=1, concat=False, dropout=self.dropout)
        self.predictor = nn.Linear(self.hidden_size, self.num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x.squeeze())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        scores = self.predictor(x)
        return scores
    
class GATwithLSTM(Module):
    def __init__(self, opt, n_node):
        super(GATwithLSTM, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.num_classes = n_node
        self.dropout = 0.6
        self.steps = opt.step

        self.embedding = nn.Embedding(self.num_classes + 1, self.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.steps, batch_first=True)
        self.conv1 = GATConv(self.hidden_size, self.hidden_size, heads=8, dropout=self.dropout)
        self.conv2 = GATConv(self.hidden_size * 8, self.hidden_size, heads=1, concat=False, dropout=self.dropout)
        self.predictor = nn.Linear(self.hidden_size, self.num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x.squeeze())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Now process x with LSTM
        # Get lengths of sequences
        lengths = torch.bincount(batch)
        max_length = lengths.max().item()
        batch_size = lengths.size(0)

        # Create padded sequences
        sequences = torch.zeros(batch_size, max_length, self.hidden_size, device=x.device)
        idx = 0
        for i, length in enumerate(lengths.tolist()):
            sequences[i, :length] = x[idx:idx+length]
            idx += length

        # Pack sequences
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_sequences)

        # Use the last hidden state as the representation
        x = h_n[-1]  # Shape: [batch_size, hidden_size]

        scores = self.predictor(x)
        return scores


class GATwithGRU(Module):
    def __init__(self, opt, n_node):
        super(GATwithGRU, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.num_classes = n_node
        self.dropout = 0.6
        self.steps = opt.step

        self.embedding = nn.Embedding(self.num_classes + 1, self.hidden_size, padding_idx=0)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.steps, batch_first=True)
        self.conv1 = GATConv(self.hidden_size, self.hidden_size, heads=8, dropout=self.dropout)
        self.conv2 = GATConv(self.hidden_size * 8, self.hidden_size, heads=1, concat=False, dropout=self.dropout)
        self.predictor = nn.Linear(self.hidden_size, self.num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x.squeeze())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Now process x with GRU
        # Get lengths of sequences
        lengths = torch.bincount(batch)
        max_length = lengths.max().item()
        batch_size = lengths.size(0)

        # Create padded sequences
        sequences = torch.zeros(batch_size, max_length, self.hidden_size, device=x.device)
        idx = 0
        for i, length in enumerate(lengths.tolist()):
            sequences[i, :length] = x[idx:idx+length]
            idx += length

        # Pack sequences
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through GRU
        packed_output, h_n = self.gru(packed_sequences)

        # Use the last hidden state as the representation
        x = h_n[-1]  # Shape: [batch_size, hidden_size]

        scores = self.predictor(x)
        return scores

  

def forward(model, data):
    data = trans_to_cuda(data)
    scores = model(data)
    targets = data.y
    return targets, scores

def train_test(model, train_loader, test_loader):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        model.optimizer.zero_grad()
        targets, scores = forward(model, batch)
        targets = trans_to_cuda(targets)
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    model.scheduler.step()
    print('Train Loss: {:.4f}'.format(total_loss / len(train_loader)))

    model.eval()
    hit, mrr = [], []
    with torch.no_grad():
        for batch in test_loader:
            targets, scores = forward(model, batch)
            sub_scores = scores.topk(20)[1]
            sub_scores = sub_scores.cpu().detach().numpy()
            targets = targets.cpu().numpy()
            for score, target in zip(sub_scores, targets):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    print('Recall@20: {:.2f}%, MRR@20: {:.2f}%'.format(hit, mrr))
    return hit, mrr

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable