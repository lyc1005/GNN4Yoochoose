# utils_gat.py

import torch
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import DataLoader as GeometricDataLoader
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def build_graph(train_data):
    G = nx.DiGraph()
    for seq in train_data[0]:
        for i in range(len(seq) - 1):
            G.add_edge(seq[i], seq[i + 1])
    return G

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples)
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class SessionData():
    def __init__(self, data, shuffle=False):
        self.inputs = data[0]
        self.targets = data[1]
        self.length = len(self.inputs)
        self.shuffle = shuffle
        self.dataset = self._build_dataset()

    def _build_dataset(self):
        dataset = []
        for seq, target in zip(self.inputs, self.targets):
            # Build graph for each session
            G = nx.DiGraph()
            G.add_nodes_from(range(len(seq)))
            edges = [(i, i + 1) for i in range(len(seq) - 1)]
            G.add_edges_from(edges)
            data = from_networkx(G)
            data.x = torch.tensor(seq, dtype=torch.long).unsqueeze(-1)
            data.y = torch.tensor(target - 1, dtype=torch.long)
            data.length = len(seq)
            # Do not set data.batch here
            dataset.append(data)
        return dataset

    def generate_loader(self, batch_size):
        if self.shuffle:
            np.random.shuffle(self.dataset)
        loader = GeometricDataLoader(self.dataset, batch_size=batch_size, shuffle=self.shuffle)
        return loader

