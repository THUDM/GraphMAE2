import os

import numpy as np
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

from .data_proc import preprocess, scale_feats
from utils import mask_edge

import logging
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# def collect_topk_ppr(graph, nodes, topk, alpha, epsilon):
#     if torch.is_tensor(nodes):
#         nodes = nodes.numpy()
#     row, col = graph.edges()
#     row = row.numpy()
#     col = col.numpy()
#     num_nodes = graph.num_nodes()

#     neighbors = build_topk_ppr((row, col), alpha, epsilon, nodes, topk, num_nodes=num_nodes)
#     return neighbors

# ---------------------------------------------------------------------------------------------------------------------


def load_dataset(data_dir, dataset_name):
    if dataset_name.startswith("ogbn"):
        dataset = DglNodePropPredDataset(dataset_name, root=os.path.join(data_dir, "dataset"))
        graph, label = dataset[0]

        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            logging.info("--- to undirected graph ---")
            graph = preprocess(graph)
        graph = graph.remove_self_loop().add_self_loop()

        split_idx = dataset.get_idx_split()
        label = label.view(-1)

        feats = graph.ndata.pop("feat") 
        if dataset_name in ("ogbn-arxiv","ogbn-papers100M"):
            feats = scale_feats(feats)
    elif dataset_name == "mag-scholar-f":
        edge_index = np.load(os.path.join(data_dir, dataset_name, "edge_index_f.npy"))
        feats = torch.from_numpy(np.load(os.path.join(data_dir, "feature_f.npy"))).float()

        graph = dgl.DGLGraph((edge_index[0], edge_index[1]))

        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        label = torch.from_numpy(np.load(os.path.join(data_dir, "label_f.npy"))).to(torch.long)
        split_idx = torch.load(os.path.join(data_dir, "split_idx_f.pt"))

        # graph.ndata["feat"] = feats
        # graph.ndata["label"] = label  

    return feats, graph, label, split_idx

class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label
    
class OnlineLCLoader(DataLoader):
    def __init__(self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))]
        sg = dgl.batch(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        sg = sg.remove_self_loop().add_self_loop()
        sg.ndata["feat"] = self.feats[nodes]
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels != None:
            label = self.labels[batch_idx]
        else:
            label = None
        
        if self._drop_edge_rate > 0:
            return sg, targets, label, nodes, drop_g1, drop_g2
        else:
            return sg, targets, label, nodes


def setup_training_data(dataset_name, data_dir, ego_graphs_file_path):
    feats, graph, labels, split_idx = load_dataset(data_dir, dataset_name)

    train_lbls = labels[split_idx["train"]]
    val_lbls = labels[split_idx["valid"]]
    test_lbls = labels[split_idx["test"]]

    labels = torch.cat([train_lbls, val_lbls, test_lbls])
    
    os.makedirs(os.path.dirname(ego_graphs_file_path), exist_ok=True)

    if not os.path.exists(ego_graphs_file_path):
        raise FileNotFoundError(f"{ego_graphs_file_path} doesn't exist")
    else:
        nodes = torch.load(ego_graphs_file_path)

    return feats, graph, labels, split_idx, nodes


def setup_training_dataloder(loader_type, training_nodes, graph, feats, batch_size, drop_edge_rate=0, pretrain_clustergcn=False, cluster_iter_data=None):
    num_workers = 8

    if loader_type == "lc":
        assert training_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")
    
    # print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = OnlineLCLoader(training_nodes, graph, feats=feats, drop_edge_rate=drop_edge_rate, batch_size=batch_size, shuffle=True, drop_last=False, persistent_workers=True, num_workers=num_workers)
    return dataloader


def setup_eval_dataloder(loader_type, graph, feats, ego_graph_nodes=None, batch_size=128, shuffle=False):
    num_workers = 8
    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, batch_size=batch_size, shuffle=shuffle, drop_last=False, persistent_workers=True, num_workers=num_workers)
    return dataloader


def setup_finetune_dataloder(loader_type, graph, feats, ego_graph_nodes, labels, batch_size, shuffle=False):
    num_workers = 8

    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")
    
    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, labels=labels, feats=feats, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, persistent_workers=True)
    
    return dataloader
