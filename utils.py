import dgl
import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import scipy.sparse as sp
from dgl.data.utils import load_graphs

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def t_v_t_split(train_ratio, val_ratio, num_nodes):
    node_idx = np.arange(num_nodes)
    train_num = int(train_ratio * num_nodes)
    val_num = int(val_ratio * num_nodes)

    selected_idx = np.random.choice(node_idx, train_num+val_num, replace=False)

    train_mask = torch.zeros(num_nodes).bool()
    val_mask = torch.zeros(num_nodes).bool()

    train_mask[selected_idx[:train_num]] = True
    val_mask[selected_idx[train_num:]] = True
    test_mask = torch.logical_and(~train_mask, ~val_mask)

    print(torch.sum(train_mask))
    print(torch.sum(val_mask))
    print(torch.sum(test_mask))
    return train_mask, val_mask, test_mask

def split_graph(graph, train_mask, val_mask, test_mask):
    train_graph = graph.subgraph(train_mask)
    val_graph = graph.subgraph(val_mask)
    test_graph = graph.subgraph(test_mask)
    return train_graph, val_graph, test_graph

def idx_sample(idxes):
    num_idx = len(idxes)
    random_add = torch.randint(low=1, high=num_idx, size=(1,), device='cpu')
    idx = torch.arange(0, num_idx)

    shuffled_idx = torch.remainder(idx+random_add, num_idx)

    return shuffled_idx

def row_normalization(feats):
    return F.normalize(feats, p=2, dim=1)

def my_load_data(dataname, path='/home/sherry/code/GADACE/data/'):
    data_dir = path + dataname + '.bin'
    graph = load_graphs(data_dir)

    return graph[0][0]

def my_load_data_bond(dataname):
    file_path='/home/sherry/code/GADACE/data/' + dataname + '.pt/' + dataname + '.pt'
    data = torch.load(file_path)
    
    # PyG to DGL
    u, v = data.edge_index
    graph = dgl.graph((u, v))
    graph.ndata['feat'] = data.x
    graph.ndata['label'] = data.y
    graph.ndata['label'] = data.y.bool()
    
    return graph

def pyg_to_dgl(pyg_graph):
    # Extract the PyG graph components
    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr
    num_nodes = pyg_graph.num_nodes
    node_attr = pyg_graph.x
    labels = pyg_graph.y
    # Create a DGL graph
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    dgl_graph.ndata['feat'] = node_attr
    dgl_graph.ndata['label'] = labels
    # Set edge attributes if they exist
    if edge_attr is not None:
        dgl_graph.edata['edge_attr'] = torch.tensor(edge_attr)

    return dgl_graph

def aug_random_edge(input_adj, drop_percent=0.2):
    return 1


def normalize_array(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return arr - mean

    normalized_arr = (arr - mean) / std
    
    return normalized_arr


def edge_permutation(graph):
    feats = graph.ndata['feat']
    labels = graph.ndata['label'] 
    src, dst = graph.edges()
    num_edges = len(src)

    num_edges_to_remove = int(0.05 * num_edges)

    indices_to_remove = np.random.choice(num_edges, num_edges_to_remove, replace=False)

    src_filtered = np.delete(src, indices_to_remove)
    dst_filtered = np.delete(dst, indices_to_remove)

    existing_edges = set(zip(src_filtered.tolist(), dst_filtered.tolist()))

    edges_to_add = []
    while len(edges_to_add) < num_edges_to_remove:
        u = np.random.randint(0, graph.number_of_nodes())
        v = np.random.randint(0, graph.number_of_nodes())
        if u != v and (u, v) not in existing_edges:
            edges_to_add.append((u, v))
            existing_edges.add((u, v))
            existing_edges.add((v, u)) 

    new_src, new_dst = zip(*edges_to_add)
    new_src = np.concatenate((src_filtered, np.array(new_src)))
    new_dst = np.concatenate((dst_filtered, np.array(new_dst)))

    new_graph = dgl.graph((new_src, new_dst), num_nodes=graph.number_of_nodes())
    new_graph.ndata['feat'] = feats
    new_graph.ndata['label'] = labels

    print("New graph with modified edges created.")
    
    return new_graph


def heat_diffusion(graph, feats, t, num_iterations):
    with graph.local_scope():
        num_nodes = graph.number_of_nodes()
        
        src, dst = graph.edges()
        adj_matrix = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))
        
        degs = adj_matrix.sum(axis=1).A1
        degs_inv_sqrt = np.power(degs, -0.5)
        degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0.
        degs_inv_sqrt_mat = sp.diags(degs_inv_sqrt)
        normalized_laplacian = sp.eye(num_nodes) - degs_inv_sqrt_mat @ adj_matrix @ degs_inv_sqrt_mat
        # normalized_laplacian = degs_inv_sqrt_mat @ adj_matrix @ degs_inv_sqrt_mat
        
        normalized_laplacian = torch.FloatTensor(normalized_laplacian.todense()).to(feats.device)
        
        h = feats.clone()
        
        for _ in range(num_iterations):
            h = (1 - t) * h + t * torch.mm(normalized_laplacian, h)
        
        return h

    
    
    
    
    

