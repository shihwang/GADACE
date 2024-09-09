import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from utils import idx_sample

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            activation,
        ]) 
    
    def forward(self, features):
        h = features
        for layer in self.encoder:
            h = layer(h)
            
        # row normalize
        h = F.normalize(h, p=2, dim=1)  
        return h
    
class GCN(nn.Module):
    def __init__(
        self, g, in_dim, hid_dim, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.gcn = GraphConv(in_dim, hid_dim, activation=activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = self.gcn(self.g, features)
        return self.dropout(h)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, features, centers):
        return torch.sum(features * centers, dim=1)

class Encoder_local(nn.Module):
    def __init__(self, args, graph, in_dim,  out_dim, activation):
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)
        self.meanAgg = MeanAggregator(args)
        self.g = graph
        
    def forward(self, h):
        h = self.encoder(h)
        mean_h = self.meanAgg(self.g ,h)

        return h, mean_h
    
class MeanAggregator(nn.Module):
    def __init__(self, args):
        super(MeanAggregator, self).__init__()
        self.arg = args

    def forward(self, graph, h):
        with graph.local_scope():
            # one hop neighbor
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), (fn.mean('m', 'neigh1')))
            neigh1 = graph.ndata['neigh1']
            
            # two hop neighbor
            graph.ndata['h1'] = neigh1
            graph.update_all(fn.copy_u('h1', 'm2'), fn.mean('m2', 'neigh2'))
            neigh2 = graph.ndata['neigh2']
            
            l = self.arg.l
            result = l * neigh1 + (1 - l) * neigh2
            
            return result

class Encoder_global(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation):
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)
        self.g = graph
        
    def forward(self, h):
        h = self.encoder(h)
        return h

class LocalModel(nn.Module):
    def __init__(self, args, graph, graph_hat, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = Encoder_local(args, graph, in_dim, out_dim, activation)
        self.encoder_hat = Encoder_local(args, graph_hat, in_dim, out_dim, activation)
        self.arg = args
        self.g = graph
        self.g_hat = graph_hat
        self.discriminator = Discriminator()
        self.loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
    
    def forward(self, h, h_hat):
        h, mean_h = self.encoder(h)
        h_hat, mean_h_hat = self.encoder_hat(h_hat)
        
        # positive
        pos = self.discriminator(h, mean_h)
        
        # negtive
        idx = torch.arange(0, h.shape[0])
        neg_idx = idx_sample(idx)
        neg_neigh_h = mean_h[neg_idx]
        neg = self.discriminator(h, neg_neigh_h)
        
        neg_node_h = h[neg_idx]
        neg_node = self.discriminator(h, neg_node_h)
        
        # positive_hat
        pos_hat = self.discriminator(h_hat, mean_h_hat)
        
        # negtive_hat
        idx_hat = neg_idx
        neg_neigh_h_hat = mean_h_hat[idx_hat]
        neg_hat = self.discriminator(h_hat, neg_neigh_h_hat)
        
        neg_node_h_hat = h_hat[idx_hat]
        neg_node_hat = self.discriminator(h_hat, neg_node_h_hat)
        
        self.g.ndata['pos'] = pos
        self.g.ndata['neg'] = neg
        
        self.g_hat.ndata['pos'] = pos_hat
        self.g_hat.ndata['neg'] = neg_hat

        l1 = self.loss(pos, torch.ones_like(pos))
        l2 = self.loss(neg, torch.zeros_like(neg))
        
        l1_hat = self.loss(pos_hat, torch.ones_like(pos))
        l2_hat = self.loss(neg_hat, torch.zeros_like(neg))

        l1_node = self.loss(pos, torch.ones_like(pos))
        l2_node = self.loss(neg_node, torch.zeros_like(neg))
        
        l1_node_hat = self.loss(pos_hat, torch.ones_like(pos))
        l2_node_hat = self.loss(neg_node_hat, torch.zeros_like(neg))

        batch_size = mean_h.size(0)
        
        positive_score = (mean_h * mean_h_hat).sum(dim=1)
        
        negative_score_1 = torch.matmul(mean_h, neg_neigh_h.T)
        negative_score_2 = torch.matmul(mean_h, neg_neigh_h_hat.T)
        
        exp_positive_score = torch.exp(positive_score)
        exp_negative_score_1 = torch.exp(negative_score_1)
        exp_negative_score_2 = torch.exp(negative_score_2)
        
        denominator = exp_negative_score_1.sum(dim=1) + exp_negative_score_2.sum(dim=1)
        
        # Calculate the final loss
        loss_sup = -torch.log(exp_positive_score / denominator)
        
        # Sum up the loss
        loss_sup = loss_sup.sum() / (2 * batch_size)

        gamma = self.arg.gamma
        alpha = self.arg.alpha
        beta = self.arg.beta
        
        l_ns = alpha * (l1 + l2) + (1 - alpha) * (l1_hat + l2_hat)
        l_nn = alpha * (l1_node + l2_node) + (1 - alpha) * (l1_node_hat + l2_node_hat)

        return (beta * l_ns + (1 - beta) * l_nn + gamma * loss_sup)/((gamma + 1)/2), l1, l2

class GlobalAutoencoder(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = Encoder_global(graph, in_dim, out_dim, activation)
        self.g = graph
    
    def forward(self, feats):
        z = self.encoder(feats)
        
        adj_reconstructed = torch.matmul(z, z.t())
        adj_reconstructed = torch.sigmoid(adj_reconstructed)
        
        return adj_reconstructed, z
    
