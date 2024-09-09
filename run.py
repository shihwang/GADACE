import torch
import torch.nn as nn
import statistics
import argparse, time
from dgl.data import register_data_args
from model import *
from utils import *
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score

def load_info_from_local(local_net, device):
    if device >= 0:
        torch.cuda.set_device(device)
        local_net = local_net.to(device)

    memo = torch.load('memo.pth')
    local_net.load_state_dict(torch.load('best_local_model.pkl'))

    if device >= 0:
        memo = {k: v.to(device) for k, v in memo.items()}
    return memo


def train_local(net, graph, new_grpah, feats, opt, args, init=True):
    memo = {}
    labels = graph.ndata['label']
    num_nodes = graph.num_nodes()

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        net = net.to(device)
        labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    if init:
        net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    best = 1e9
    dur = []

    for epoch in range(args.local_epochs):
        net.train()
        t0 = time.time()

        opt.zero_grad()
        loss, l1, l2 = net(feats, feats)

        loss.backward()
        opt.step()

        dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(net.state_dict(), 'best_local_model.pkl')
        
        pos = graph.ndata['pos']
        local_score = -pos.detach().cpu().numpy()
        
        labels_cpu = labels.cpu()
        local_auc = roc_auc_score(labels_cpu, local_score)
        
        print("Epoch {} | Time(s) {:.4f} | Local_auc {:.4f} | Loss {:.4f}"
              .format(epoch+1, np.mean(dur), local_auc, loss.item()))

    memo['graph'] = graph
    net.load_state_dict(torch.load('best_local_model.pkl'))
    h, mean_h = net.encoder(feats)
    h, mean_h = h.detach(), mean_h.detach()
    memo['h'] = h
    memo['mean_h'] = mean_h

    torch.save(memo, 'memo.pth')


def compute_reconstruction_loss(graph, model, features):
    
    adj_matrix = graph.adjacency_matrix().to_dense()
    reconstructed_adj, _ = model(features)
    recon_loss = F.mse_loss(reconstructed_adj, adj_matrix)
    
    return recon_loss

def compute_node_reconstruction_errors(graph, model, features):

    adj_matrix = graph.adjacency_matrix().to_dense()
    reconstructed_adj, _ = model(features)
    node_errors = torch.sum(abs(reconstructed_adj - adj_matrix), dim=1)
    
    return node_errors.detach().cpu().numpy()


def train_global(graph, features, args):
    epochs = args.structure_epochs
    in_dim = features.shape[1]
    model = GlobalAutoencoder(graph, in_dim, args.out_dim, nn.PReLU())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.structure_lr)
    device = args.gpu
    
    if device >= 0:
        torch.cuda.set_device(device)
        model = model.to(device)
        features = features.cuda()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon_loss = compute_reconstruction_loss(graph, model, features)
        recon_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {recon_loss.item()}")

    node_errors = compute_node_reconstruction_errors(graph, model, features)
    
    return node_errors


def main(args):
    seed_everything(args.seed)

    if args.data in ["Cora", "Citeseer", "ACM"]:
        graph = my_load_data(args.data)
    else:
        graph = my_load_data_bond(args.data)
    
    feats = graph.ndata['feat']
    labels = graph.ndata['label']
    new_graph = edge_permutation(graph)

    mean = feats.mean()
    std = feats.std()
    standardized_feats = feats
    if args.data not in ["reddit", "disney"]:
        standardized_feats = (feats - mean) / std
    diffused_features = heat_diffusion(graph, standardized_feats, args.lamd, args.t)
    
    if args.gpu >= 0:
        graph = graph.to(args.gpu)
        new_graph = new_graph.to(args.gpu)

    in_feats = feats.shape[1]
    local_net = LocalModel(args,
                           graph,
                           new_graph,
                           in_feats,
                           args.out_dim,
                           nn.PReLU(),)

    local_opt = torch.optim.Adam(local_net.parameters(), 
                                 lr=args.local_lr, 
                                 weight_decay=args.weight_decay)
    
    t1 = time.time()
    train_local(local_net, graph, new_graph, feats, local_opt, args)
    
    # load information from local module
    memo = load_info_from_local(local_net, args.gpu)
    t2 = time.time()
    graph = memo['graph']
    
    node_reconstruction_errors = train_global(graph, standardized_feats, args)
    node_reconstruction_errors_hat = train_global(graph, diffused_features, args)
    print("Global Reconstruction Errors:", node_reconstruction_errors)
    
    pos = - graph.ndata['pos']
    attribute_score = pos.detach().cpu().numpy()
    print("Local Contrast Anomaly Scores:", attribute_score)
    
    pred_labels = np.zeros_like(labels)
    
    p = args.p
    normalized_attribute = normalize_array(attribute_score)
    normalized_structure = normalize_array((1 - p) * node_reconstruction_errors 
                                           + p * node_reconstruction_errors_hat)
    
    q = args.q
    mix_score = (1 - q) * normalized_structure + q * normalized_attribute
    mix_auc = roc_auc_score(labels, mix_score)
        
    sorted_idx = np.argsort(mix_score)
    k = int(sum(labels))
    topk_idx = sorted_idx[-k:]
    pred_labels[topk_idx] = 1

    recall_k = recall_score(np.ones(k), labels[topk_idx])
    ap = average_precision_score(labels, mix_score)

    ans = "seed: " + str(args.seed) + "------" + "mix_auc {:.4f} | recall@k {:.4f} | ap {:.4f}\n".format(mix_auc, recall_k, ap)
    print(ans)

    return mix_auc
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='...')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="books",
                        help="dataset")
    parser.add_argument("--l", type=float, default="0.6",
                        help="weight of one order neigh")
    parser.add_argument("--gamma", type=float, default="0.5",
                        help="weight of sup loss")
    parser.add_argument("--alpha", type=float, default="0.6",
                        help="weight of first view")
    parser.add_argument("--beta", type=float, default="0.7",
                        help="weight of context-contrast")
    parser.add_argument("--lamd", type=float, default="0.7",
                        help="lambda of diffusion")
    parser.add_argument("--t", type=int, default="4",
                        help="iteartion of diffusion")
    parser.add_argument("--p", type=float, default="0.7",
                        help="weight of enhanced")
    parser.add_argument("--q", type=float, default="0.9",
                        help="weight of attribute")
    parser.add_argument("--local-lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--local-epochs", type=int, default=80,
                        help="number of training contrast model epochs")
    parser.add_argument("--structure-lr", type=float, default=1e-5,
                        help="structure autoencoder lr")
    parser.add_argument("--structure-epochs", type=int, default=110,
                        help="number of training structure epochs")
    parser.add_argument("--out-dim", type=int, default=128,
                        help="number of output units")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)
    parser.add_argument("--seed", type=int, default=2072,
                        help="default random seed")
    args = parser.parse_args()
    print(args)
    file_path = "/home/sherry/code/GADACE/" + args.data + "_ans.txt"

    result = []
    for seed in [random.randint(1, 1000000) for _ in range(10)]:
        args.seed = seed
        result.append(main(args))
    
    mean_value = statistics.mean(result)
    std_dev = statistics.stdev(result)
    
    ans = "final auc:{:.4f}±{:.4f}\n".format(mean_value, std_dev)
    print(ans)


