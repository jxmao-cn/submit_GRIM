import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle

import argparse
import scipy.sparse as sp
import warnings
import pandas as pd

from main.utils import diffusion_evaluation, diffusion_evaluation_v2
from main.model.gat import SpGAT, SpGATv2
from main.model.model import VAEModel, Encoder, Decoder
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"device: {device}")

parser = argparse.ArgumentParser(description="GenIM")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5']
parser.add_argument("-d", "--dataset", default="cora_ml", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="LT", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))

seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=1, type=int,
                    help="one of: {}".format(", ".join(map(str, sorted(seed_rate)))))

mode = ['normal', 'budget constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))

parser.add_argument("-e", "--epochs", default=600, type=int,
                    help="number of epochs for training")

parser.add_argument("-ev", "--evaluation", default=300, type=int,
                    help="number of epochs for evaluation")

parser.add_argument("-g", "--gemma", default=0.8, type=float,
                    help="the ratio of seed nodes to be selected")

parser.add_argument("-md","--model", default="", type=str,
                    help="enable using local model training results")

parser.add_argument("-sv","--save_model", default=False, action="store_true",
                    help="save the model after training")

args = parser.parse_args()

if args.dataset == 'random5':
    batch_size = 2
    hidden_dim = 4096
    latent_dim = 1024
else:
    batch_size = 16
    hidden_dim = 1024
    latent_dim = 512

print("All your args are as follow, please have a check!")

print("\nParsed Arguments:")
print("=" * 30)
for key, value in vars(args).items():
    print(f"{key:20}: {value}")
print("=" * 30)

def normalize_adj(adj):  
    '''this function is used to normalize the adjacency matrix'''
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    mx=adj + sp.eye(adj.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    adj=mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return torch.Tensor(adj.toarray()).to_sparse()

#graph_path = f'data/{args.dataset}_mean_{args.diffusion_model}{10 * args.seed_rate}.SG'
graph_path = 'data/cora_ml/cora_ml_with_seed_IC10_2025_10_28_1530.SG'
print(f"Loading graph: {graph_path}")
with open(graph_path, 'rb') as f:
    graph = pickle.load(f)

normal_adj, adj = graph['adj'], graph['adj']
seed_tensor = graph['seed']
coverage_list = graph['coverage']
adj=normalize_adj(adj)
adj = adj.to(device)

seed_tensor_np = np.asarray(seed_tensor)
coverage_np = np.asarray(coverage_list, dtype=np.float32)
train_loader = DataLoader(TensorDataset(torch.tensor(seed_tensor_np).float(),
                                        torch.tensor(coverage_np).float()),
                         batch_size=batch_size, shuffle=True, drop_last=False)

if args.model:
    forward_model=torch.load("saved_models/"+args.model+"forward_model_1.pth", 
                       map_location=device,
                       weights_only=False)
    vae_model=torch.load("saved_models/"+args.model+"vae_model_1.pth",map_location=device,weights_only=False)
else:
    encoder = Encoder(input_dim=normal_adj.shape[0],
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim)

    decoder = Decoder(input_dim=latent_dim,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=normal_adj.shape[0])

    vae_model = VAEModel(Encoder=encoder, Decoder=decoder).to(device)

    forward_model = SpGATv2(nfeat=1,
                          nhid=64,
                          nclass=1,
                          dropout=0.2,
                          nheads=4,
                          alpha=0.2)

    optimizer = Adam([{'params': vae_model.parameters()}, {'params': forward_model.parameters()}],
                    lr=1e-3)
    
    forward_model = forward_model.to(device)
    forward_model.train()

    print("Total training epochs:{:}".format(args.epochs))

    overall_begin = time.time()
    for epoch in range(args.epochs):
        begin = time.time()
        total_overall = 0.0
        total_forward_loss = 0.0
        total_reconstruction_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)  # [B, N]
            y_batch = y_batch.to(device)  # [B]

            loss = 0.0
            for i, x_i in enumerate(x_batch):
                y_i = y_batch[i].unsqueeze(0)  # [1]
                x_hat = vae_model(x_i.unsqueeze(0))  # [1, N]
                x_hat_res = x_hat.squeeze(0).unsqueeze(-1)  # [N, 1]

                y_hat = forward_model(x_hat_res, adj)  # [1, 1]
                reproduction_loss = F.binary_cross_entropy(x_hat, x_i.unsqueeze(0), reduction='sum')
                forward_loss = F.mse_loss(y_hat.squeeze(), y_i)

                loss += (reproduction_loss + forward_loss)
                total_reconstruction_loss += reproduction_loss.item()
                total_forward_loss += forward_loss.item()

            total_overall += loss.item()
            loss = loss / x_batch.size(0)

            loss.backward()
            optimizer.step()

        end = time.time()
        print("Epoch: {}".format(epoch + 1),
        "\tTotal: {:.4f}".format(total_overall / len(train_loader.dataset)),
        "\tForward MSE: {:.4f}".format(total_forward_loss / len(train_loader.dataset)),
        "\tReconstruction BCE: {:.4f}".format(total_reconstruction_loss / len(train_loader.dataset)),
        "\tTime: {:.4f}".format(end - begin)
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    overall_end = time.time()
    print("Total Training Time: {:.4f} seconds".format(overall_end - overall_begin))

    insert_str=args.dataset + "_"+args.diffusion_model + str(10 * args.seed_rate)
    if args.save_model:
        torch.save(forward_model, "./saved_models/" + insert_str + "forward_model.pth")
        torch.save(vae_model, "./saved_models/" + insert_str + "vae_model.pth")

def solve(epochs: int = None):
    if epochs is None:
        epochs = args.evaluation

    for p in vae_model.parameters():
        p.requires_grad = False
    for p in forward_model.parameters():
        p.requires_grad = False

    encoder = vae_model.Encoder
    decoder = vae_model.Decoder

    idx_top = int(np.argmax(coverage_np))
    x_init = torch.tensor(seed_tensor_np[idx_top]).float().to(device)
    with torch.no_grad():
        z_hat = encoder(x_init.unsqueeze(0))
    z_hat = z_hat.detach().clone()
    z_hat.requires_grad = True

    z_optimizer = Adam([z_hat], lr=1e-3)
    l1_coeff = 0.001
    for i in range(epochs):
        x_hat = decoder(z_hat)  # [1, N]
        x_hat_res = x_hat.squeeze(0).unsqueeze(-1)  # [N, 1]
        y_hat = forward_model(x_hat_res, adj)  # [1, 1]

        loss = -y_hat.squeeze() + l1_coeff * (torch.sum(torch.abs(x_hat)) / x_hat.shape[1])

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if (i + 1) % 50 == 0 or i == 0 or i == epochs - 1:
            print(f"[Solve] Iter {i+1}: objective={-loss.item():.5f}, pred={y_hat.item():.3f}")

    n_nodes = normal_adj.shape[0]
    k_total = max(1, int(round(n_nodes * (args.seed_rate / 100.0))))
    x_hat = decoder(z_hat).detach()
    model_topk = x_hat.topk(k_total, dim=1)
    seeds_model = model_topk.indices[0].cpu().numpy().tolist()

    influence_model = diffusion_evaluation_v2(normal_adj, seeds_model, diffusion=args.diffusion_model)
    print(f"Influence model-only: {influence_model} (k={k_total})")

    return seeds_model, influence_model

seeds_model, influence_model = solve()

print("Model-only selection done.")
print({
    'model_only': {'seeds': seeds_model[:10], 'influence': influence_model}
})
