# Code Adapted from Pro-GNN(https://github.com/ChandlerBang/Pro-GNN)
import time
import argparse
import numpy as np
import torch
import csv 

from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test

import scipy.sparse as sp
from scipy.sparse.linalg import norm



# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--bound', type=float, default=0, help='weight of bound importance')
parser.add_argument('--two_stage',type = str,help = "Use Two Stage",default="n")
parser.add_argument('--optim',type = str,help = "Optimizer",default="sgd")
parser.add_argument('--lr_optim',type = float, help = "learning rate for the graph weight update" ,default=1e-2)
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--decay',type=str,default="n",help="whether to use decay or not")
parser.add_argument('--plots',type=str,default="n",help="whether to plot the acc or not")
parser.add_argument('--test',type=str,default="y",help="Test only")
parser.add_argument('--manual_feature_attack', action='store_true',
        default=False, help='test the performance on manual feature attack')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')

parser.add_argument('--normalize', action='store_true',
        default=False, help='normalize features')


parser.add_argument('--only_structure', action='store_true',
        default=False, help='test the performance of structure and gcn without other components')
parser.add_argument('--only_features', action='store_true',
        default=False, help='test the performance of features and gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate for GNN model.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'citeseer', 'cora_ml', 'polblogs','pubmed', 'acm', 'blogcatalog', 'uai', 'flickr'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--mean', type=float, default=0.0, help="mean for gaussian noise")
parser.add_argument('--variance', type=float, default=1.0, help="variance for gaussian noise")
parser.add_argument('--n_flip', type=int,  default=50, help='Number of epochs to train.')
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--epochs_pre', type=int,  default=500, help='Number of epochs to train in Two-Stage.')
parser.add_argument('--alpha', type=float, default=1, help='')
parser.add_argument('--gamma', type=float, default=1, help='weight of GCN')
parser.add_argument('--eta', type=float, default=0, help='')
parser.add_argument('--delta', type=float, default=0, help='')
parser.add_argument('--eps', type=float, default=1e-3, help='')
parser.add_argument('--beta', type=float, default=1, help='weight of feature smoothing')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--feature_steps', type=int, default=1, help='steps for feature optimization')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')


args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape)).to_dense()

def torch_dense_sensor_to_sparse_mx(dense_tensor):
  return sp.csr_matrix(dense_tensor.numpy())

# Function to add Gaussian noise to a tensor
def add_gaussian_noise(tensor, mean=0.0, variance=1.0):
    torch.manual_seed(42)
    std = (variance**0.5)
    noise = torch.randn_like(tensor) * std + mean
    noisy_tensor = tensor + noise
    torch.manual_seed(args.seed)
    return noisy_tensor

def feature_flip(number_flip, x):
    x1=x.copy()
    np.random.seed(42)
    if number_flip > 0:
        for i in range(x1.shape[0]):
            flip_indices = np.random.choice(x1.shape[1], size=int(number_flip), replace=False)
            idex_fea = x1[i, flip_indices].toarray()
            x1[i,flip_indices] = np.where(idex_fea == 0, 1, 0)
    np.random.seed(args.seed)
    return x1

def normalize_sparse_matrix_by_norm(matrix):
    row_norms = norm(matrix, axis=1)
    return matrix / row_norms[:, np.newaxis]

# Here the random seed is to split the train/val/test data,
# we need to set the random seed to be the same as that when you generate the perturbed graph
# but now change the setting from nettack to prognn which directly loads the prognn splits
# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
data = Dataset(root='/tmp/', name=args.dataset,setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
print("Adj Shape",adj.shape)
print("Feature shape", features.shape)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if args.dataset == 'pubmed':
    # just for matching the results in the paper, see details in https://github.com/ChandlerBang/Pro-GNN/issues/2
    #print("just for matching the results in the paper," + \
    #      "see details in https://github.com/ChandlerBang/Pro-GNN/issues/2")
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
            val_size=0.1, test_size=0.8, stratify=encode_onehot(labels), seed=15)

if args.attack == 'no':
    perturbed_adj = adj

if args.manual_feature_attack:
    # features1= sparse_mx_to_torch_sparse_tensor(features)
    # # Add Gaussian noise to the dense tensor
    # noise_features = add_gaussian_noise(features1, args.mean, args.variance)
    # perturbed_features=torch_dense_sensor_to_sparse_mx(noise_features)
    perturbed_features = feature_flip(args.n_flip, features)
else:
    perturbed_features=features
  
if args.normalize:
    perturbed_features = normalize_sparse_matrix_by_norm(perturbed_features)

# print(perturbed_features)

if args.attack == 'meta' or args.attack == 'nettack':
    perturbed_data = PrePtbDataset(root='/tmp/',
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj
    if args.attack == 'nettack':
        idx_test = perturbed_data.target_nodes

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.only_gcn:
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
else:
    from bounded_gcn import BoundedGCN
    model = BoundedGCN(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout, device=device, bound=args.bound)

    if args.two_stage == "y":
        from Bounded_two_stage import RSGNN
    else:
        from BoundedJointLearning import RSGNN


if args.only_gcn:

    # perturbed_adj, perturbed_features, labels = preprocess(perturbed_adj, perturbed_features, labels, preprocess_adj=False, device=device)

    model.fit(perturbed_features, perturbed_adj, labels, idx_train, idx_val, verbose=True, train_iters=args.epochs)
    result=model.test(idx_test)
    with open('output_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['seed','only_gcn','only_features','only_structure','all']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the value of z to the CSV file
        writer.writerow({'seed':args.seed,'only_gcn': result})


else:
    perturbed_adj, perturbed_features, labels = preprocess(perturbed_adj, perturbed_features, labels, preprocess_adj=False, device=device)
    rsgnn = RSGNN(model, args, device)
    if args.two_stage=="y":
        adj_new = rsgnn.fit(perturbed_features, perturbed_adj)
        model.fit(perturbed_features, adj_new, labels, idx_train, idx_val, verbose=False, train_iters=args.epochs,bound=args.bound) #######
        model.test(idx_test)
    else:
        rsgnn.fit(perturbed_features, perturbed_adj, labels, idx_train, idx_val)
        result=rsgnn.test( labels, idx_test)
        with open('output_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['seed','only_gcn','only_features','only_structure','all']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file is empty, write the header
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the value of z to the CSV file
            if(args.only_features):
                writer.writerow({'seed':args.seed,'only_features': result})
            elif(args.only_structure):
                writer.writerow({'seed':args.seed,'only_structure': result})
            else:
                writer.writerow({'seed':args.seed,'all': result})


