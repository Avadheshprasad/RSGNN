import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
import matplotlib.pyplot as plt
import warnings
from utils import *
import csv

class RSGNN:
    """ RWL-GNN (Robust Graph Neural Networks using Weighted Graph Laplacian)
    Parameters
    ----------
    model:
        model: The backbone GNN model in RWLGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
    See details in https://github.com/Bharat-Runwal/RWL-GNN.
    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.best_features= None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

        # self.train_cost = []
        self.valid_cost = []
        self.train_acc = []
        self.valid_acc = []



    def fit(self, noise_features, adj, labels, idx_train, idx_val):
        """Train RWL-GNN.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """

       # print(" Bounded Joint Learning .........")
        args = self.args
        self.symmetric = args.symmetric
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        
        optim_sgl = args.optim
        lr_sgl = args.lr_optim

        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L_noise = D - adj

        # INIT
        self.weight = self.Linv(L_noise)
        self.noise_features=noise_features
        self.weight.requires_grad = True
        self.w_old = torch.zeros_like(self.weight)  ####################  To store previous w value ( w^{t-1} )
        self.weight = self.weight.to(self.device)
        self.w_old = self.w_old.to(self.device)
        # self.features = torch.randn_like(noise_features)
        # a= torch.rand_like(noise_features)
        # range_a = noise_features.max() - noise_features.min()
        # self.features = a* (range_a)+ noise_features.min()

        # shape_a = noise_features.shape
        # # Specify the epsilon value for noise
        # std = 1e-2
        # # Generate random values in the range [-eps/2, eps/2)
        # noise = torch.randn(shape_a) * std

        # # Add the noise to the original tensor
        self.features = noise_features
        
        # self.features = torch.zeros_like(noise_features)
        # print(self.features)
        # print(self.noise_features)
        self.features_old= torch.zeros_like(noise_features)

        self.bound = args.bound

        self.d =  self.features.shape[1]
    
        # c = self.Lstar(2*L_noise*args.alpha - args.beta*(torch.matmul(self.features,self.features.t())) )



        if optim_sgl == "Adam":
            self.sgl_opt =AdamOptimizer(self.weight,lr=lr_sgl)
        elif optim_sgl == "RMSProp":
            self.sgl_opt = RMSProp(self.weight,lr = lr_sgl)
        elif optim_sgl == "sgd_momentum":
            self.sgl_opt = sgd_moment(self.weight,lr=lr_sgl)
        else:
            self.sgl_opt = sgd(self.weight,lr=lr_sgl) 

        t_total = time.time()

        for epoch in range(args.epochs):
            print(epoch)
            bound_error=2 * self.bound**2  * ( torch.log(torch.norm(self.model.gc1.weight)) + torch.log(torch.norm(self.model.gc2.weight)) )
            adj = self.normalize()
            temp_features=self.features
            if(args.only_gcn or args.only_structure):
                temp_features=self.noise_features
            output = self.model(temp_features, adj)
            gcn_loss=args.gamma *F.nll_loss(output[idx_train], labels[idx_train])
            print("loss1+2",gcn_loss+bound_error)
            # loss_x=args.alpha*torch.norm(self.features-self.noise_features, p="fro")**2
            # loss_phi=args.beta*torch.norm(self.L()-L_noise, p="fro")**2
            # loss_trace=args.delta*torch.trace(torch.matmul(self.features.t(),torch.matmul(self.L(),self.features)))
            # sparse_error=args.eta*(torch.sum(torch.log(self.weight+args.eps)))
            # total_error=gcn_loss+loss_x+loss_phi+loss_trace+sparse_error+bound_error
            # print("bound error",bound_error)
            # print("gcn loss",gcn_loss)
            # print("loss features",loss_x)
            # print("loss phi", loss_phi)
            # print("loss trace",loss_trace)
            # print("sparse error",sparse_error)
            # print("total error",total_error)
            # with open('error.csv', 'a', newline='') as csvfile:
            #     fieldnames = ['epoch','bound error','gcn loss','loss features','loss phi','loss trace','sparse error','total error']
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #     # If the file is empty, write the header
            #     if csvfile.tell() == 0:
            #         writer.writeheader()

            #     # Write the value of z to the CSV file
            #     writer.writerow({'epoch':epoch,
            #                      'bound error': bound_error.detach().numpy(),
            #                      'gcn loss':gcn_loss.detach().numpy(),
            #                      'loss features':loss_x.detach().numpy(),
            #                      'loss phi':loss_phi.detach().numpy(),
            #                      'loss trace':loss_trace.detach().numpy(),
            #                      'sparse error':sparse_error.detach().numpy(),
            #                      'total error':total_error.detach().numpy()})

        
            if args.only_gcn:
                estimate_adj = self.A()
                self.train_gcn(epoch, self.noise_features, estimate_adj,
                        labels, idx_train, idx_val)
            elif args.only_structure: 
                for i in range(int(args.outer_steps)):
                    self.train_specific(epoch, self.noise_features, L_noise, labels,
                            idx_train, idx_val)

                for i in range(int(args.inner_steps)):
                    estimate_adj = self.A()
                    self.train_gcn(epoch, self.noise_features, estimate_adj,
                            labels, idx_train, idx_val)
            elif args.only_features:
                for i in range(int(args.feature_steps)):
                    estimate_adj = self.A()
                    self.train_x(epoch, self.features, estimate_adj, labels,
                            idx_train, idx_val)

                for i in range(int(args.inner_steps)):
                    estimate_adj = self.A()
                    self.train_gcn(epoch, self.features, estimate_adj,
                            labels, idx_train, idx_val)
            else:
                for i in range(int(args.feature_steps)):
                    estimate_adj = self.A()
                    self.train_x(epoch, self.features, estimate_adj, labels,
                            idx_train, idx_val)
                    
                for i in range(int(args.outer_steps)):
                    self.train_specific(epoch, self.features, L_noise, labels,
                            idx_train, idx_val)

                for i in range(int(args.inner_steps)):
                    estimate_adj = self.A()
                    self.train_gcn(epoch, self.features, estimate_adj,
                            labels, idx_train, idx_val)

        if args.plots=="y":
            self.plot_acc()
            self.plot_cost()

        #print("Optimization Finished!")
        #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        #print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)



    def w_grad(self,beta,c):
      with torch.no_grad():
        grad_f = self.Lstar(beta*self.L()) - c 
        return grad_f 
      
    def x_grad(self,delta,new_term):
        with torch.no_grad():
            grad_f = torch.matmul((delta*self.L()),self.features) + new_term
            return grad_f 

    def train_x(self,epoch, features, adj, labels, idx_train, idx_val):
        print("sum",torch.sum(self.features))
        difference_mask = self.features != self.noise_features
        count_differing_elements = difference_mask.sum().item()
        print("count_differing_elements", count_differing_elements)
        
        args = self.args
        # estimator = self.estimator
        adj = self.normalize()
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        
        x=features.clone().detach()
        x= x.to(self.device)
        x.requires_grad = True

        output = self.model(x, adj)
        loss_gcn =args.gamma * F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # loss_diffiential = loss_fro + gamma*loss_gcn+args.lambda_ * loss_smooth_feat
        gcn_grad = torch.autograd.grad(
        inputs= x,
        outputs=loss_gcn,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(loss_gcn),
        only_inputs= True,
          )[0]
        
        new_term = 2*args.alpha*(self.features-self.noise_features)
        sgl_grad = self.x_grad(args.delta ,new_term)

        # print("gcn",gcn_grad)
        # print("sgl",sgl_grad)
        total_grad  = sgl_grad + gcn_grad 

        self.features_old = self.features
        # print(total_grad)
        self.features -= -1*1e-3*total_grad
        
        tensor_min = self.features.min()
        print("avd",tensor_min)
        normalized_tensor = self.features - tensor_min
        # Step 2: Divide by the range (max - min)
        tensor_max = self.features.max()
        print(tensor_max)
        range_tensor = tensor_max - tensor_min
        normalized_tensor = normalized_tensor / range_tensor
        
        # Step 3: Apply threshold to set values >= 0.5 to 1 and the rest to 0
        self.features = (normalized_tensor >= 0.5).float()
        print(self.features)
        # self.features = torch.clamp(self.features,min=0)
        


        self.model.eval()
        output = self.model(self.features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print("acc_val",acc_val," best_val" ,self.best_val_acc)
        print("loss", loss_val)

        if args.plots == "y":
            self.train_acc.append(acc_train.detach().cpu().numpy())
            self.valid_cost.append(loss_val.detach().cpu().numpy())
            self.valid_acc.append(acc_val.detach().cpu().numpy())
        
        if args.test=="n":
            print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.best_features= self.features.detach()
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.best_features= self.features.detach()
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1))
                # print('Epoch: {:04d}'.format(epoch+1),
                #       'loss_fro: {:.4f}'.format(loss_fro.item()),
                #       'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                #       'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                #       'loss_total: {:.4f}'.format(total_loss.item()))


    def train_specific(self,epoch, features, L_noise, labels, idx_train, idx_val):
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        
        y = self.weight.clone().detach()
        y = y.to(self.device)
        y.requires_grad = True


        normalized_adj = self.normalize(y)
        output = self.model(features, normalized_adj)
        loss_gcn =args.gamma * F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # loss_diffiential = loss_fro + gamma*loss_gcn+args.lambda_ * loss_smooth_feat

        gcn_grad = torch.autograd.grad(
        inputs= y,
        outputs=loss_gcn,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(loss_gcn),
        only_inputs= True,
          )[0]

        # sq_norm_Aw = torch.norm(self.A(), p="fro")**2

        # new_term = self.bound**2  * (2 * self.Astar(self.A()) - self.w_old) / \
        #            (sq_norm_Aw - self.w_old.t() * self.weight)
        c = self.Lstar(2*L_noise - (args.delta/args.beta)*(torch.matmul(features,features.t())) )
        sgl_grad = self.w_grad(args.beta ,c)

        sparse_grad=args.eta*1 / (args.eps + self.weight)
        total_grad  = sgl_grad + gcn_grad +sparse_grad

        self.w_old = self.weight
        self.weight = self.sgl_opt.backward_pass(total_grad)
        self.weight = torch.clamp(self.weight,min=1e-5)

        self.model.eval()
        normalized_adj = self.normalize()
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # print("loss val", loss_val)
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if args.plots == "y":
            self.train_acc.append(acc_train.detach().cpu().numpy())
            self.valid_cost.append(loss_val.detach().cpu().numpy())
            self.valid_acc.append(acc_val.detach().cpu().numpy())
        
        if args.test=="n":
            print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.best_features= features.detach()
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.best_features= features.detach()
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1))
                # print('Epoch: {:04d}'.format(epoch+1),
                #       'loss_fro: {:.4f}'.format(loss_fro.item()),
                #       'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                #       'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                #       'loss_total: {:.4f}'.format(total_loss.item()))
                


                

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        # estimator = self.estimator
        adj = self.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)

        self.l2_reg =  2 * self.bound**2  * ( torch.log(torch.norm(self.model.gc1.weight)) + torch.log(torch.norm(self.model.gc2.weight)) )  # Added by me
        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.l2_reg
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward(retain_graph = True)
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.best_features= features.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.best_features= features.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def test(self, labels, idx_test):
        """Evaluate the performance of RWL-GNN on test set
        """
        #print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        features=self.best_features
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def A(self,weight=None):
        # with torch.no_grad():
        if weight == None:
            k = self.weight.shape[0]
            a = self.weight
        else:
            k = weight.shape[0]
            a = weight
        n = int(0.5 * (1 + np.sqrt(1 + 8 * k)))
        Aw = torch.zeros((n,n),device=self.device)
        b=torch.triu_indices(n,n,1)
        Aw[b[0],b[1]] =a
        Aw = Aw + Aw.t()
        return Aw

    def Astar(self,adjacency):
        n = adjacency.shape[0]
        k = n * (n - 1) // 2
        weight = torch.zeros(k,device= self.device)
        b = torch.triu_indices(n, n, 1)
        weight = adjacency[b[0], b[1]]
        return weight



    def L(self,weight=None):
        if weight==None:
            k= len(self.weight)
            a = self.weight 
        else:
            k = len(weight)
            a = weight
        n = int(0.5*(1+ np.sqrt(1+8*k)))
        Lw = torch.zeros((n,n),device=self.device)
        b=torch.triu_indices(n,n,1)
        Lw[b[0],b[1]] = -a  
        Lw = Lw + Lw.t()
        row,col = np.diag_indices_from(Lw)
        Lw[row,col] = -Lw.sum(axis=1)
        return Lw     



    def Linv(self,M):
      with torch.no_grad():
        N=M.shape[0]
        k=int(0.5*N*(N-1))
        # l=0
        w=torch.zeros(k,device=self.device)
        indices=torch.triu_indices(N,N,1)
        M_t=torch.tensor(M)
        w=-M_t[indices[0],indices[1]]
        return w


    def Lstar(self,M):
        N = M.shape[1]
        k =int( 0.5*N*(N-1))
        w = torch.zeros(k,device=self.device)
        tu_enteries=torch.zeros(k,device=self.device)
        tu=torch.triu_indices(N,N,1)
    
        tu_enteries=M[tu[0],tu[1]]
        diagonal_enteries=torch.diagonal(M)

        b_diagonal=diagonal_enteries[0:N-1]
        x=torch.linspace(N-1,1,steps=N-1,dtype=torch.long,device=self.device)
        x_r = x[:N]
        diagonal_enteries_a=torch.repeat_interleave(b_diagonal,x_r)
     
        new_arr=torch.tile(diagonal_enteries,(N,1))
        tu_new=torch.triu_indices(N,N,1)
        diagonal_enteries_b=new_arr[tu_new[0],tu_new[1]]
        w=diagonal_enteries_a+diagonal_enteries_b-2*tu_enteries
   
        return w



    def normalize(self,w=None):

        if self.symmetric:
            if w == None:
                adj = (self.A() + self.A().t())
            else:
                adj = self.A(w)
            
            adj = adj + adj.t()
        else:
            if w == None:
                adj = self.A()
            else:
                adj = self.A(w)

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
