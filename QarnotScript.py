#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import Graphs
import networkx as nx
import torch.optim as optim
import os
import pickle as pkl
import argparse
from datetime import timedelta
from tsmoothie.smoother import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from river import preprocessing, compose
from torchsummaryX import summary
from operator import add
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from itertools import islice
from utils import gradient, seed_everything, gap_fn, online_gap, inner_prod, lmo_fn




def testcutoff(week, dataset):
    '''
    Parameters
    ----------
    week : int
        Week to be left for validation
    dataset : DataFrame

    Returns
    -------
    trainidx : DataFrame
        Training Dataframe
    testidx : DataFrame
        Validation DataFrame

    '''
    cutoff = dataset.index.max() - timedelta(weeks=week)
    trainidx = dataset.loc[dataset.index < cutoff]
    testidx = dataset.loc[dataset.index >= cutoff]
    return trainidx, testidx


def MovingAverage(data, window=10):
    rolling = data.rolling(window=window)
    rolling_mean = rolling.mean()
    rolling_mean[:window] = data[:window]
    return rolling_mean


def createXY(dataset, lookback, lookahead, lag, y_feature, step=1):
    '''
    Parameters
    ----------
    dataset : Dataframe
        Data to create X and Y
    lookback : int
        Length of X sequence. Shape (lookback, nb_feature)
    lookahead : int
        Length of Y sequence. Shape (lookahead,)
    lag : int
        Distance between X and Y

    y_feature : str

    Returns
    -------
    X : list
        List of X
    y : list
        List of Y

    '''
    data = dataset.values
    X, y = [], []
    for i in range(0,data.shape[0]-lookahead-lag,step):
        end_idx = i + lookback
        if (end_idx > data.shape[0]) or (end_idx+lag+lookahead > data.shape[0]):
            break
        data_feature = data[i:end_idx]
        data_target = dataset[y_feature][end_idx+lag:end_idx+lag+lookahead]
        data_target = data_target.values
        X.append(data_feature)
        y.append(data_target)
    return X,y


def toDataLoader(dataX, dataY, batch_size, window_len=4,smooth_=True):
    '''
    Parameters
    ----------
    dataX : List
        X from createXY
    dataY : List
        Y from createXY
    batch_size : int
        number of data point for each iterations of dataloader
    window_len : int
        Time window for exponential smoother. The default is 4.
    smooth_ : bool, optional
        The default is True.

    Returns
    -------
    loader : torch.DataLoader
    '''
    batch = len(dataX)
    timestepX = dataX[0].shape[0]
    timestepY = dataY[0].shape[0]
    features = dataX[0].shape[1]
    tensorX, tensorY = torch.empty((batch,timestepX,features)), torch.empty((batch,timestepY))
    for i,x,y in zip(range(batch),dataX,dataY):
        value = x
        recompense = x[:window_len] #first window_len-th element of smoothed data will be NaN.
        if smooth_:
            expSmooth = ExponentialSmoother(window_len=window_len,alpha=0.3)
            expSmooth.smooth(x.T)
            value = expSmooth.smooth_data.T
            value = np.concatenate([recompense,value],axis=0) #replace NaN by original data

        transX = torch.tensor(value, dtype=torch.float32)
        transY = torch.tensor(y, dtype=torch.float32)

        tensorX[i,:] = transX
        tensorY[i,:] = transY

    datawrap = TensorDataset(tensorX,tensorY)
    loader = DataLoader(datawrap,batch_size=batch_size,drop_last=False)
    return loader




class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_encoder, time_step_in, time_step_out):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.output_encoder = output_encoder
        self.time_step_in = time_step_in
        self.time_step_out = time_step_out
        self.num_layers = 1

        self.encoder = nn.LSTM(self.input_size, self.output_encoder,
                               num_layers=self.num_layers, batch_first=True, bias=True)
        self.decoder = nn.LSTM(self.output_encoder,
                               self.output_encoder, batch_first=True)

        self.dropout = nn.Dropout(0.20)
        #self.batchnorm = nn.BatchNorm1d(self.time_step_in)

        self.LinearCombine = nn.Linear(
            self.input_size + self.output_encoder, 1, bias=False)

        self.linear = nn.Linear(
            self.time_step_in, self.time_step_out, bias=True)
        #self.linear2 = nn.Linear(
        #    self.time_step_out, self.time_step_out, bias=True)

    def forward(self,x):
        #---Encoder---#
        out_en,(h_en,_) = self.encoder(x)
        h_en = h_en.squeeze().unsqueeze(1)
        h_en = h_en.repeat(1,self.time_step_in,1)
        h_en = torch.tanh(h_en)

        #---Decoder---#
        out_de, (h_de,_) = self.decoder(h_en)
        out_de = self.dropout(out_de)

        #---Concatenate Input---#
        concat = torch.cat((out_de,x),dim=-1)
        concat = torch.tanh(concat)
        out = self.LinearCombine(concat)
        out = out.view(-1,self.time_step_in)
        #---MLP---#
        out = torch.sigmoid(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class RMSELoss(torch.nn.Module):
    def __init__(self):
        self.eps = 1e-6
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss




class Oracle:
    def __init__(self, dim, radius_W, t):
        self.oracle_type = t

        self.oracle_name = 'Follow The Perturbed Leader'
        self.G = [torch.zeros(k) for k in dim]
        self.r = radius_W
        self.V = self.lmo_fn()

    def lmo_fn(self):
        res = [0.]*len(self.G)
        for k in range(len(self.G)):
            shape = self.G[k].shape
            if len(shape) == 4:
                myelem = torch.zeros(shape)
                for chan_out in range(shape[0]):
                    for chan_in in range(shape[1]):
                        small_g = self.G[k][chan_out][chan_in]
                        num_rows, num_cols = small_g.shape
                        P = small_g + (-0.5 + torch.randn(*small_g.shape))
                        cols = torch.argmax(torch.abs(P), 1)
                        rows = torch.arange(num_rows)
                        indices = torch.LongTensor([rows.tolist(), cols.tolist()])
                        flatten_P = P.T.flatten()[rows + cols * num_rows]
                        values = -self.r  * torch.sign(flatten_P)
                        res_tmp = torch.sparse.FloatTensor(indices, values, small_g.shape).to_dense()
                        myelem[chan_out][chan_in] = res_tmp
                res[k] = myelem

            elif len(shape)==3:
                myelem = torch.zeros(shape)
                for chan_out in range(shape[0]):
                    small_g = self.G[k][chan_out]
                    num_rows, num_cols = small_g.shape
                    P = small_g + (-0.5 + torch.randn(*small_g.shape))
                    cols = torch.argmax(torch.abs(P), 1)
                    rows = torch.arange(num_rows)
                    indices = torch.LongTensor([rows.tolist(), cols.tolist()])
                    flatten_P = P.T.flatten()[rows + cols * num_rows]
                    values = -self.r  * torch.sign(flatten_P)
                    res_tmp = torch.sparse.FloatTensor(indices, values, small_g.shape).to_dense()
                    myelem[chan_out] = res_tmp

                res[k] = myelem

            elif len(shape)==2:
                small_g = self.G[k]
                num_rows, num_cols = small_g.shape
                P = small_g + (-0.5 + torch.rand(*small_g.shape))
                cols = torch.argmax(torch.abs(P), 1)
                rows = torch.arange(num_rows)
                indices = torch.LongTensor([rows.tolist(), cols.tolist()])
                flatten_P = P.T.flatten()[rows + cols * num_rows]
                values = -self.r  * torch.sign(flatten_P)
                res_tmp = torch.sparse.FloatTensor(indices, values, P.shape).to_dense()
                res[k] = res_tmp

            elif len(shape)==1:
                nb_rows = shape
                P = self.G[k] +  (-0.5 + torch.rand(nb_rows))
                rows = torch.argmax(torch.abs(P))
                s = torch.zeros(nb_rows)
                s[rows] = -self.r * torch.sign(P[rows])
                res[k] = s

        return res

    def update_FTPL(self, fb):
        for k in range(len(self.G)):
            self.G[k] = self.G[k] + fb[k]
        self.V = self.lmo_fn()

    def update(self, fb):
        self.update_FTPL(fb)


# In[8]:


class DMFW(optim.Optimizer):

    def __init__(self, params, eta_coef, eta_exp, reg_coef, L,oracle_type, radius, loss_function,
                 adjacency_matrix_line):
        defaults = dict(eta_coef=eta_coef,eta_exp = eta_exp,reg_coef = reg_coef, L=L)
        super(DMFW, self).__init__(params, defaults)



        self.num_layers = len(self.param_groups[0]['params'])
        self.A = adjacency_matrix_line
        self.init_weights = self.param_groups[0]['params']
        #print(self.init_weights)
        self.loss_function = loss_function
        self.radius = radius

        for group in self.param_groups:
            self.eta_coef = group['eta_coef']
            self.eta_exp = group['eta_exp']
            self.reg_coef = group['reg_coef']
            self.L = group['L']

        self.dim = [k.shape for k in self.param_groups[0]['params']]
        self.Ws = [[torch.zeros(k.shape, requires_grad=True) for k in self.param_groups[0]['params']]
                   for l in range(self.L+1)]
        self.Ws[0] = self.init_weights#[torch.zeros(k, requires_grad=True) for k in self.dim]#self.init_weights

        self.oracles = [Oracle(self.dim,radius,'FTPL') for l in range(self.L)]


    def resetWs(self, param):
        self.Ws = [[torch.zeros(k.shape, requires_grad=True) for k in self.param_groups[0]['params']]
                   for l in range(self.L+1)]
        self.Ws[0] = param #[torch.zeros(k, requires_grad=True) for k in self.dim]#param
        self.curr_as = [torch.zeros(k.shape, requires_grad=True) for k in self.param_groups[0]['params']]


    def networkAverage(self,l,weight=True):
        if weight:
            weighted_sum = []
            for k in range(self.num_layers):
                weighted_tmp = torch.zeros(self.init_weights[k].shape)
                for j in range(self.num_nodes):
                    weighted_tmp += self.A[j]*self.neighbors[j].Ws[l][k]
                weighted_sum.append(weighted_tmp)
        else:
            weighted_sum = []
            for k in range(self.num_layers):
                weighted_tmp = torch.zeros(self.init_weights[k].shape)
                for j in range(self.num_nodes):
                    weighted_tmp += self.A[j]*self.neighbors[j].grad[l][k]
                weighted_sum.append(weighted_tmp)
        return weighted_sum


    @torch.no_grad()
    def step(self,neighbors, l, closure=None):
        self.num_nodes = len(neighbors)
        self.neighbors = neighbors

        eta = min(self.eta_coef/((l+1)**self.eta_exp),1)
        network_val = self.networkAverage(l,weight=True)
        for k in range(self.num_layers):
            self.Ws[l+1][k] = network_val[k] + eta * (self.oracles[l].V[k] -
                                                                    network_val[k])
            if self.Ws[l+1][k].grad is not None:
                self.Ws[l+1][k].grad.zeros_()

            self.Ws[l+1][k] = self.Ws[l+1][k].detach().clone().requires_grad_(True)


    def assignWeight(self, model, state):
        with torch.no_grad():
            for (name,param),ele in zip(model.named_parameters(),state):
                param.copy_(nn.Parameter(ele))
        return model


    @torch.enable_grad()
    def gradient(self,loss, weight, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(loss)
        grad = torch.autograd.grad(loss, weight, grad_outputs = grad_outputs, create_graph=True)
        return grad


    def computeGrad(self, model, weight, dataX, dataY):
        model_tmp = self.assignWeight(model, weight)
        for param in model_tmp.parameters():
            if param.grad != None:
                param.grad.zero_()
                param.requires_grad_(True)
        x,y = dataX, dataY#iter(data).next()
        output = model_tmp(x)
        loss_tmp = self.loss_function(output, y)
        #loss_tmp.backward()
        grad_val = self.gradient(loss_tmp, list(model_tmp.parameters()))
        return grad_val

    def initGrad(self, model, dataX,dataY):
        self.grad = [[torch.zeros(k.shape) for k in self.param_groups[0]['params']]
                     for l in range(self.L+1)]
        modeltmp = deepcopy(model)
        self.grad[0] = self.computeGrad(model, self.Ws[0], dataX,dataY)


    def updateAs(self, ds, l):
        tmp = [(2/(l+1)**(2/3))*(m-n) for m,n in zip(ds,self.curr_as)]
        self.curr_as = list(map(add,tmp, self.curr_as))


    def updateOracle(self,model,dataX,dataY,l,stochastic=False):
        model1 = deepcopy(model)
        model2 = deepcopy(model)

        curr_ds = self.networkAverage(l, weight=False)
        reg_ds = [x*self.reg_coef for x in curr_ds]

        if stochastic:
            self.updateAs(curr_ds,l)
            reg_as = [self.reg_coef*x for x in self.curr_as]
            self.oracles[l].update(reg_as)
            grad_tmp1 = self.computeGrad(model1,self.Ws[l+1], dataX,dataY)
            grad_tmp = [e1 - e2 for e1, e2 in zip(grad_tmp1, self.grad[l])]
            self.grad[l+1] = list(map(add,grad_tmp, self.curr_as))
        else :
            self.oracles[l].update(reg_ds)
            grad_tmp1 = self.computeGrad(model1,self.Ws[l+1], dataX,dataY)
            grad_tmp = [e1 - e2 for e1, e2 in zip(grad_tmp1, self.grad[l])]
            self.grad[l+1] = list(map(add,grad_tmp, curr_ds))



def loadPickle(filepath):
    with open(filepath,"rb") as f:
        file = pkl.load(f)
    return file

def savePickle(filepath,obj):
    with open(filepath,"wb") as f:
        pkl.dump(obj, f)




class Trainer:
    def __init__(self, graph, dataroom1, model_name, model_arguments, loss, iterations, print_freq):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.model_name = model_name
        self.model_arguments = model_arguments
        self.A = torch.tensor(nx.adjacency_matrix(graph).toarray())
        self.data1 = dataroom1
        #self.data2 = dataroom2
        #self.data3 = dataroom3
        self.num_iterations = iterations
        self.loss = loss
        self.print_freq = print_freq
        self.obj_values = np.ndarray(
            (int(np.round(self.num_iterations / self.print_freq)) + 1, 4),
            dtype='float')

        self.optimizers = [0.] * self.num_nodes
        self.models = [0.] * self.num_nodes
        self.gaps = [0.] * self.num_nodes

    def reset(self):
        self.optimizers = [0.]*self.num_nodes
        self.models = [0.]*self.num_nodes
        self.losses = [0.]*self.num_nodes
        self.valid_losses=[0.]*self.num_nodes
        self.gaps = [0.]*self.num_nodes

        self.obj_values = np.ndarray(
            (int(np.round(self.num_iterations / self.print_freq)) + 1,2),
            dtype='float')

    def nodeInit(self, datalist,stochastic = False,minibatch=False):
        nodes = {}
        for i in range(self.num_nodes):
            nodewrap = TensorDataset(datalist[i][0],datalist[i][1])
            #print(datalist[i][0].shape)
            if stochastic==True:
                if minibatch:
                    nodes[i] = DataLoader(nodewrap, batch_size=int(torch.randint(1,datalist[i][0].shape[0],size=(1,1))), shuffle=True)
                else :
                    nodes[i] = DataLoader(nodewrap, batch_size=1,shuffle=True)
            else:
                 nodes[i] = DataLoader(nodewrap, batch_size=datalist[i][0].shape[0])
        return nodes

    def weight_reset(self,layer):
        if isinstance(layer, nn.LSTM) or isinstance(layer, nn.Linear):
            layer.reset_parameters()

    def assignWeight(self, model, state):
        with torch.no_grad():
            for (name,param),ele in zip(model.named_parameters(),state):
                param.copy_(nn.Parameter(ele))
        return model

    def saveCheckPts(self,t,weight,path):
        check_pts = {}
        for i in range(self.num_nodes):
            ckp_i = {"t":t,
                    "weight":weight[i],
                    "oracles":[self.optimizers[i].oracles[l].G for l in range(self.optimizers[i].L)],
                    "loss": self.losses[i]}#,
                    #"avg_loss": self.avg_loss[i]}
            check_pts[i] = ckp_i
        torch.save(check_pts, path + "checkpts_models" + ".tar")

    def saveModel(self,path_to_save):
        model = self.model_name(*self.model_arguments)
        torch.save(model, path_to_save + "model_architecture.pt")
        for i in range(self.num_nodes):
            torch.save(self.models[i].state_dict(), path_to_save + "model_state_dict_"+str(i)+".pt")


    def loadCheckPts(self,path):
        self.checkpts = torch.load(path)

    def saveOutput(self, obj,path,name):
        np.savetxt(path + name, obj, delimiter=",")

    def tensorScaling(self,X,y, scalerX, scalerY):
        toPdX = pd.DataFrame(X.view(X.shape[0]*X.shape[1],X.shape[2]).numpy())
        toPdy = pd.DataFrame(y.view(y.shape[0]*y.shape[1],1).numpy())
        scalerX.learn_many(toPdX)
        scalerY.learn_many(toPdy)
        Xscaled = scalerX.transform_many(toPdX)
        yscaled = scalerY.transform_many(toPdy)
        tensorX = torch.tensor(Xscaled.values, dtype=torch.float32).view(X.shape[0],X.shape[1],X.shape[2])
        tensory = torch.tensor(yscaled.values, dtype=torch.float32).view(y.shape[0], y.shape[1])
        return tensorX, tensory

    def train(self, optimizer, L, eta_coef, eta_exp, reg_coef, oracle_type, radius,
              lookback, lookahead, batch_size, lag, smooth_, valid_cutoff, yfeat,
              path_to_data1, path_to_data2, path_to_data3, path_checkpts,
              path_to_save, continue_training = False, stochastic=False, minibatch=False):
        '''

        Parameters
        ----------
        optimizer : Object
            Optimizer class
        L : int
            Number of linear oracles
        eta_coef : float
        eta_exp : float
        reg_coef : float
        oracle_type : string
            type of oracle
        radius : int
            Raidus of linear oracle
        lookback : int
            Length of X sequence
        lookahead : int
            Length of Y sequence
        batch_size : int
            Batch to load for each iterations
        lag : int
            Distance between X and Y
        smooth_ : bool
            Whether to use smoother
        valid_cutoff : int
            Number of week to be left for validation
        yfeat : str
            Feature to learning
        path_to_data1 : str
            Path to data for node 1 (Babbage)
        path_to_data2 : str
            Path to data for node 2 (Babyfoot)
        path_to_data3 : str
            Path to data for node 3 (Jacquard)
        path_checkpts : str
            Path to checkpoint. For the first run, this value will be None
        path_to_save : str
            Path to save checkpoint
        continue_training : Bool, optional
            Indicate if the training is a continuation from the last checkpoint or not. The default is False.
        stochastic : Bool, optional
            Use stochastic algorithm (Let it be False for now). The default is False.
        minibatch : Bool, optional
            Use minibatch if stochastic is True (Let it be False for now). The default is False.

        Raises
        ------
        AttributeError
            If checkpoints is not defined

        Returns
        -------
        Wnew : List
            Learned Weights
        loss_list : list
            Loss values

        '''

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)



        seed_everything()
        self.reset()
        loss_list = []
        gap_list = []
        lossvalid_list = []
        scalerX = preprocessing.StandardScaler()
        #scalerY = preprocessing.StandardScaler()

        for i in range(self.num_nodes):
            self.models[i] = self.model_name(*self.model_arguments)
            self.optimizers[i] = optimizer(self.models[i].parameters(),
                                                    eta_coef,
                                                    eta_exp,
                                                    reg_coef,
                                                    L,
                                                    oracle_type,
                                                    radius,
                                                    self.loss,
                                                    self.A[i])

        if continue_training:
            scalerXbab = loadPickle(path_checkpts+"scalerXbabbage.pkl")
            #scalerYbab = loadPickle(path_checkpts+"scalerYbab.pkl")
            scalerXbaby = loadPickle(path_checkpts+"scalerXbabyfoot.pkl")
            #scalerYbaby = loadPickle(path_checkpts+"scalerYbaby.pkl")
            scalerXjac = loadPickle(path_checkpts+"scalerXjacquard.pkl")
            #scalerYjac = loadPickle(path_checkpts+"scalerYjac.pkl")

            self.loadCheckPts(path_checkpts+"checkpts_models.tar")
            try :
                for i in range(self.num_nodes):
                    for l in range(L):
                        self.optimizers[i].oracles[l].G = self.checkpts[i]["oracles"][l].copy()
            except:
                raise AttributeError("Checkpoint is not defined, please call loadcheckpts(path_to_checkpts)")

        else:
            scalerXbab = preprocessing.StandardScaler()
            #scalerYbab = preprocessing.StandardScaler()
            scalerXbaby = preprocessing.StandardScaler()
            #scalerYbaby = preprocessing.StandardScaler()
            scalerXjac = preprocessing.StandardScaler()
            #scalerYjac = preprocessing.StandardScaler()


        grad_avg = [[torch.zeros(k.shape) for k in list(self.models[i].parameters())]
                   for i in range(self.num_nodes)]
        inner_avg = [0.] * self.num_nodes

        print("Loading Data")
        babbage_chunk = pd.read_csv(path_to_data1, parse_dates=["time"],index_col=0)
        baby_chunk = pd.read_csv(path_to_data2, parse_dates=["time"],index_col=0)
        jacquard_chunk = pd.read_csv(path_to_data3, parse_dates=["time"],index_col=0)

        train_bab,valid_bab = testcutoff(week=valid_cutoff,dataset=babbage_chunk)
        train_baby,valid_baby = testcutoff(week=valid_cutoff,dataset=baby_chunk)
        train_jac,valid_jac = testcutoff(week=valid_cutoff, dataset=jacquard_chunk)

        scalerXbab.learn_many(train_bab)
        train_bab = scalerXbab.transform_many(train_bab)

        scalerXbaby.learn_many(train_baby)
        train_baby = scalerXbaby.transform_many(train_baby)

        scalerXjac.learn_many(train_jac)
        train_jac = scalerXjac.transform_many(train_jac)


        bab, y_bab = createXY(train_bab,lookback=lookback,lookahead=lookahead,lag=lag,y_feature=yfeat)
        baby, y_baby = createXY(train_baby,lookback=lookback,lookahead=lookahead,lag=lag,y_feature=yfeat)
        jac, y_jac = createXY(train_jac,lookback=lookback,lookahead=lookahead,lag=lag,y_feature=yfeat)
        print("Creating Bab Loader")
        loader_bab = toDataLoader(bab,y_bab,batch_size,window_len=10,smooth_=True)
        print("Creating Baby Loader")
        loader_baby = toDataLoader(baby,y_baby,batch_size,window_len=10,smooth_=True)
        print("Creating Jac Loader")
        loader_jac = toDataLoader(jac,y_jac,batch_size,window_len=10,smooth_=True)
        print("---Done Loading---")

        for t, ((dt1,lb1),(dt2,lb2),(dt3,lb3)) in enumerate(zip(loader_bab,loader_baby,loader_jac)):

            tmp_dts = [(dt1,lb1),(dt2,lb2),(dt3,lb3)]


            for i in range(self.num_nodes):
                self.models[i].apply(self.weight_reset)
                self.optimizers[i].resetWs([param for param in self.models[i].parameters()])
            #---Frank-Wolfe Steps---#
            for l in range(L):
                for i in range(self.num_nodes):
                    self.optimizers[i].step(self.optimizers,l)

            self.Wnew = [self.optimizers[i].Ws[L] for i in range(self.num_nodes)]
            #nodes_data = self.nodeInit(tmp_dts,stochastic=stochastic, minibatch=minibatch)

            #---Compute regret---#
            for i in range(self.num_nodes):
                self.models[i] = self.assignWeight(self.models[i],self.Wnew[i])
                outputs = self.models[i](tmp_dts[i][0])
                curr_loss = self.loss(outputs,tmp_dts[i][1])
                #curr_gradient = gradient(curr_loss, list(self.models[i].parameters()))

                #wnew_detach = [p.clone().detach() for p in self.Wnew[i]]
                #inner_grad = sum(list(map(inner_prod, curr_gradient, wnew_detach)))
                #inner_avg[i] += inner_grad
                #inner_avg[i] /= (t+1)

                #grad_avg[i] = list(map(add, grad_avg[i], curr_gradient))
                #grad_avg[i] = list(map(lambda x: x/(t+1),grad_avg[i]))

                #self.gaps[i] = online_gap(grad_avg[i],inner_avg[i],radius)
                self.losses[i] = curr_loss #- opt_loss

                self.optimizers[i].initGrad(self.models[i],tmp_dts[i][0],tmp_dts[i][1])


            #---Feedback Oracle---#
            for l in range(L):
                for i in range(self.num_nodes):
                    self.optimizers[i].updateOracle(self.models[i], tmp_dts[i][0], tmp_dts[i][1],l, stochastic)

            #print(torch.tensor(self.losses).detach().numpy())
            gap_list.append(torch.tensor(self.gaps).detach().numpy())
            loss_list.append(torch.tensor(self.losses).detach().numpy())
            loss = max(self.losses)
            gap=max(self.gaps)

            print("t : {} loss : {} gap : {} scaler mean:{}".format(t, loss,gap,scalerXbab.means["temperature"]))

        loss_list = np.array(loss_list)
        #gap_list = torch.tensor(gap_list).detach().numpy()
        #self.obj_values[10*it + t] = [10*it+t,loss.detach().numpy()]

        self.saveCheckPts(t,self.Wnew, path_to_save) #Uncomment this line to save checkpoints
        self.saveModel(path_to_save) #Uncomment this line to save model
        self.saveOutput(loss_list, path_to_save, "trainloss.csv")
        self.saveOutput(gap_list, path_to_save,"traingaps.csv")#Uncomment this line to save output

        #savePickle(path_to_save+"scalerYbab.pkl",scalerYbab)
        savePickle(path_to_save+"scalerXbabbage.pkl",scalerXbab)

        #savePickle(path_to_save+"scalerYbaby.pkl",scalerYbaby)
        savePickle(path_to_save+"scalerXbabyfoot.pkl",scalerXbaby)

        #savePickle(path_to_save+"scalerYjac.pkl",scalerYjac)
        savePickle(path_to_save+"scalerXjacquard.pkl",scalerXjac)


        '''
        #----Validation----
        bab, y_bab = createXY(valid_bab,lookback=lookback,lookahead=lookahead,lag=lag,y_feature=yfeat)
        baby, y_baby = createXY(valid_baby,lookback=lookback,lookahead=lookahead,lag=lag,y_feature=yfeat)
        jac, y_jac = createXY(valid_jac,lookback=lookback,lookahead=lookahead,lag=lag,y_feature=yfeat)

        loader_bab = toDataLoader(bab,y_bab,batch_size,smooth_=False, training=False)
        loader_baby = toDataLoader(baby,y_baby,batch_size,smooth_=False, training=False)
        loader_jac = toDataLoader(jac,y_jac,batch_size,smooth_=False, training=False)


        for t, ((dt1,lb1),(dt2,lb2),(dt3,lb3)) in enumerate(zip(loader_bab,loader_baby,loader_jac)):
            if t%10==0:
                print("t validation :{}".format(t))

            tmp_dts = [(dt1,lb1),(dt2,lb2),(dt3,lb3)]
            for i in range(self.num_nodes):
                self.models[i] = self.assignWeight(self.models[i],self.Wnew[i])
                outputs = self.models[i](dt1)
                curr_loss = self.loss(outputs,lb1)
                self.valid_losses[i] = curr_loss

            lossvalid_list.append(torch.tensor(self.valid_losses).detach().numpy())

        lossvalid_list = np.array(lossvalid_list)
        self.saveOutput(lossvalid_list, path_to_save, "validloss.csv")'''

        return self.Wnew, loss_list


# In[11]:

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--L",type=int,help="Number of linear oracles",default=20)
parser.add_argument("--lookback",type=int,help="Length of learning sequence",default=40)
parser.add_argument("--lookahead",type=int,help="Length of prediction sequence",default=20)
parser.add_argument("--batchsize",type=int,help="batchsize",default=50)
parser.add_argument("--eta_exp",type=float,help="learning rate exponent",default=0.90)
parser.add_argument("--reg_coef",type=float,help="regularizer of oracle update",default=2)
parser.add_argument("--chunk_checkpts",type=int,help="last learned datachunk",required=True)
parser.add_argument("--chunk_tosave",type=int,help="next learning datachunk",required=True)
parser.add_argument("--continuous_training",choices=['True','False'],default='True',
    help="Continue training from last checkpts",required=True)

argv = parser.parse_args()


nb_nodes = 3
cycle,name = Graphs.cycle_graph(nb_nodes)
grid1, name = Graphs.gridgraph(1,1)
loss_func = nn.SmoothL1Loss()


# In[12]:


print_freq = 1e0
num_iters_base = 586
eta_coef_DMFW = 1
eta_exp_DMFW = argv.eta_exp
rho_coef_DMFW = 4e-0
rho_exp_DMFW = 1/2
reg_coef_DMFW = argv.reg_coef
L_DMFW = argv.L

lookback = argv.lookback
lookahead = argv.lookahead
batch_size=argv.batchsize
lag = 0


path_jacquard = "./QarnotData/jacquard/"
path_babyfoot = "./QarnotData/babyfoot/"
path_babbage = "./QarnotData/babbage/"
list_babbage = sorted([f for f in os.listdir(path_babbage) if not f.startswith('.')])
list_jacquard = sorted([f for f in os.listdir(path_jacquard) if not f.startswith('.')])
list_babyfoot = sorted([f for f in os.listdir(path_babyfoot) if not f.startswith('.')])

path_log = "./TrainLog/"


chunkcheckpts = argv.chunk_checkpts
chunkload = argv.chunk_tosave



if argv.continuous_training=='False':
    to_load = None
    continue_training=False
else:
    to_load = path_log + str(chunkcheckpts) + "Chunk/"
    continue_training=True

to_save = path_log + str(chunkload) + "Chunk/"
trainXMFW = Trainer(cycle,L_DMFW,AutoEncoder,(5,30,lookback,lookahead),
                    loss_func,num_iters_base,print_freq)


path_bab = path_babbage+list_babbage[chunkload]
path_baby = path_babyfoot+list_babyfoot[chunkload]
path_jac = path_jacquard+list_jacquard[chunkload]


AE = AutoEncoder(5, 30, lookback, lookahead)
x = torch.rand((batch_size, lookback, 5))
summary(AE, x)


sol_dmfw, values_dmfw_noniid = trainXMFW.train(optimizer=DMFW, L=L_DMFW, eta_coef=eta_coef_DMFW, eta_exp=eta_exp_DMFW,
                                               reg_coef=reg_coef_DMFW,
                                               oracle_type='FTPL',radius=10,
                                               lookback=lookback, lookahead=lookahead,
                                               batch_size=batch_size,lag=lag, smooth_=True,
                                               valid_cutoff=1, yfeat="temperature",
                                               path_to_data1=path_bab,
                                               path_to_data2=path_baby,
                                               path_to_data3=path_jac,
                                               path_checkpts=to_load,
                                               path_to_save=to_save,
                                               continue_training=continue_training)



