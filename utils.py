import os
import numpy as np
import torch
import random


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def assignWeight(self, model, state):
    with torch.no_grad():
        for (name, param), ele in zip(model.named_parameters(), state):
            param.copy_(nn.Parameter(ele))
    return model


def customLossFull(weight, y):
    data_size = weight.shape[0]
    tmp_exp = torch.exp(weight)
    tmp_num = torch.zeros(data_size)
    for i in range(data_size):
        tmp_num[i] = tmp_exp[i, y[i]]
    loss = -torch.mean(torch.log(tmp_num / torch.sum(tmp_exp, 1)))
    return loss


def customLoss(weight, y):
    logsoft = F.log_softmax(weight, dim=1)
    loss = -torch.mean(logsoft)
    return loss


def LinearGradient(weight, X, y):
    data_size = X.shape[0]
    W1 = weight[0]
    b1 = weight[1]
    W2 = weight[2]
    b2 = weight[3]

    grad = [0.] * len(weight)
    U = torch.matmul(X, W1.T) + b1
    sigmoid = 1. / (1. + torch.exp(-U))
    P = torch.matmul(sigmoid, W2.T) + b2
    tmp_exp = torch.exp(P)
    tmp_denom = torch.sum(tmp_exp, 0)
    tmp_exp = tmp_exp / tmp_denom
    for i in range(data_size):
        tmp_exp[i, y[i]] = tmp_exp[i, y[i]] - 1

    d_P = tmp_exp
    d_W2 = torch.matmul(d_P.T, sigmoid) / data_size
    d_b2 = torch.mean(d_P, 0)
    d_U = torch.multiply(torch.matmul(d_P, W2),
                         torch.multiply(sigmoid, (1 - sigmoid)))
    d_W1 = torch.matmul(d_U.T / data_size, X)
    d_b1 = torch.mean(d_U, 0)
    grad[0] = d_W1
    grad[1] = d_b1
    grad[2] = d_W2
    grad[3] = d_b2
    return grad


def lmo_fn(G, radius):
    res = [0.] * len(G)
    for k in range(len(G)):
        shape = G[k].shape
        if len(shape) == 4:
            myelem = torch.zeros(shape)
            for chan_out in range(shape[0]):
                for chan_in in range(shape[1]):
                    tmpG = G[k][chan_out][chan_in]
                    num_rows, num_cols = tmpG.shape
                    cols = torch.argmax(torch.abs(tmpG), 1)
                    rows = torch.arange(num_rows)
                    indices = torch.LongTensor([rows.tolist(), cols.tolist()])
                    flatten_G = tmpG.T.flatten()[rows + cols * num_rows]
                    values = -radius * torch.sign(flatten_G)
                    res_tmp = torch.sparse.FloatTensor(indices, values,
                                                       tmpG.shape).to_dense()
                    myelem[chan_out][chan_in] = res_tmp
            res[k] = myelem

        elif len(shape) == 3:
            myelem = torch.zeros(shape)
            for chan_out in range(shape[0]):
                tmpG = G[k][chan_out]
                num_rows, num_cols = tmpG.shape
                cols = torch.argmax(torch.abs(tmpG), 1)
                rows = torch.arange(num_rows)
                indices = torch.LongTensor([rows.tolist(), cols.tolist()])
                flatten_G = tmpG.T.flatten()[rows + cols * num_rows]
                values = -radius * torch.sign(flatten_G)
                res_tmp = torch.sparse.FloatTensor(indices, values,
                                                   tmpG.shape).to_dense()
                myelem[chan_out] = res_tmp

            res[k] = myelem

        elif len(shape) == 2:
            tmpG = G[k]
            num_rows, num_cols = tmpG.shape
            cols = torch.argmax(torch.abs(tmpG), 1)
            rows = torch.arange(num_rows)
            indices = torch.LongTensor([rows.tolist(), cols.tolist()])
            flatten_G = tmpG.T.flatten()[rows + cols * num_rows]
            values = -radius * torch.sign(flatten_G)
            res_tmp = torch.sparse.FloatTensor(indices, values,
                                               tmpG.shape).to_dense()
            res[k] = res_tmp

        elif len(shape) == 1:
            tmpG = G[k]
            nb_rows = tmpG.shape
            rows = torch.argmax(torch.abs(tmpG))
            s = torch.zeros(nb_rows)
            s[rows] = -radius * torch.sign(tmpG[rows])
            res[k] = s

    return res


def gap_fn(grad, weight, radius):
    V = lmo_fn(grad, radius)
    fw_gap = 0
    for k in range(len(V)):
        fw_gap += torch.sum(torch.mul(grad[k], (weight[k] - V[k])))
    return fw_gap


def online_gap(grad_sum, inner_sum, radius):
    V = lmo_fn(grad_sum, radius)
    compute1 = 0
    for k in range(len(V)):
        compute1 += torch.sum(torch.mul(grad_sum[k], V[k]))
    ol_gap = inner_sum - compute1
    return ol_gap


def inner_prod(tensor1, tensor2):
    return torch.sum(torch.mul(tensor1, tensor2))


def gradient(loss, weight, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(loss)

    grad = torch.autograd.grad(loss,
                               weight,
                               grad_outputs=grad_outputs,
                               create_graph=True)
    return grad


def ModelPrediction(model_path, state_dict_path, loader):
    model = torch.load(model_path)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    corr_pred = 0.
    data_len = 0
    for data, label in loader:
        data_len += data.shape[0]
        output = model(data)
        _, pred = torch.max(output, 1)
        corr_pred += torch.sum(pred == label)
    return corr_pred / data_len
