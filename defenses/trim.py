import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from copy import deepcopy
import time
from sklearn.metrics import roc_auc_score

def trim(old_gradients, param_list, net, lr, b=0, hvp=None):
    '''
    gradients: the list of gradients computed by the worker devices
    net: the global model
    lr: learning rate
    byz: attack
    f: number of compromised worker devices
    b: trim parameter
    '''
    if hvp is not None:
        pred_grad = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    # sort
    sorted_array = nd.array(np.sort(nd.concat(*param_list, dim=1).asnumpy(), axis=-1), ctx=mx.gpu(5))
    #sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    # trim
    n = len(param_list)
    m = n - b * 2
    trim_nd = nd.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=1)

    # update global model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size

    return trim_nd, distance
