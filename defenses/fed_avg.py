import copy

import torch


def federated_averaging(w_list, global_weight, eta=5, n=100):
        w_avg = copy.deepcopy(w_list[0])
        for k in w_avg.keys():
            for i in range(1, len(w_list)):
                w_avg[k] += w_list[i][k]
            w_avg[k] = torch.mul(w_avg[k], eta/n) + torch.mul(global_weight[k], 1 - eta * len(w_list)/n)
        return w_avg
