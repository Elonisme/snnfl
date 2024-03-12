import copy
import torch


def partial_layer_aggregation(w):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        # 如果参数名称中包含 "fc" 或者 "linear"，跳过全连接层的参数
        if "fc" in k or "linear" in k:
            print("model is checked out for linear layer!!!!")
            continue

        for i in range(1, len(w)):
            w_avg[k] += w[i][k]

        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg
