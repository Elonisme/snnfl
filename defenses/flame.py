import copy
import time

import torch
import hdbscan
from collections import Counter


def vectorize_net(static_dict):
    return torch.cat([p.view(-1) for p in static_dict.values()])


def flame_defense(model_list, global_model, device):
    fc_avg = copy.deepcopy(global_model)
    cos = []
    cos_ = []

    for fc_model_out in model_list:
        x1 = vectorize_net(fc_model_out) - vectorize_net(global_model)
        for fc_model_in in model_list:
            x2 = vectorize_net(fc_model_in) - vectorize_net(global_model)
            cos.append(torch.cosine_similarity(x1, x2, dim=0).detach().cpu())
        cos_.append(torch.cat([p.view(-1) for p in cos]).reshape(-1, 1))
        cos = []

    cos_ = torch.cat([p.view(1, -1) for p in cos_])

    cluster_er = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster_er.fit_predict(cos_)
    majority = Counter(cluster_labels)
    res = majority.most_common(len(model_list))

    out = []  # 正常模型列表

    for i in range(len(cluster_labels)):  # 筛选出正常模型
        if cluster_labels[i] == res[0][0]:
            out.append(i)

    e = []  # 每个模型和全局模型的欧氏距离
    for i in range(len(model_list)):
        e.append(
            torch.sqrt(
                torch.sum(
                    (vectorize_net(fc_avg) -
                     vectorize_net(model_list[i])) ** 2)))

    e = torch.cat([p.view(-1) for p in e])
    st = torch.median(e)
    whole_aggregator = []
    par = []

    for i in range(len(out)):
        par.append(min(1, st / e[out[i]]))

    wa = []
    for p_index, p in enumerate(fc_avg):
        wa.append(fc_avg[p])

    for p_index, p in enumerate(model_list[0]):
        # initial
        params_aggregator = torch.zeros(model_list[0][p].size()).to(device)

        for i in range(len(out)):
            net = model_list[out[i]]
            params_aggregator = params_aggregator + \
                                wa[p_index] + (net[p] - wa[p_index]) * par[i]

        sum = 0
        for i in range(len(par)):
            sum += 1
        params_aggregator = params_aggregator / sum
        whole_aggregator.append(params_aggregator)
    lamda = 1e-3
    sigma = st * lamda

    for param_index, p in enumerate(fc_avg):
        fc_avg[p] = whole_aggregator[param_index] + \
                    (sigma ** 2) * torch.randn(whole_aggregator[param_index].shape).to(device)

    return fc_avg


def flame(model_list, global_model, device, calculate_time):
    if calculate_time:
        start_time = time.time()
        _ = flame_defense(
            model_list=model_list,
            global_model=global_model,
            device=device)
        end_time = time.time()
        # 计算函数调用的时间
        elapsed_time = end_time - start_time
        print(f"函数调用耗时: {elapsed_time} 秒")
        raise SystemExit("error aggregate function!")
    else:
        temp_weight = flame_defense(
            model_list=model_list,
            global_model=global_model,
            device=device)
        return temp_weight
