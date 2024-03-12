import copy

import snntorch.functional as SF
import torch

from models.spikingnet import SpikingNet


def build_model(args, device):
    """Build a global model for training."""
    if args['model'] == 'spiking_net' and args['dataset'] == 'mnist':
        glob_model = SpikingNet().to(device)
    else:
        raise SystemExit('Error: unrecognized model')
    return glob_model


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.values()])


def train(model, data_loader, device, optimizer, loss_fn):
    for i, (data, targets) in enumerate(iter(data_loader)):
        data = data.to(device)
        targets = targets.to(device)
        model.train()

        spk_rec, _ = model(data)  # forward-pass
        loss_val = loss_fn(spk_rec, targets)  # loss calculation
        optimizer.zero_grad()  # null gradients
        loss_val.backward()  # calculate gradients
        optimizer.step()  # update weights


def fltrust(model_weights_list, global_model_weights, root_train_dataset, device, args):
    root_net = build_model(args, device)
    root_net.load_state_dict(global_model_weights)
    net_num = len(model_weights_list)

    global_model = copy.deepcopy(global_model_weights)
    optimizer = torch.optim.Adam(root_net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    # training a root net using root dataset
    for _ in range(0, 3):  # server side local training epoch could be adjusted
        train(model=root_net, data_loader=root_train_dataset, device=device, optimizer=optimizer, loss_fn=loss_fn)

    root_net.eval()  # 冻结参数
    root_update = copy.deepcopy(global_model_weights)
    root_net_weight = root_net.state_dict()  # 获取根服务器模型参数

    # get  root_update
    whole_aggregator = []
    for p_index, p in enumerate(global_model):
        params_aggregator = root_net_weight[p] - global_model[p]
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(root_update):
        root_update[p] = whole_aggregator[param_index]

    # get user nets updates
    for i in range(net_num):
        whole_aggregator = []
        user_model_weights = model_weights_list[i]
        for p_index, p in enumerate(global_model):
            params_aggregator = user_model_weights[p] - global_model[p]
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(user_model_weights):
            user_model_weights[p] = whole_aggregator[param_index]

    # compute Trust Score for all users
    root_update_vec = vectorize_net(root_update)
    TS = []
    net_vec_list = []
    for i in range(net_num):
        user_model_weights = model_weights_list[i]
        net_vec = vectorize_net(user_model_weights)
        net_vec_list.append(net_vec)
        cos_sim = torch.cosine_similarity(net_vec, root_update_vec, dim=0)
        ts = torch.relu(cos_sim)
        TS.append(ts)
    if torch.sum(torch.Tensor(TS)) == 0:
        return global_model

    # get the regularized users' updates by aligning with root update
    norm_list = []
    for i in range(net_num):
        norm = torch.norm(root_update_vec) / torch.norm(net_vec_list[i])
        norm_list.append(norm)

    for i in range(net_num):
        whole_aggregator = []
        user_model_weights = model_weights_list[i]

        for p_index, p in enumerate(global_model):
            params_aggregator = norm_list[i] * user_model_weights[p]
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(user_model_weights):
            user_model_weights[p] = whole_aggregator[param_index]

    # aggregation: get global update
    whole_aggregator = []
    global_update = copy.deepcopy(global_model_weights)

    zero_model_weights = model_weights_list[0]
    for p_index, p in enumerate(zero_model_weights):
        params_aggregator = torch.zeros(zero_model_weights[p].size()).to(device)
        for net_index, net in enumerate(model_weights_list):
            params_aggregator = params_aggregator + TS[net_index] * net[p]
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(global_update):
        global_update[p] = (1 / torch.sum(torch.tensor(TS))) * whole_aggregator[param_index]

    # get global model
    final_global_model = copy.deepcopy(global_model_weights)
    for i in range(net_num):
        whole_aggregator = []
        for p_index, p in enumerate(global_model):
            params_aggregator = global_update[p] + global_model[p]
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(final_global_model):
            final_global_model[p] = whole_aggregator[param_index]

    return final_global_model
