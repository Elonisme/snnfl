import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import trange

from defenses.fed_avg import federated_averaging
from defenses.flame import flame
from defenses.median import median
from defenses.snn_fltrust import fltrust
from models.spikingnet import SpikingNet
from tasks.mnist_task import load_mnist_data_subsets
from tasks.task import PoisonTrainDataset, UserDataset, PoisonDataset
from test import fl_test
from tools.plot_experimental_results import initialize_summary_writer, plot_line_chart
from user_side import UserSide


def load_dataset(args):
    if args['dataset'] == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
        train_dataset = datasets.MNIST(
            'data/mnist/',
            train=True,
            download=True,
            transform=trans_mnist)
        test_dataset = datasets.MNIST(
            'data/mnist/',
            train=False,
            download=True,
            transform=trans_mnist)
    elif args['dataset'] == 'cifar10':
        trans_cifar10 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=True,
            download=True,
            transform=trans_cifar10)
        test_dataset = datasets.CIFAR10(
            'data/cifar10',
            train=False,
            download=True,
            transform=test_cifar10)
    elif args['dataset'] == 'cifar100':
        trans_cifar100 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        train_dataset = datasets.CIFAR100(
            'data/cifar100',
            train=True,
            download=True,
            transform=trans_cifar100)
        test_dataset = datasets.CIFAR100(
            'data/cifar100',
            train=False,
            download=True,
            transform=trans_cifar100)
    elif args['dataset'] == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = ImageFolder(
            'data/tiny-imagenet/train',
            transform=train_transform)
        test_dataset = ImageFolder(
            'data/tiny-imagenet/test',
            transform=train_transform)
    else:
        raise SystemExit('Error: unrecognized dataset')

    return train_dataset, test_dataset


def build_model(args, device):
    """Build a global model for training."""
    if args['model'] == 'spiking_net' and args['dataset'] == 'mnist':
        glob_model = SpikingNet().to(device)
    else:
        raise SystemExit('Error: unrecognized model')
    return glob_model


def define_train_data_subsets(args, train_dataset):
    """Build a dateset subset loader for training."""
    if args['model'] == 'spiking_net' and args['dataset'] == 'mnist':
        train_data_subsets = load_mnist_data_subsets(
            args=args,
            train_dataset=train_dataset)
    else:
        raise SystemExit('Error: loading dataset')

    # make root dataset for fltrust
    root_train_dataset = []
    if args['aggregate_function'] in ['fltrust', 'small_fltrust', 'rc_median']:
        root_train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False)
        all_indices = list(range(len(train_dataset)))
        random_indices = random.sample(all_indices, 100)

        for idx, (image, label) in enumerate(root_train_loader):
            if idx in random_indices:
                root_train_dataset.append((image, label))

    return train_data_subsets, root_train_dataset


def federated_learning_train(
        args,
        train_data_subsets,
        device,
        model,
        global_weight,
        chosen_malicious_user_list,
        chosen_normal_user_list,
        all_user_model_weight_list,
        root_train_dateset=None,
        epoch=None):
    sum_loss = 0
    model.train()
    for chosen_user_id in chosen_normal_user_list:
        if args['verbose']:
            print("user %d join in train" % chosen_user_id)
        if chosen_user_id not in chosen_malicious_user_list:
            client_dataset = UserDataset(train_data_subsets[chosen_user_id])
            client_dataloader = DataLoader(
                client_dataset,
                batch_size=args["local_bs"],
                drop_last=True,
                shuffle=True)
            client_device = UserSide(
                args=args,
                train_dataset_loader=client_dataloader
            )
            user_model = copy.deepcopy(model).to(device)
            user_model_weight, user_loss = client_device.train(
                model=user_model)
        else:
            if args['verbose']:
                print("user %d is malicious user" % chosen_user_id)
            client_dataset = PoisonTrainDataset(
                train_data_subsets[chosen_user_id],
                args["dataset"],
                args["attack_method"])
            client_dataloader = DataLoader(
                client_dataset,
                batch_size=args["local_bs"],
                drop_last=True,
                shuffle=True)
            client_device = UserSide(
                args=args,
                train_dataset_loader=client_dataloader
            )
            user_model = copy.deepcopy(model).to(device)
            user_model_weight, user_loss = client_device.train(
                model=user_model)
        if args['all_clients']:
            all_user_model_weight_list[chosen_user_id] = copy.deepcopy(
                user_model_weight)
        else:
            all_user_model_weight_list.append(copy.deepcopy(user_model_weight))
        sum_loss += user_loss

    loss_avg = sum_loss / len(chosen_normal_user_list)

    # aggregate models
    """Define the aggregate function for training."""
    if args['aggregate_function'] == 'layer_defense':
        temp_weight = partial_layer_aggregation(w=all_user_model_weight_list)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'fed_avg':
        temp_weight = federated_averaging(
            w_list=all_user_model_weight_list,
            global_weight=global_weight)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'small_flame':
        temp_weight = small_flame(
            model_list=all_user_model_weight_list,
            global_model=global_weight,
            device=device,
            calculate_time=False
        )
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'flame':
        temp_weight = flame(
            model_list=all_user_model_weight_list,
            global_model=global_weight,
            device=device,
            calculate_time=False
        )
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'median':
        temp_weight = median(model_list=all_user_model_weight_list)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'flame_median':
        temp_weight = hdbscan_median(model_list=all_user_model_weight_list,
                                     global_model=global_weight,
                                     calculate_time=False,
                                     device=device)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'trimmed_mean':
        temp_weight = trimmed_mean(model_list=all_user_model_weight_list)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'fltrust':
        temp_weight = fltrust(model_weights_list=all_user_model_weight_list,
                              global_model_weights=global_weight,
                              root_train_dataset=root_train_dateset,
                              device=device,
                              args=args)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'small_fltrust':
        temp_weight = small_fltrust(
            model_weights_list=all_user_model_weight_list,
            global_model_weights=global_weight,
            root_train_dataset=root_train_dateset,
            device=device,
            args=args,
            flr=epoch)
        return temp_weight, loss_avg
    elif args['aggregate_function'] == 'rc_median':
        temp_weight = trust_median(model_weights_list=all_user_model_weight_list,
                                   global_model_weights=global_weight,
                                   root_train_dataset=root_train_dateset,
                                   device=device,
                                   args=args,
                                   flr=epoch)
        return temp_weight, loss_avg
    else:
        raise SystemExit("error aggregate function!")


def federated_learning(args):
    # parse args
    device = torch.device(
        "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
    if args['verbose']:
        print(f"This Lab is on the {device}")

    # load dataset and split users
    train_dataset, test_dataset = load_dataset(args)

    # build poisonous dataset
    use_poisonous_test_dataset = copy.deepcopy(test_dataset)
    poisonous_test_dataset = PoisonDataset(
        use_poisonous_test_dataset, args['dataset'], args['attack_method'])

    train_data_subsets, root_train_dataset = define_train_data_subsets(
        args=args, train_dataset=train_dataset)

    if args['verbose']:
        print("Malicious data generated successfully.")

    # calculate normal user number
    normal_users_number = max(int(args['frac'] * args['num_users']), 1)

    # build malicious user list
    malicious_users_number = max(
        int(args['malicious_user_rate'] * args['num_users']), 0)
    chosen_malicious_user_list = np.random.choice(
        range(args['num_users']), malicious_users_number, replace=False)

    # build model
    model = build_model(args, device)
    # copy weights
    global_weight = model.state_dict()
    # define 进度条样式
    bar_style = "{l_bar}{bar}{r_bar}"

    writer = initialize_summary_writer()
    for epoch in trange(
            args['epochs'],
            desc="Federated Learning Training",
            unit="epoch",
            bar_format=bar_style):
        all_user_model_weight_list = []
        chosen_normal_user_list = np.random.choice(
            range(args['num_users']), normal_users_number, replace=False)

        if args['all_clients']:
            print("Aggregation over all clients")
            all_user_model_weight_list = [
                global_weight for i in range(
                    args['num_users'])]

        # federated learning train
        temp_weight, loss_avg = federated_learning_train(args=args,
                                                         train_data_subsets=train_data_subsets,
                                                         device=device,
                                                         model=model,
                                                         global_weight=global_weight,
                                                         chosen_malicious_user_list=chosen_malicious_user_list,
                                                         chosen_normal_user_list=chosen_normal_user_list,
                                                         all_user_model_weight_list=all_user_model_weight_list,
                                                         root_train_dateset=root_train_dataset,
                                                         epoch=epoch)

        # Calculation accuracy
        ma, ba = fl_test(model=model,
                         temp_weight=temp_weight,
                         test_dataset=test_dataset,
                         poisonous_dataset_test=poisonous_test_dataset,
                         device=device,
                         args=args)

        global_weight = copy.deepcopy(temp_weight)

        plot_line_chart(writer, ma, "Main accuracy", epoch)
        plot_line_chart(writer, ba, "Backdoor accuracy", epoch)
        plot_line_chart(writer, loss_avg, "Loss average", epoch)

    # 关闭SummaryWriter
    writer.close()
