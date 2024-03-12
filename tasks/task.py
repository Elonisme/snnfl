import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset, Subset

from attacks.blended_attack import poison_data_with_blended
from attacks.semantic_attack import poison_data_with_semantic
from attacks.sig_attack import poison_data_with_sig
from attacks.trigger_attack import poison_data_with_trigger


class PoisonDataset(Dataset):
    def __init__(self, dataset, dataset_name, attack_function):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.attack_function = attack_function

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        data_sample = self.dataset[idx]
        image, label = data_sample
        if self.attack_function == 'trigger':
            image, label = poison_data_with_trigger(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'semantic' and label == 5:
            image, label = poison_data_with_semantic(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'blended':
            image = poison_data_with_blended(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'sig':
            image = poison_data_with_sig(image=image, dataset_name=self.dataset_name)
        else:
            raise SystemExit("No gain attack function")
        return image, label


class PoisonTrainDataset(Dataset):
    def __init__(self, dataset, dataset_name, attack_method):
        self.attack_function = attack_method
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        dataset = self.dataset
        image, _ = dataset[idx]
        if self.attack_function == 'trigger':
            image, label = poison_data_with_trigger(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'semantic' and label == 5:
            image, label = poison_data_with_semantic(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'blended':
            image = poison_data_with_blended(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'sig':
            image = poison_data_with_sig(image=image, dataset_name=self.dataset_name)
        else:
            raise SystemExit("No gain attack function")
        return image, label


class UserDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        dataset = self.dataset
        image, label = dataset[idx]
        return image, label


def sample_dirichlet_train_data(train_dataset, no_participants, alpha=0.9):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: dataset_classes, a preprocessed class-indices dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as
        parameters for
        dirichlet distribution to sample number of images in each class.
    """

    dataset_classes = {}
    for ind, x in enumerate(train_dataset):
        _, label = x
        if label in dataset_classes:
            dataset_classes[label].append(ind)
        else:
            dataset_classes[label] = [ind]
    class_size = len(dataset_classes[0])
    per_participant_list = defaultdict(list)
    no_classes = len(dataset_classes.keys())

    for n in range(no_classes):
        random.shuffle(dataset_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = dataset_classes[n][
                           :min(len(dataset_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            dataset_classes[n] = dataset_classes[n][
                                 min(len(dataset_classes[n]), no_imgs):]

    return per_participant_list


def get_train_data_subset_no_iid(args, indices, train_dataset):
    """
    This method is used along with Dirichlet distribution
    :param args:
    :param train_dataset:
    :param indices:
    :return:
    """

    data_subset = Subset(train_dataset, indices)
    return data_subset


def get_train_data_subset_iid(args, all_range, user_id, train_dataset):
    """
    This method equally splits the dataset.
    :param train_dataset:
    :param args:
    :param all_range:
    :param user_id:
    :return:
    """

    data_len = int(len(train_dataset) / args["num_users"])
    sub_indices = all_range[user_id * data_len: (user_id + 1) * data_len]

    data_subset = Subset(train_dataset, sub_indices)
    return data_subset
