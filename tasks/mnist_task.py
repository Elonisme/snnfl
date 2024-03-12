import random

from torch.utils.data import Subset

from tasks.task import get_train_data_subset_iid, sample_dirichlet_train_data, get_train_data_subset_no_iid


def load_mnist_data_subsets(args, train_dataset):
    if args["iid"]:
        # sample indices for participants that are equally
        # split to 500 images per participant
        split = min(args["num_users"] / 20, 1)
        all_range = list(range(int(len(train_dataset) * split)))
        train_dataset = Subset(train_dataset, all_range)
        random.shuffle(all_range)
        data_subsets = [get_train_data_subset_iid(args, all_range, pos, train_dataset) for pos in
                        range(args["num_users"])]
    else:
        # sample indices for participants using Dirichlet distribution
        split = min(args["num_users"] / 20, 1)
        all_range = list(range(int(len(train_dataset) * split)))
        train_dataset = Subset(train_dataset, all_range)
        indices_per_participant = sample_dirichlet_train_data(
            train_dataset, args["num_users"], alpha=0.9)
        data_subsets = [get_train_data_subset_no_iid(args, indices, train_dataset) for indices in
                        indices_per_participant.items()]

    random.shuffle(data_subsets)
    return data_subsets
