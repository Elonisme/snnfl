import logging
from collections import OrderedDict

import torch
from plato.config import Config


def bulyan(updates, baseline_weights, weights_attacked):
    """Aggregate weight updates from the clients using bulyan."""

    total_clients = Config().clients.total_clients
    num_attackers = len(Config().clients.attacker_ids)  # ?

    remaining_weights = flatten_weights(weights_attacked)
    bulyan_cluster = []

    # Search for bulyan cluster based on distance
    while (len(bulyan_cluster) < (total_clients - 2 * num_attackers)) and (
        len(bulyan_cluster) < (total_clients - 2 - num_attackers)
    ):
        distances = []
        for weight in remaining_weights:
            distance = torch.norm((remaining_weights - weight), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(
            distances[:, : len(remaining_weights) - 2 - num_attackers], dim=1
        )
        indices = torch.argsort(scores)[
            : len(remaining_weights) - 2 - num_attackers
        ]

        # Add candidate into bulyan cluster
        bulyan_cluster = (
            remaining_weights[indices[0]][None, :]
            if not len(bulyan_cluster)
            else torch.cat(
                (bulyan_cluster, remaining_weights[indices[0]][None, :]), 0
            )
        )

        # Remove candidates from remainings
        remaining_weights = torch.cat(
            (
                remaining_weights[: indices[0]],
                remaining_weights[indices[0] + 1 :],
            ),
            0,
        )

    # Perform sorting
    n, d = bulyan_cluster.shape
    median_weights = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - median_weights), dim=0)
    sorted_weights = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # Average over sorted bulyan cluster
    mean_weights = torch.mean(sorted_weights[: n - 2 * num_attackers], dim=0)

    # Update global model
    start_index = 0
    bulyan_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        bulyan_update[name] = mean_weights[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Bulyan server aggregation.")
    return bulyan_update

    def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights
