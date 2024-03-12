import torch
import logging

from collections import OrderedDict


def trimmed_mean(model_list):
    """Aggregate weight updates from the clients using trimmed-mean."""
    flattened_weights = flatten_weights(model_list)
    num_attackers = 0  # ?

    n, d = flattened_weights.shape
    median_weights = torch.median(flattened_weights, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_weights - median_weights), dim=0)
    sorted_weights = flattened_weights[sort_idx, torch.arange(d)[None, :]]

    mean_weights = torch.mean(sorted_weights[: n - 2 * num_attackers], dim=0)

    start_index = 0
    trimmed_mean_update = OrderedDict()
    for name, weight_value in model_list[0].items():
        trimmed_mean_update[name] = mean_weights[
                                    start_index: start_index + len(weight_value.view(-1))
                                    ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Trimmed mean server aggregation.")

    return trimmed_mean_update


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
