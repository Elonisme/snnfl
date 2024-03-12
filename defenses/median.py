import torch
import logging
from collections import OrderedDict


def median(model_list):
    """Aggregate weight updates from the clients using median."""

    flattened_weights = flatten_weights(model_list)

    median_weight = torch.median(flattened_weights, dim=0)[0]

    # Update global model
    start_index = 0
    median_update = OrderedDict()
    for name, weight_value in model_list[0].items():
        median_update[name] = median_weight[
                              start_index: start_index + len(weight_value.view(-1))
                              ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    logging.info(f"Finished Median server aggregation.")

    return median_update


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
