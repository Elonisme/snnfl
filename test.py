import torch
from torch.utils.data import DataLoader
import snntorch.functional as SF


# function to measure accuracy on full test set
def test_accuracy(net, data_loader, device):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


def fl_test(
        model,
        temp_weight,
        test_dataset,
        poisonous_dataset_test,
        device,
        args):
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args['bs'],
        shuffle=False)
    poisonous_dataset_test_loader = DataLoader(
        dataset=poisonous_dataset_test,
        batch_size=args['bs'],
        shuffle=False)

    model.load_state_dict(temp_weight)
    model.eval()

    ma = test_accuracy(model, test_loader, device)
    ba = test_accuracy(model, poisonous_dataset_test_loader, device)

    return ma * 100, ba * 100
