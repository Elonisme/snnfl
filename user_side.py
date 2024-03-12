import snntorch.functional as SF
import torch


class UserSide(object):
    def __init__(
            self,
            args,
            train_dataset_loader
    ):
        self.train_dataset_loader = train_dataset_loader
        self.verbose = args['verbose']
        self.num_epochs = args['local_ep']
        self.learning_rate = args['lr']
        self.momentum_rate = args['momentum']
        self.weight_decay = args['weight_decay']
        self.device = torch.device(
            "cuda" if args['gpu'] and torch.cuda.is_available() else "cpu")
        self.batch_size = args['local_bs']

    def train(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
        loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

        num_epochs = 1  # run for 1 epoch - each data sample is seen only once

        loss_hist = []  # record loss over iterations
        acc_hist = []  # record accuracy over iterations

        # training loop
        for epoch in range(num_epochs):
            for i, (data, targets) in enumerate(iter(self.train_dataset_loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)

                model.train()
                spk_rec, _ = model(data)  # forward-pass
                loss_val = loss_fn(spk_rec, targets)  # loss calculation
                optimizer.zero_grad()  # null gradients
                loss_val.backward()  # calculate gradients
                optimizer.step()  # update weights
                loss_hist.append(loss_val.item())  # store loss

                # print every 25 iterations
                if i % 25 == 0:
                    model.eval()
                    print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

                    # check accuracy on a single batch
                    acc = SF.accuracy_rate(spk_rec, targets)
                    acc_hist.append(acc)
                    print(f"Accuracy: {acc * 100:.2f}%\n")

        return model.state_dict(), sum(loss_hist) / num_epochs
