import torch
import numpy as np

class TestValidate:

    def __init__(self, device, batch_size, criterion, net, trip_dim):
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion
        self.net = net
        self.trip_dim = trip_dim

    def test(self, dataloader):
        def score_to_modality(scores: torch.Tensor):
            tensor_list = scores.tolist()
            modality = []
            for row in tensor_list:
                modality.append(row.index(max(row)))
            return modality

        with torch.no_grad():
            t_correct = 0
            t_total = 0
            total_per_mode = [0] * 9
            correct_per_mode = [0] * 9

            val_losses = []
            for i, data in enumerate(dataloader):
                inputs, labels = data["frame"], data["labels"]
                inputs = inputs.to(self.device)

                outputs = self.net(inputs)
                labels = labels.view(self.batch_size * self.trip_dim, -1).squeeze(1).long().to(self.device)
                loss = self.criterion(outputs.view(self.batch_size * self.trip_dim, -1), labels)
                val_losses.append(loss.item())
                predicted = score_to_modality(outputs.view(self.batch_size * self.trip_dim, -1))

                for o, elem in enumerate(predicted):
                    total_per_mode[int(labels[o])] += 1
                    if labels[o] == predicted[o]:
                        correct_per_mode[predicted[o]] += 1
                        t_correct += 1
                    t_total += 1

            mode_statistics = []
            for k in range(len(correct_per_mode)):
                if correct_per_mode[k] == 0 or total_per_mode[k] == 0:
                    mode_statistics.append(0)
                    continue
                mode_statistics.append(1 / (total_per_mode[k] / correct_per_mode[k]))

            print('Accuracy: %d %%' % (100 * t_correct / t_total))
            print("Loss: {:.6f}".format(np.mean(val_losses)))
            print("Mode-correct:")
            print(total_per_mode)
            print(mode_statistics)
            print("Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8")

