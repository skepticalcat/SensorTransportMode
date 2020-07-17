import torch
import sys
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from SensorDataset import SensorDataset
from TestValidate import TestValidate
from models import CNNwithLSTM

if len(sys.argv) < 4:
    print("Usage: python TransModeDetect.py name_of_tensor_pickle cnn_window_size_sec lstm_window_size_sec")

net = CNNwithLSTM()
net.cuda()
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # , eps=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ds = SensorDataset(sys.argv[1])
datasets = ds.train_test_dataset(0.2, 0.1)

print("{}, {}, {} Train, Test and Val Samples".format(len(datasets['train']),
                                                      len(datasets['test']),
                                                      len(datasets['val'])))

batch_size = 8
losses = []
running_loss = 0
val_counter = 0
epochs = 30
trip_dim = int(sys.argv[3]) // int(sys.argv[2])


testval = TestValidate(device,batch_size,criterion,net,trip_dim)
dataloaders = {x: DataLoader(datasets[x], batch_size, shuffle=True, num_workers=0, drop_last=True) for x in
               ['train', 'test', "val"]}

for epoch in range(epochs):
    for i, data in enumerate(dataloaders["train"], 0):
        val_counter+=1

        inputs, labels = data["frame"], data["labels"]
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        labels = labels.view(batch_size * trip_dim, -1).squeeze(1).long().to(device)
        loss = criterion(outputs.view(batch_size * trip_dim, -1), labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            losses.append(running_loss / 10)
            print('[%d, %5d] loss: %.3f' % (epoch, i, losses[-1]))
            running_loss = 0.0
        if val_counter % 500 == 0:
            testval.test(dataloaders["val"])

testval.test(dataloaders["test"])
fig = plt.figure()
ax = plt.axes()


ax.plot(losses)
plt.show()