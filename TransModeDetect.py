import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from SensorDataset import SensorDataset
from TestValidate import TestValidate
from models import CNNwithLSTM

net = CNNwithLSTM()
net.cuda()
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # , eps=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8

ds = SensorDataset("lstm_examples_25_3_10")
datasets = ds.train_test_dataset(0.2, 0.1)
print(len(datasets['train']))
print(len(datasets['test']))
print(len(datasets['val']))

dataloaders = {x: DataLoader(datasets[x], batch_size, shuffle=True, num_workers=0, drop_last=True) for x in
               ['train', 'test', "val"]}
losses = []
val_acc = []
running_loss_logger = 0
running_loss = 0

testval = TestValidate(device,batch_size,criterion,net)

r = 0
for epoch in range(30):

    for i, data in enumerate(dataloaders["train"]):
        r+=1
        i+=1
        inputs, labels = data["frame"], data["labels"]
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        labels = labels.view(batch_size * (600 // 3), -1).squeeze(1).long().to(device)
        loss = criterion(outputs.view(batch_size * (600 // 3), -1), labels)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        running_loss_logger += loss.item()
        if i % 10 == 0:
            losses.append(running_loss_logger / 10)
            running_loss_logger = 0
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        if r % 500 == 0:  # print every 2000 mini-batches
            testval.test(dataloaders["val"])

testval.test(dataloaders["test"])
#Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8