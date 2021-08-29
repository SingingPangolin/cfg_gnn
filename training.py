import dgl
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from utils import CFGDataset, Classifier

import os
import pandas as pd


edges = pd.read_csv('./cfg_edges.csv')
properties = pd.read_csv('./cfg_label.csv')

print(edges.head())
print(properties.head())

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)



dataset = CFGDataset()
graph, label = dataset[0]
dataloader = GraphDataLoader(
    dataset,
    batch_size=1,
    drop_last=False,
    shuffle=True, collate_fn=collate)


num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False)



model = Classifier(1, 256, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(50):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(dataloader):
        bg = dgl.add_self_loop(bg)
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)


torch.save(model.state_dict(), 'model_para.pt')
torch.save(model, 'model_weights.pt')
model.eval()

# Calculate accuracy
test_X, test_Y = map(list, zip(*dataloader))
pred = []
for g in test_X:
    g = dgl.add_self_loop(g)
    probs_g = torch.softmax(model(g), 1)
    sampled_g = torch.multinomial(probs_g, 1)
    pred.append(sampled_g.item())

count = 0
for i in range(len(pred)):
    if pred[i] == test_Y[i]:
        count += 1


print('Accuracy: ', 100*count/len(test_Y))






# probs_Y = torch.softmax(model(test_bg), 1)
# print(probs_Y)
# sampled_Y = torch.multinomial(probs_Y, 1)
# print(sampled_Y.size())
# print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
#     (test_Y == sampled_Y.int()).sum().item() / len(test_Y) * 100))


# num_correct = 0
# num_tests = 0
# for batched_graph, labels in test_dataloader:
#     pred = model(batched_graph,7)
#     num_correct += (pred.argmax(1) == labels).sum().item()
#     num_tests += len(labels)

# print('Test accuracy:', num_correct / num_tests)