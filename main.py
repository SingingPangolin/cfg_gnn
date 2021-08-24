import dgl
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

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

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')     ### Readout Function
        return self.classify(hg)

#generate dgl dataset from csv file
class CFGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='cfg')

    def process(self):
        edges = pd.read_csv('./cfg_edges.csv')
        properties = pd.read_csv('./cfg_label.csv')
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)




dataset = CFGDataset()
graph, label = dataset[0]
print(graph, label)
dataloader = GraphDataLoader(
    dataset,
    batch_size=1,
    drop_last=False,
    shuffle=True)


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