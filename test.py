import torch
import pandas as pd
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from main import Classifier, CFGDataset


edges = pd.read_csv('./cfg_edges.csv')
properties = pd.read_csv('./cfg_label.csv')

print(edges.head())
print(properties.head())


#load dataset
dataset = CFGDataset()
graph, label = dataset[0]
print(graph, label)
dataloader = GraphDataLoader(
    dataset,
    batch_size=1,
    drop_last=False,
    shuffle=True)


#init & load model
model = torch.load('model_weights.pt')
# model.eval()

#predicted list from model
test_X, test_Y = map(list, zip(*dataloader))
pred = []
for g in test_X:
    g = dgl.add_self_loop(g)
    probs_g = torch.softmax(model(g), 1)
    sampled_g = torch.multinomial(probs_g, 1)
    pred.append(sampled_g.item())

print('Graph label: ', test_Y)
print('Predicted label: ', pred)






