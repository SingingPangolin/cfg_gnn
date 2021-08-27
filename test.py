import torch
import pandas as pd
import dgl
from dgl.data import DGLDataset
from sklearn.metrics import confusion_matrix
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
# model = torch.load('model_weights.pt')
model = Classifier(1, 256, 2)
model.load_state_dict(torch.load('model_para.pt'))
model.eval()

#predicted list from model
test_X, test_Y = map(list, zip(*dataloader))
pred = []
for g in test_X:
    g = dgl.add_self_loop(g)
    probs_g = torch.softmax(model(g), 1)
    sampled_g = torch.multinomial(probs_g, 1)
    argmax_Y = torch.max(probs_g, 1)[1].view(-1, 1)
    pred.append(argmax_Y.item())
    # print(argmax_Y)

true = [i.tolist()[0] for i in test_Y]
print(confusion_matrix(true,pred))


# print('Graph label: ', test_Y)
# print('Predicted label: ', pred)








