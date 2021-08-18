import pandas as pd
import urllib.request
import csv


# urllib.request.urlretrieve(
#     'https://data.dgl.ai/tutorial/dataset/graph_edges.csv', './graph_edges.csv')
# urllib.request.urlretrieve(
#     'https://data.dgl.ai/tutorial/dataset/graph_properties.csv', './graph_properties.csv')
# edges = pd.read_csv('./graph_edges.csv')
# properties = pd.read_csv('./graph_properties.csv')

# edges.head()

# properties.head()

def append_csv(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)


edges = ['graph_id','src','dst']
label = ['graph_id','label','num_nodes']
with open('cfg_edges.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(edges)

with open('cfg_label.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(label)

# append_csv('countries.csv', header)
