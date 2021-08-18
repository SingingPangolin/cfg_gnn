import networkx as nx
import os
import json
import matplotlib.pyplot as plt
import dgl
import csv

#return list of non repeate nodes
def non_repeat_list(l):
    res = []
    for i in l:
        if i not in res:
            res.append(i)
    return res

#append newline to csv
def append_csv(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)


malware_path = '/media/trunghieu/Others/SinginP/cfg/malware'
benign_path = '/media/trunghieu/Others/SinginP/cfg/benign'

# create malware dataset
obj = os.scandir(malware_path)
graph_idx = 0
for entry in obj:
    if entry.name.endswith(".json") and entry.is_file():
        print(entry.name)
        cfg_json = os.path.join(malware_path, entry.name)
        li = []
        with open(cfg_json) as json_file:
            try:
                rjson = json.load(json_file)
                bl = rjson[0]['blocks']
            except IndexError:
                continue
            else:
                for block in rjson[0]['blocks']:
                    if 'jump' not in block:
                        continue
                    else:
                        #list for num_nodes
                        li.append(block['offset'])
                        li.append(block['jump'])
                    if 'fail' not in block:
                        continue
                    else:
                        #list for num_nodes
                        li.append(block['fail'])

                graph_li = non_repeat_list(li)
                graph_li.sort()
                for block in rjson[0]['blocks']:
                    if 'jump' not in block:
                        continue
                    else:
                        if li:
                            row = [graph_idx]
                            row.append(graph_li.index(block['offset']))
                            row.append(graph_li.index(block['jump']))
                            append_csv('cfg_edges.csv', row)

                    if 'fail' not in block:
                        continue
                    else:
                        if li:
                            row = [graph_idx]
                            row.append(graph_li.index(block['offset']))
                            row.append(graph_li.index(block['fail']))
                            append_csv('cfg_edges.csv', row)
        try:
            bl = rjson[0]
        except IndexError:
            continue
        else:
            graph_li = non_repeat_list(li)
            if graph_li:
                row = [graph_idx]
                row.append(1)
                row.append(len(graph_li)+1)
                append_csv('cfg_label.csv', row)
                graph_idx += 1


# create benign dataset
obj = os.scandir(benign_path)
for entry in obj:
    if entry.name.endswith(".json") and entry.is_file():
        print(entry.name)
        cfg_json = os.path.join(benign_path, entry.name)
        li = []
        with open(cfg_json) as json_file:
            try:
                rjson = json.load(json_file)
                bl = rjson[0]['blocks']
            except IndexError:
                continue
            else:
                for block in rjson[0]['blocks']:
                    if 'jump' not in block:
                        continue
                    else:
                        #list for num_nodes
                        li.append(block['offset'])
                        li.append(block['jump'])
                    if 'fail' not in block:
                        continue
                    else:
                        #list for num_nodes
                        li.append(block['fail'])

                graph_li = non_repeat_list(li)
                graph_li.sort()
                for block in rjson[0]['blocks']:
                    if 'jump' not in block:
                        continue
                    else:
                        if li:
                            row = [graph_idx]
                            row.append(graph_li.index(block['offset']))
                            row.append(graph_li.index(block['jump']))
                            append_csv('cfg_edges.csv', row)

                    if 'fail' not in block:
                        continue
                    else:
                        if li:
                            row = [graph_idx]
                            row.append(graph_li.index(block['offset']))
                            row.append(graph_li.index(block['fail']))
                            append_csv('cfg_edges.csv', row)

        try:
            bl = rjson[0]
        except IndexError:
            continue
        else:
            if graph_li:
                row = [graph_idx]
                row.append(0)
                row.append(len(graph_li)+1)
                append_csv('cfg_label.csv', row)
                graph_idx += 1

# G = nx.Graph()



# json_test = os.path.join(malware_path, "1d7c7c7b2bf5a5be38ac5ac332d6b02d.json")
# li = []
# print(json_test)
# with open(json_test) as json_file:
#     rjson = json.load(json_file)
#     print(rjson[0]['name'])
#     for block in rjson[0]['blocks']:
#         li.append(block['offset'])
#         li.append(block['jump'])
#         if 'fail' not in block:
#             continue
#         else:
#             li.append(block['fail'])


# print(li)
# graph_li = non_repeat_list(li)
# print(graph_li)

#draw graph with networkx
# G.add_nodes_from(graph_li)
# print(G.number_of_nodes())
# with open(json_test) as json_file:
#     rjson = json.load(json_file)
#     print(rjson[0]['name'])
#     for block in rjson[0]['blocks']:
#         G.add_edge(block['offset'], block['jump'])
#         if 'fail' not in block:
#             continue
#         else:
#             G.add_edge(block['offset'], block['fail'])
# print(G.number_of_edges())

# dgl_graph = dgl.from_networkx(G)
# print(dgl_graph)
# nx.draw(G, with_labels = True)
# plt.savefig("filename.png")



#create dlg graph
# src_idx = []
# dst_idx = []
# with open(json_test) as json_file:
#     rjson = json.load(json_file)
#     print(rjson[0]['name'])
#     for block in rjson[0]['blocks']:
#         src_idx.append(block['offset'])
#         dst_idx.append(block['jump'])
#         if 'fail' not in block:
#             continue
#         else:
#             src_idx.append(block['offset'])
#             dst_idx.append(block['fail']

# in_edges = torch.tensor(src_idx)
# out_edges = torch.tensor(dst_idx)
# dgl_graph = dgl.graph((in_edges, out_edges))
# list_graph = dgl.graph((src_idx, dst_idx), num_nodes = len(graph_li))
# print(dgl_graph)
# print(list_graph)