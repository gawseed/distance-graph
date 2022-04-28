import networkx as nx
import matplotlib.pyplot as plt
import pickle
import re

class NetworkGraph():
    def __init__(self, args, data, templates):
        self.threshold = args.args.edge_weight
        self.pos = args.args.position
        self.output_file = args.args.output_file[0]
        self.figsize = tuple([args.args.width,args.args.height])
        self.weighted_edges = []
        self.G = nx.Graph()
        self.labels = {}
        self.clusters = {}
        self.nodeTypeDic = {}
        self.colorslist = []

        self.calculate_weighted_edges(args, data)
        self.build_graph()
        self.generate_labels(args,templates)

    def calculate_weighted_edges(self, args, data):
        edgeweight = [tuple(list(k)+[v]) for k,v in data.weightDic.items()]
        if args.args.top_k:
            k = args.args.top_k
            self.weighted_edges = self.get_topK_edges(k, edgeweight, k_edges=args.args.top_k_edges)
        else:
            self.weighted_edges = [x for x in edgeweight if x[2] > self.threshold]
        self.weighted_edges = sorted(self.weighted_edges)

    def get_topK_edges(self, k, edgeweight, k_edges):
        topK = {}
        edgeweight = sorted(edgeweight, key=lambda x: x[2], reverse=True)

        if k_edges: ## get k edges for each node
            for i in range(len(edgeweight)):
                edge = edgeweight[i]
                cmd1,cmd2,weight = edge

                if cmd1 in topK:
                    if len(topK[cmd1]) < k:
                        topK[cmd1] = topK[cmd1] + [edge]
                else:
                    topK[cmd1] = [edge]
                
                if cmd2 in topK:
                    if len(topK[cmd2]) < k:
                        topK[cmd2] = topK[cmd2] + [edge]
                else:
                    topK[cmd2] = [edge]
        else: ## get top k nodes
            for i in range(len(edgeweight)):
                edge = edgeweight[i]
                cmd1,cmd2,weight = edge

                if len(topK) < k:
                    if cmd1 in topK:
                        topK[cmd1] = topK[cmd1] + [edge]
                    else:
                        topK[cmd1] = [edge]
                    
                    if cmd2 in topK:
                        topK[cmd2] = topK[cmd2] + [edge]
                    else:
                        topK[cmd2] = [edge]
                        
        topK_edges = list(set([tups for lst in list(topK.values()) for tups in lst]))
        
        return topK_edges

    def build_graph(self):
        self.G.add_weighted_edges_from(sorted(self.weighted_edges))

    def generate_labels(self, args, templates):
        if (args.args.labels):
            labels = pickle.load(open(args.args.labels,"rb"))
            self.labels = self.add_new_labels_to_existing(labels,templates)
        else:
            self.labels = self.add_number_labels()

    def add_new_labels_to_existing(self,labels,cmd2templates):
        nodes = self.G.nodes()
        num_labels = [labels[cmd]['label'] for cmd in labels]
        template2label = {labels[cmd]['template']:labels[cmd]['label'] for cmd in labels}
        new_labels = {}
        
        i=max(num_labels)+1
        for node in nodes:
            if node in labels:
                new_labels[node] = labels[node]['label']
            else:
                if cmd2templates != {}:
                    template = cmd2templates[node]
                    if template in template2label:
                        label = template2label[template]
                        new_labels[node] = label
                    else:
                        new_labels[node] = i
                    i+=1
                else:
                    new_labels[node] = i
                    i+=1
        return new_labels

    def add_number_labels(self):
        nodes = sorted(self.G.nodes())
        labels = {}

        i=0
        for node in nodes:
            labels[node] = i
            i += 1
        
        return labels