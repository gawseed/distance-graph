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
        self.build_graph(args,data)
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

    def build_graph(self, args, data):
        self.G.add_weighted_edges_from(sorted(self.weighted_edges))
        self.find_clusters()

        if args.id_name != '':
            self.add_IP_nodes(data)
        
        self.assign_node_colors(data,args)

    def add_IP_nodes(self, data):
        nodes = list(self.G.nodes())

        for node in nodes:
            cmd = data.cmdToArray[node]
            ips = self.get_IPs(cmd,data.cmdIPsDic)
            edges = [(node,ip) for ip in ips]
            self.G.add_edges_from(edges)

    def find_IPs(cmd,dic):
        ips = []

        for label,label_ips in dic[cmd].items():
            ips = ips + label_ips
        
        return list(set(ips))
    
    def assign_node_colors(self,data,args):
        nx.set_node_attributes(self.G,name="source",values=data.sourceDic)
        sources = set(sorted(nx.get_node_attributes(self.G,"source").values()))

        if args.id_name != '':
            id_labels = [label for label in sources if args.id_name in label]
        else:
            id_labels = []

        source_labels = [label for label in sources if label not in id_labels]
        types = sorted(source_labels) + sorted(id_labels)

        self.colorslist = self.get_colors(types)
        self.nodeTypeDic = self.create_nodeTypeDic(types,data.sourceDic)

    def get_colors(self,types):
        sourceToColor = {}
        colors_to_use = []
        colorslist = ["b","c","r","tab:orange","y","lime","tab:pink","g","tab:brown","tab:purple"]

        for type in types:
            source = type.split('_')[0]

            if source not in sourceToColor:
                if 'new' in source: ## make new nodes red color
                    if "r" in colorslist:
                        sourceToColor[source] = 'r'
                        colorslist.pop(colorslist.index('r'))
                    else:
                        sourceToColor[source] = 'lime'
                        colorslist.pop(colorslist.index('lime'))
                else:
                    sourceToColor[source] = colorslist[0]
                    colorslist = colorslist[1:]
            
            colors_to_use.append(sourceToColor[source])
        
        return colors_to_use
    
    def create_nodeTypeDic(self,types,sourceDic):
        nodeTypeDic = {nodetype:[] for nodetype in types}
        
        for node in self.G.nodes:
            source = sourceDic[node]
            nodeTypeDic[source] = nodeTypeDic[source]+[node]
                
        return nodeTypeDic

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

    def find_clusters(self):
        ip_regex = r'^\d+\.\d+\.\d+\.\d+$'
        components = sorted([sorted(list(comp)) for comp in list(nx.connected_components(self.G))])
        
        clusters = {}
        i=0
        for comp in components:
            cluster = [node for node in comp if not re.search(ip_regex,node)]
            clusters[i] = cluster
            i+=1

        cmdToCluster = {}
        for cluster,commands in clusters.items():
            ids = {cmd:cluster for cmd in commands}
            cmdToCluster.update(ids)
        
        self.clusters = cmdToCluster
    
    def plot_networkx(self,args,templates):
        node_size = args.args.node_size
        font_size = 10
        ip_alpha = 0.2
        cmd_alpha = 0.2
        edge_alpha = 0.2

        fig,ax = plt.subplots(1,figsize=self.figsize)
        if not self.pos:
            self.pos=nx.spring_layout(self.G)
        else:
            self.pos=pickle.load(open(self.pos,"rb"))
            fixed_nodes = self.pos.keys()
            self.pos=nx.spring_layout(self.G,pos=self.pos,fixed=fixed_nodes,k=0.3)

        i=0
        handles = []
        for nodetype in self.nodeTypeDic:
            nodelist = self.nodeTypeDic[nodetype]
            color = self.colorslist[i]
            i+=1

            if args.id_name and args.id_name in nodetype:
                alpha=ip_alpha
                nx.draw_networkx_nodes(self.G,pos=self.pos,nodelist=nodelist,ax=ax,\
                            label=nodetype,alpha=alpha,node_size=node_size,node_shape="^",node_color=color)
            else:
                if 'new' in nodetype:
                    alpha=0.4
                else:
                    alpha=cmd_alpha
                    alpha = [alpha if templates.cmd2template[node] not in templates.old_templates else 0.085 for node in nodelist]
                
                if templates.cmd2template_count != {}:
                    node_sizes = [templates.cmd2template_count[node] for node in nodelist]
                    node_size_template = [int((5*node)**0.5) for node in node_sizes]

                    points = nx.draw_networkx_nodes(self.G,pos=self.pos,nodelist=nodelist,ax=ax,\
                            label=nodetype,alpha=alpha,node_size=node_size_template,node_color=color)
                    handles.append(points.legend_elements("sizes", num=4))
                else:
                    nx.draw_networkx_nodes(self.G,pos=self.pos,nodelist=nodelist,ax=ax,\
                                label=nodetype,alpha=alpha,node_size=node_size,node_color=color)

        nx.draw_networkx_edges(self.G,pos=self.pos,alpha=edge_alpha)
        nx.draw_networkx_labels(self.G,pos=self.pos,labels=self.labels,font_size=font_size)

        legend = ax.legend(scatterpoints=1, markerscale=0.75)
        for leg in legend.legendHandles:
            leg.set_alpha(0.5)
            leg._sizes = [250]
        plt.gca().add_artist(legend)

        if handles != []:
            legend_handles, legend_labels = self.get_size_legend(handles)
            legend2 = ax.legend(handles=legend_handles,labels=legend_labels,bbox_to_anchor = (1,1.15), title='command count')
            for leg in legend2.legendHandles:
                leg.set_alpha(0.3)

        ## remove black border
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.savefig(args.output_file, dpi=300)

    def get_size_legend(self,handles):
        handle_points = []
        handle_labels = []
        regex = r'(\d+)'
        
        for handle in handles:
            handle_points += handle[0]
            handle_labels += handle[1]

        handle_labels = [int(re.search(regex, handle).group(1)) for handle in handle_labels]
        all_handles = sorted([(handle_points[i],handle_labels[i]) for i in range(len(handle_points))], key=lambda x: x[1])
        to_keep = [min(handle_labels), max(handle_labels), all_handles[int(len(all_handles)/2)][1]]
        all_handles = [handle for handle in all_handles if handle[1] in to_keep]

        legend_handles = []
        legend_labels = []
        for handle in all_handles:
            if handle[1] not in legend_labels:
                legend_handles.append(handle[0])
                legend_labels.append(handle[1])
        
        legend_labels = [int((label**2)/5) for label in legend_labels]
        legend_labels = [f'{label:,}' for label in legend_labels]
        return legend_handles, legend_labels