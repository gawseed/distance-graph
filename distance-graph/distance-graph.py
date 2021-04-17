from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
import sys
import pyfsdb
import pandas as pd
import os
import json
import random
import collections
import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import Levenshtein
import itertools
import re

TOKENIZE_PATTERN = re.compile('[\s"=;|&><]')
_NIX_COMMANDS = None

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__,
                            epilog='Example Usage: python distance-graph.py -n command -l source --templatize -e 0.945 -E myedgelist.fsdb -c myclusterlist.fsdb -w 12.5 -H 8.5 -fs 8 -ns 250 "../example/commands.fsdb" distance-graph.png')

    parser.add_argument("-n", "--node-name", default=None, type=str,
                        required=True, help="The column name of node")

    parser.add_argument("-l", "--label-name", default=None, type=str,
                        help="The column name of the node's label")

    parser.add_argument("-i", "--id-name", default=None, type=str,
                        help="The column name of the node's identifier")

    parser.add_argument("--templatize", default=False, action="store_true",
                        help="Set this argument to templatize the nodes")

    parser.add_argument("-e", "--edge-weight", default=.95, type=float,
                        help="The edge weight threshold to use")

    parser.add_argument("-E", "--edge-list", default=None, type=str,
                        help="Output enumerated edge list to here")

    parser.add_argument("-c", "--cluster-list", default=None, type=str,
                        help="Output enumerated cluster list to here")

    parser.add_argument("-w", "--width", default=12, type=float,
                        help="The width of the plot in inches")

    parser.add_argument("-H", "--height", default=8, type=float,
                        help="The height of the plot in inches")

    parser.add_argument("-fs", "--font-size", default=10, type=int,
                        help="The font size of the node labels") 

    parser.add_argument("-ns", "--node-size", default=300, type=int,
                        help="The node size for each node")

    parser.add_argument("input_file", type=str,
                        nargs='*',
                        help="")

    parser.add_argument("output_file", type=str,
                        nargs=1, help="")

    args = parser.parse_args()

    return args

def get_cmd2template(input_file,node_name,label):
    """ Given data file with commands, return dict with command templates
    Input: input_file (str): name of FSDB file that contains IP and command data
    Output: cmd2template (dict): maps templatizable commands to highest degree template
    """
    cmds = get_commandCounts(input_file,node_name,label)
    cmd_graph = CommandGraph()
    for cmd in tqdm.tqdm(cmds.keys()):
    #  print("==== CMD IS====\n%s" % cmd) 
        cmd_graph.add(CommandNodeEval(cmd))

    cmd_graph.finalize_degrees()
    sorted_by_degree = sorted(cmd_graph.template2degree.items(), key=lambda k: -k[1])
    cmd2template = cmd_graph.cmd_to_template()

    # print("Got templates. Done.")
    return cmd2template

def get_commandCounts(input_file,node_name,label):
    """ Counts number of commands run in the dataset and returns dict with command and respective counts
    Input: input_file (str): FSDB file with IP and command data
    Output: cmdCount (dict): maps command to number of times the cmd appears in the data
    """
    db = pyfsdb.Fsdb(input_file)

    node_index = db.get_column_number(node_name)
    label_index = db.get_column_number(label)

    cmdCount = {}

    for row in db:
        node = row[node_index]
        label = row[label_index]
        
        if node[0] != "[":
            node = str([node])
        
        if node not in cmdCount:
            cmdCount[node] = 1
        else:
            cmdCount[node] += 1

    return cmdCount

def get_info(input_file,node_name,label,id_name,cmd2template):
    """ Return four dictionaries: (1) weights between commands, (2) IPs that ran commands, (3) sources for each command, and (4) command to array style string
    Input: input_file (str) - FSDB file with IP and command data, template_file (str) - JSON file with templatized commands
    Output: weightDic (dict) - key: pair of commands (tuple) / value: weight (float), cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list),
    sourceDic (dict) - key: command (str) / value: source label (str), cmdToArray (dict) - key: command (str) / value: array style command (str)
    """
    db = pyfsdb.Fsdb(input_file)
    df = db.get_pandas(data_has_comment_chars=True)

    df[node_name] = df[node_name].apply(lambda x: str([x]) if x[0]!="[" else x)

    if id_name:
        loggedInOnly = get_loggedInOnly(df,node_name,label,id_name)

        df2 = df.copy()[~df[id_name].isin(loggedInOnly)]
    df2 = df2[df2[node_name]!='[]']
        cmds = list(df2[node_name].unique())

        cmdIPsDic = get_cmdIPsDic(input_file,loggedInOnly,node_name,label,id_name)
    else:
        df2 = df.copy()
        df2 = df2[df2[node_name]!='[]']
    cmds = list(df2[node_name].unique())
        labelDic = get_labelDic(input_file,node_name,label)
        cmdIPsDic = None
    
    if cmd2template:
        templates = get_templates(cmd2template)
        unique_cmds,cmdIPsDic = get_uniqueCmds(cmds,cmdIPsDic,templates)
    else:
        unique_cmds = cmds

    cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
    unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

    distDic = get_distances(unique_cmds)
    weightDic = get_weights(distDic)

    if cmdIPsDic:
    sourceDic = {cmd:"+".join(list(cmdIPsDic[cmdToArray[cmd]].keys()))+"_"+node_name for cmd in unique_cmds}
    else:
        sourceDic = {cmd:"+".join(labelDic[cmdToArray[cmd]])+"_"+node_name for cmd in unique_cmds}

    return weightDic,cmdIPsDic,sourceDic,cmdToArray

def get_loggedInOnly(df,node_name,label,id_name):
    """ Returns list of IP addresses that only logged in and did not run any commands 
    Input: df (Pandas DataFrame) - dataframe with info on IPs, commands
    Output: loggedInOnly (list) - IPs that only logged in
    """
    loggedIn = df[df[node_name]=='[]'][id_name].unique()
    loggedInOnly = []
    labels = df[label].unique()

    for ip in loggedIn:
        for label_name in labels:
            cmdsRun = list(df[(df[id_name]==ip) & (df[label]==label_name)][node_name].unique())
            if cmdsRun == ['[]']:
            loggedInOnly.append(ip)

    return loggedInOnly

def get_cmdIPsDic(input_file,loggedInOnly,node_name,label):
    """ Returns dict that contains IP addresses that ran the command and from what source
    Input: input_file (str) - FSDB input file, loggedInOnly (list) - list of IPs that only logged in
    Output: cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list)
    """
    cmdIPsDic = {}

    db = pyfsdb.Fsdb(input_file)

    ip_index = db.get_column_number("ip")
    node_index = db.get_column_number(node_name)
    label_index = db.get_column_number(label)

    for row in db:
        ip = row[ip_index]
        
        if ip in loggedInOnly: ## if IP only logged in, do not record
            continue
        
        node = row[node_index]
        label = row[label_index]
        
        if node[0]!="[":
            node = str([node])
        
        if node not in cmdIPsDic:
            cmdIPsDic[node] = {label: [ip]}
        else:
            if label in cmdIPsDic[node]:
                if ip not in cmdIPsDic[node][label]:
                    cmdIPsDic[node][label].append(ip)
            else:
                cmdIPsDic[node][label] = [ip]

    return cmdIPsDic

def get_templates(cmd2template):
    """ Gets list of commands that belong to each template and returns a dictionary
    Input: cmd2template (dict) - key: command (str) / value: template (tuple)
    Output: template2cmd (dict) - key: template (tuple) / value: commands that belong to the template (list)
    """
    template2cmd = {}
    for cmd,basename in cmd2template.items():
        template = basename[2]
        if template not in template2cmd:
            template2cmd[template] = [cmd]
        else:
            template2cmd[template] = template2cmd[template] + [cmd]

    return template2cmd

def get_uniqueCmds(cmds,cmdIPsDic,template2cmd):
    """ Returns list of unique commands, dict that maps command to a dict that has source and IPs that ran command
    Input: cmds (list) - list of commands,
    cmdIPsDic (dict) - dict that maps command to a dictionary that has source and IPs that ran the command
    Output: unique_cmds (list) - list of unique commands, cmdIPsDic (dict) - maps command to a dict that has source and IPs that ran the command
    """
    unique_cmds = cmds
    cmdTemplateDic = {}

    for template,cmds in template2cmd.items():
        first_cmd = cmds[0]
        cmds = cmds[1:]
        
        if first_cmd not in cmdIPsDic:
            first_cmd = cmds[0]
            cmds = cmds[1:]

        cmdTemplateDic[first_cmd] = cmds
        unique_cmds = [x for x in unique_cmds if x not in cmds]

    for cmd_key in cmdTemplateDic:
        if cmd_key not in cmdIPsDic:
            template_cmds = cmdTemplateDic[cmd_key]
            first_cmd = template_cmds[0]
            template_cmds = template_cmds[1:]
        else:
            for cmd in cmdTemplateDic[cmd_key]:
                if cmd not in cmdIPsDic:
                    cmdTemplateDic[cmd_key].remove(cmd)

    cmdIPsDic = update_cmdIPsDic(cmdIPsDic,cmdTemplateDic)
    
    return unique_cmds,cmdIPsDic

def update_cmdIPsDic(cmdIPsDic,cmdTemplateDic):
    """ Returns updated cmdIPsDic dict so templatized command IPs include all IPs that ran cmd with templatized cmd
    Input: cmdIPsDic (dict) - maps command to a dictionary that has source and IPs that ran the command,
    cmdTemplateDic (dict) - maps first command of template to list of all other commands of same template
    Output: (dict) maps command to dict that contains source and IP addresses
    """
    template_cmdIPsDic = {}
    for cmd in cmdTemplateDic: ## for every template
        haasIPs = []
        cowrieIPs = []
        
        if "haas" in cmdIPsDic[cmd]:
            haasIPs = cmdIPsDic[cmd]["haas"]
        if "cowrie" in cmdIPsDic[cmd]:
            cowrieIPs = cmdIPsDic[cmd]["cowrie"]
        
        haasIPs = [haasIPs] + [cmdIPsDic[cmds]["haas"] for cmds in cmdTemplateDic[cmd] if "haas" in cmdIPsDic[cmds]]
        haasIPs = [ip for lst in haasIPs for ip in lst]
        
        cowrieIPs = [cowrieIPs] + [cmdIPsDic[cmds]["cowrie"] for cmds in cmdTemplateDic[cmd] if "cowrie" in cmdIPsDic[cmds]]
        cowrieIPs = [ip for lst in cowrieIPs for ip in lst]
        
        if haasIPs == []:
            dic = {"cowrie":cowrieIPs}
        elif cowrieIPs == []:
            dic = {"haas":haasIPs}
        else:
            dic = {"haas":haasIPs,"cowrie":cowrieIPs}
            
        template_cmdIPsDic[cmd] = dic
    
    cmdIPs = cmdIPsDic.copy()

    for cmd in template_cmdIPsDic:
        cmdIPs[cmd] = template_cmdIPsDic[cmd]
        for cmds in cmdTemplateDic[cmd]:
            if cmds in cmdIPs:
                del cmdIPs[cmds]
    
    return cmdIPs


def get_distances(cmds):
    """ Returns dict that maps every pair of commands with their calculated distance
    Input: cmds (list) - list of commands
    Output: distDic (dict) - key: pair of commands (tuple) / value: distance between commands (float)
    """
    cmdCombos = list(itertools.combinations(cmds,2))

    distDic = {}

    for combo in cmdCombos:
        cmd1, cmd2 = combo
        length = len(cmd1)+len(cmd2)
        distDic[combo] = Levenshtein.distance(cmd1, cmd2)/length

    return distDic

def get_weights(distDic):
    """ Returns dict with weights of every pair of commands. Weights are inversely proportional to distance
    Input: distDic (dict) - dict with distances of every pair of commands
    Output: weightDic (dict) - key: pair of commands (tuple) / value: weight between commands (float)
    """
    distances = sorted(list(set(distDic.values())))

    weights = {}
    maxWeight = max(distances)

    for i in range(len(distances)):
        if i==0:
            weights[distances[i]] = maxWeight
            weight = maxWeight
        else:
            diff = distances[i]-distances[i-1]
            weights[distances[i]] = weight-diff
            weight = weight-diff
    
    weightDic = {}
    for pair,dist in distDic.items():
        weightDic[pair] = weights[dist]

    return weightDic

def draw_networkx(args,weightDic,cmdIPsDic,sourceDic,cmdToArray):
    """ Finds the weighted edges and plots the NetworkX graph
    Input: threshold (float) - weight threshold for weighted edges, output_file (str) - filename for network graph,
    weightDic (dict) - weights for each pair of commands, cmdIPsDic (dict) - maps command to a dictionary that has source and IPs that ran the command,
    sourceDic (dict) - maps command to source label, cmdToArray (dict) - maps command to array style command
    Output: labeled_G (NetworkX graph) - labeled graph with cmd and IP nodes, weighted_edges (list) - list of tuples containing command pair and weight,
    labels (dict) - maps command node to integer
    """
    threshold = args.edge_weight
    output_file = args.output_file[0]
    figsize = tuple([args.width,args.height])

    edgeweight = [tuple(list(k)+[v]) for k,v in weightDic.items()]

    weighted_edges = [x for x in edgeweight if x[2] > threshold]

    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    clusters = get_clusters(G)
    add_IPnodes(G,cmdToArray,cmdIPsDic)
    
    nodeTypeDic,colorslist = set_nodeColors(G,sourceDic)
    labels = get_numberNodes(G,sourceDic)
    plot_networkx(G,output_file,labels,colorslist,nodeTypeDic,figsize=figsize,font_size=args.font_size,node_size=args.node_size)

    return G,weighted_edges,labels,clusters

def add_IPnodes(G,cmdToArray,cmdIPsDic):
    """  Adds IP edges to command nodes
    Input: G (NetworkX graph) - graph to add IP edges to, cmdToArray (dict) - maps command to array style command
    cmdIPsDic (dict) - maps command to IPs that ran command
    """
    nodes = list(G.nodes())

    for node in nodes:
        cmd = cmdToArray[node]
        ips = get_IPs(cmd,cmdIPsDic)
        edges = [(node,ip) for ip in ips]
        G.add_edges_from(edges)

def get_IPs(cmd,dic):
    """ Finds and returns all IP addressses that ran a command
    Input: cmd (str) - command in array style, dic (dict) - dict with commands mapped to source IPs
    Output: (list) unique IPs that ran command
    """
    ips = []
    if "cowrie" in dic[cmd]:
        ips = ips + dic[cmd]["cowrie"]
    if "haas" in dic[cmd]:
        ips = ips + dic[cmd]["haas"]
    
    return list(set(ips))

def set_nodeColors(G,sourceDic):
    """ Sets source as an attribute for each command node, returns nodeTypeDic and color list
    Input: G (NetworkX graph) - graph with command nodes, sourceDic (dict) - maps command nodes to source label
    Output: nodeTypeDic (dict) - maps node type with nodes, colorslist (list) - list of color for node types
    """
    nx.set_node_attributes(G,name="source",values=sourceDic)
    sources = set(nx.get_node_attributes(G,"source").values())

    mapping = dict(zip(sorted(sources),itertools.count()))
    mapping["ip"]=3

    types=list(mapping.keys())
    colorslist = ["b","c","r","y"]

    nodeTypeDic = get_nodeTypeDic(types,G.nodes(),sourceDic)
    return nodeTypeDic,colorslist

def get_nodeTypeDic(types,nodes,sourceDic):
    """ Maps node label to list of nodes and returns dict
    Input: types (list) - list of node types/labels, nodes (NetworkX nodes) - nodes of NetworkX graph, 
    sourceDic (dict) - maps command nodes to source label
    Output: nodeTypeDic (dict) - maps node type to list of nodes with that node type
    """
    nodeTypeDic = {nodetype:[] for nodetype in types}
    
    for node in nodes:
        if node not in sourceDic:
            nodeTypeDic["ip"] = nodeTypeDic["ip"]+[node]
        else:
            source = sourceDic[node]
            nodeTypeDic[source] = nodeTypeDic[source]+[node]
            
    return nodeTypeDic

def get_numberNodes(G,sourceDic):
    """ Returns dict that has nodes mapped to a unique integer label
    Input: G (NetworkX graph) - graph with IP and command nodes, sourceDic (dict) - maps command nodes to source label
    Output: labeled_G (NetworkX graph) - , labels (dict) - maps node to labeled number
    """
    nodes = G.nodes()
    nodeToNum = {}

    i=0
    for node in nodes:
        nodeToNum[node] = i
        i += 1
        
    numToNode = {v:k for k,v in nodeToNum.items()}
    labeled_G = G

    labels={}
    for node in labeled_G.nodes():
        if node in sourceDic:
            labels[node] = nodeToNum[node]
    
    return labels

def plot_networkx(G,output_file,labels,colorslist,nodeTypeDic,figsize=(12,8),font_size=10,node_size=350,ip_alpha=0.2,cmd_alpha=0.2,edge_alpha=0.2):
    """ Plots NetworkX graph and saves image to output file
    Input: G (NetworkX graph) - graph with IP and command nodes to graph, output_file (str) - filename for network graph image
    labels (dict) - maps node to integer label, colorslist (list) - list of node colors, nodeTypeDic (dict) - maps node type to list of nodes
    """  
    fig,ax = plt.subplots(1,figsize=figsize)

    pos=nx.spring_layout(G)
    i=0
    for nodetype in nodeTypeDic:
        nodelist = nodeTypeDic[nodetype]
        color = colorslist[i]
        i+=1

        if nodetype=="ip":
            alpha=ip_alpha
        else:
            alpha=cmd_alpha

        nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                               label=nodetype,alpha=alpha,node_size=node_size,node_color=color)

    nx.draw_networkx_edges(G,pos=pos,alpha=edge_alpha)
    nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=font_size)
    ax.legend(scatterpoints=1)

    ## remove black border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_file, dpi=300)

def get_clusters(G):
    """ Finds clusters of commands in NetworkX graph. A cluster is considered to be nodes that are connected by an edge
    Input: G (NetworkX graph) - graph to find clusters
    Output: cmdToCluster (dict) - key: command node (str) / value: cluster ID (int)
    """
    ip_regex = r'^\d+\.\d+\.\d+\.\d+$'
    components = [list(comp) for comp in list(nx.connected_components(G))]
    
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
        
    return cmdToCluster

## command templatizer code 
def is_bash_directive(s):
  """Retruns True if s is bash syntax (e.g., fi), standard command (e.g., wget), or looks like an argument (e.g., -c).
  """
  if s.startswith('-'): return True
  global _NIX_COMMANDS
  if _NIX_COMMANDS is None:
    _NIX_COMMANDS = set(open('/nfs/lander/working/erinszet/extra/nix_commands.txt').read().split('\n'))
  return s in _NIX_COMMANDS


class CommandNode:
  """Command node is a graph on its own.

  cmd_arr is an array of commands run
  """
  def __init__(self, cmd_arr):
    self.basename2tok_ids = collections.defaultdict(list)
    self.cmd_str = str(cmd_arr)
    all_tokens = []
    for cmd_entry in cmd_arr:
      sub_tokens = TOKENIZE_PATTERN.split(cmd_entry)
      sub_tokens = [t for t in sub_tokens if t]
      all_tokens += sub_tokens
    all_tokens = tuple(all_tokens)
    self.all_tokens = all_tokens

    # Make sets out of path parts.
    for i, tok in enumerate(all_tokens):
      if is_bash_directive(tok):
        continue
      basename = os.path.basename(tok)
      self.basename2tok_ids[basename].append(i)


class CommandNodeEval(CommandNode): ## declare that CommandNodeEval class inherits from CommandNode
  """Command array string gets tokenized."""
  def __init__(self, encoded_command_str):
    super().__init__(eval(encoded_command_str))


class CommandNodeJson(CommandNode):
  """Input object is a json encoded array to convert to a command array""" 
  def __init__(self, json_command_str):
    super().__init__(json.loads(json_command_str))


def _basename_template(basename, command_node):
  token_ids = set(command_node.basename2tok_ids[basename])
  template_tokens = []
  for tid, token in enumerate(command_node.all_tokens):
    if tid in token_ids:
      template_tokens.append(os.path.join(os.path.dirname(token), '%0'))
    else:
      template_tokens.append(token)

  template = ('basename', '%0', tuple(template_tokens))
  return template

def possible_templates(command_node):
  # Basename
  for basename in command_node.basename2tok_ids.keys():
    # templetize basename
    yield _basename_template(basename, command_node)


class CommandGraph:

  def __init__(self):
    self.command_nodes = []
    self.template2commands = collections.defaultdict(list)  # Edges template -> command nodes
  
  def add(self, cn):
    # Step 1: create command node. It creates sets of basenames.
    self.command_nodes.append(cn)
    
    for template in possible_templates(cn):
      self.template2commands[template].append(cn)

  def finalize_degrees(self):
    self.template2degree = {t: len(cmds) for t, cmds in self.template2commands.items()}

  def write_debug_sharded_json(self, output_prefix, entries_per_shard=1000, num_examples_per_template=100):
    import json
    import random
    """Writes template to sample of commands."""
    dirname = os.path.dirname(output_prefix)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self.finalize_degrees()
    sorted_by_degree = sorted(self.template2degree.items(), key=lambda k: -k[1])

    next_file_id = 0
    entries = []
    for i, (template, degree) in enumerate(sorted_by_degree):
      if degree <= 1:
        continue
      # Shallow copy
      commands = [x for x in self.template2commands[template]]
      random.shuffle(commands)
      entries.append({
        'template': template,
        'num_commands': len(commands),
        'commands': [c.cmd_str for c in commands[:entries_per_shard]],
      })
      if len(entries) >= entries_per_shard:
        # Write
        with open(output_prefix + ('_%i.json' % next_file_id), 'w') as fout:
          fout.write(json.dumps(entries))
        entries = []
        next_file_id += 1

    if len(entries) >= 1:
      # Write
      with open(output_prefix + ('_%i.json' % next_file_id), 'w') as fout:
        fout.write(json.dumps(entries))
    print('wrote %i jsons with prefix %s' % (next_file_id, output_prefix))

  def cmd_to_template(self):
    self.finalize_degrees()
    processed_cmds = set()
    sorted_by_degree = sorted(self.template2degree.items(), key=lambda k: -k[1])
    command2template = {}
    for i, (template, degree) in enumerate(sorted_by_degree):
      if degree <= 1: continue
      for cmd in self.template2commands[template]:
        if cmd in processed_cmds:
          continue
        command2template[cmd.cmd_str] = template 
        processed_cmds.add(cmd)
    #
    return command2template

  def write_cmd_to_template(self, output_file=None):
    command2template = self.cmd_to_template()
    if output_file is None:
      output_file = FLAGS.output
    with open(output_file, 'wb') as fout:
      pickle.dump(command2template, fout)
    print('wrote ' + output_file)


def main():
    args = parse_args()
    node_name = args.node_name
    label = args.label

    if args.templatize:
        cmd2template = get_cmd2template(args.input_file[0],node_name,label)
    else:
        cmd2template = None

    weightDic,cmdIPsDic,sourceDic,cmdToArray = get_info(args.input_file[0],node_name,label,cmd2template)
    G,weighted_edges,labels,clusters = draw_networkx(args,weightDic,cmdIPsDic,sourceDic,cmdToArray)

    ## create edge list to FSDB file
    if (args.edge_list):
        outh = pyfsdb.Fsdb(out_file=args.edge_list)
        outh.out_column_names=['cluster_id', 'node1_id', 'node2_id', 'node1', 'node2', 'weight']
        for cmd1,cmd2,weight in weighted_edges:
            cluster_id = clusters[cmd1]
            num1 = labels[cmd1]
            num2 = labels[cmd2]
            outh.append([cluster_id,num1,num2,cmd1,cmd2,round(weight,3)])
        outh.close()

    ## create cluster list to FSDB file
    if (args.cluster_list):
        outh = pyfsdb.Fsdb(out_file=args.cluster_list)
        outh.out_column_names=['cluster_id','command']
        for cmd,cluster_id in clusters.items():
            outh.append([cluster_id,cmd])
        outh.close()

if __name__ == "__main__":
    main()