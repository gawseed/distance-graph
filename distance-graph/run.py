from tools.arguments import *
from tools.templatizer import *
from tools.templates import Templates
from tools.data import Data
from tools.graph import NetworkGraph
import pyfsdb
import pickle


def get_info2(args, cmd2template):
    """ Cleans and transforms input data according to user arguments. Finds weights and labels of nodes.
        Return four dictionaries: (1) weights between commands, (2) IPs that ran commands, (3) sources for each command,
        and (4) command to array style string
    Input:
        file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
        output_names (list) - list of output names to use for node, label, and identifier
        cmd2template (dict) - maps templatizable commands to highest degree template
        args (class) - ArgumentParser class with input arguments
    Output:
        weightDic (dict) - key: pair of commands (tuple) / value: weight (float)
        cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list)
        sourceDic (dict) - key: command (str) / value: source label (str)
        cmdToArray (dict) - key: command (str) / value: array style command (str)
        cmd2template (dict) - key: command (str) / value: template (tuple)
    """
    template_nodes = args.args.template_nodes
    templates_class = cmd2template
    temporal = args.args.temporal
    templates = {}
    cmd2templateCount = {}

    data = Data(args)
    login_index = False

    if args.id_name != '':
        cmdIPsDic = data.labelDic
        sourceDic = data.sourceDic
    else:
        labelDic = data.labelDic
        cmdIPsDic = None

    # got_unique_cmds = False

    if cmd2template:
        data.find_unique_templatized_cmds(args, cmd2template, temporal)
        unique_cmds = data.unique_cmds
        # cmdIPsDic = None
        labelDic = data.labelDic
        templates = cmd2template.template2cmd
        old_templates = cmd2template.old_templates

        # templates = cmd2template.templates
        cmd2template = cmd2template.cmd2template
        # if cmdIPsDic:
        #     unique_cmds,cmdIPsDic,templates,old_templates = get_uniqueCmds(data.unique_cmds,cmdIPsDic,{},templates,temporal)
        # else:
        #     unique_cmds,labelDic,templates,old_templates = get_uniqueCmds(data.unique_cmds,cmdIPsDic,labelDic,templates,temporal)

        if template_nodes:
            # unique_cmds2 = []
            # for cmd in unique_cmds:
            #     if (cmd in [cmd for lst in templates.values() for cmd in lst]):
            #     # if (cmd in [cmd for lst in templates[0].values() for cmd in lst]) or (cmd in [cmd for lst in templates[1].values() for cmd in lst]):
            #         unique_cmds2.append(cmd)
            
            # unique_cmds = unique_cmds2
            data.get_template_nodes(templates_class, args)
            unique_cmds = data.unique_cmds

            templates_class.calculate_template_counts(data.df, args.node_name, unique_cmds)
            # templateCounts = templates_class.template_counts
            cmd2templateCount = templates_class.cmd2template_count
            
            # templateCounts = calc_templateCount(templates,data.df,args.node_name)
            # cmd2templateCount = map_cmd2templateCount(cmd2template,templateCounts,unique_cmds)

            if args.args.labels:
                labels = pickle.load(open(args.args.labels,"rb"))
                data.update_representative_cmd(labels,templates_class)
            #     labels = pickle.load(open(args.args.labels,"rb"))
            #     unique_cmds,cmd_to_old_label = update_representativeCmd(unique_cmds,labels,cmd2template,templates)
            #     cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
            #     unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

            #     if cmdIPsDic:
            #         cmdIPsDic = remap_dic(cmdIPsDic,cmd_to_old_label)
            #     elif labelDic:
            #         labelDic = remap_dic(labelDic,cmd_to_old_label)

            #     cmd2templateCount = remap_dic(cmd2templateCount,cmd_to_old_label,'cmd')
                cmd2templateCount = data.remap_dic(cmd2templateCount, data.cmd_to_old_label, 'cmd')
            #     got_unique_cmds = True
    
    data.get_unique_cmds()
    # if data.got_unique_cmds == False:
    #     cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
    #     unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

    # cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
    # unique_cmds = [cmd[2:-2] for cmd in unique_cmds]
    
    data.calculate_weights()
    weightDic = data.weightDic
    # distDic = get_distances(unique_cmds)
    # weightDic = get_weights(distDic)

    data.update_source_dic(args)
    sourceDic = data.sourceDic

    # if cmdIPsDic:
    #     sourceDic.update({cmd:"+".join(list(cmdIPsDic[cmdToArray[cmd]].keys()))+"_"+args.node_name for cmd in unique_cmds})
    # else:
    #     sourceDic = {cmd:"+".join(labelDic[cmdToArray[cmd]])+"_"+args.node_name for cmd in unique_cmds}

    return data,weightDic,cmdIPsDic,sourceDic,data.cmdToArray,cmd2template,templates,cmd2templateCount,old_templates,templates_class

def draw_networkx2(args,data,cmdIPsDic,sourceDic,cmdToArray,templates,cmd2templates,cmd2templateCount,old_templates):
    """ Finds the weighted edges and plots the NetworkX graph. Obtains labels for nodes and cluster IDs for connected nodes
    Input:
        args (argument parser) - parser of command line arguments,
        weightDic (dict) - weights for each pair of commands, cmdIPsDic (dict) - maps command to a dictionary that has source and IPs that ran the command,
        cmdIPsDic (dict) - maps command to dict that contains source and IP addresses
        sourceDic (dict) - maps command to source label, cmdToArray (dict) - maps command to array style command,
        cmdToArray (dict) - maps command (str) to array style command (str)
    Output:
        G (NetworkX graph) - labeled graph with weighted edges,
        weighted_edges (list) - list of tuples containing command pair and weight,
        labels (dict) - maps command node to integer,
        clusters (dict) - key: command node (str) / value: cluster ID (int)
    """
    # threshold = args.args.edge_weight
    # pos = args.args.position
    # output_file = args.args.output_file[0]
    # figsize = tuple([args.args.width,args.args.height])
    # node_name, label_name, id_name = get_outputNames(output_names)

    # edgeweight = [tuple(list(k)+[v]) for k,v in weightDic.items()]

    # if args.args.top_k:
    #     k = args.args.top_k
    #     weighted_edges = get_topK_edges(k, edgeweight, k_edges=args.args.top_k_edges)
    # else:
    #     weighted_edges = [x for x in edgeweight if x[2] > threshold]

    # weighted_edges = sorted(weighted_edges)
    networkgraph = NetworkGraph(args,data,cmd2templates)
    #weighted_edges = networkgraph.weighted_edges
    # G = nx.Graph()
    # G.add_weighted_edges_from(sorted(weighted_edges))

    # if (args.args.labels):
    #     labels = pickle.load(open(args.args.labels,"rb"))
    #     labels = add_newLabels(G,labels,cmd2templates)
    # else:
    #     labels = get_numberNodes(G)

    labels = networkgraph.labels
    clusters = networkgraph.clusters
    nodeTypeDic = networkgraph.nodeTypeDic
    colorslist = networkgraph.colorslist

    # clusters = get_clusters(G)

    # if cmdIPsDic:
    #     add_IPnodes(G,cmdToArray,cmdIPsDic)

    # nodeTypeDic,colorslist = set_nodeColors(G,sourceDic,args.id_name)

    G = networkgraph.G
    networkgraph.plot_networkx(args, templates)
    pos = networkgraph.pos

    # if (args.args.temporal):
    #     pos = plot_temporal_networkx(G,networkgraph.pos,networkgraph.output_file,labels,colorslist,nodeTypeDic,args.id_name,cmd2templateCount,cmd2templates,old_templates,figsize=networkgraph.figsize,font_size=args.args.font_size,node_size=args.args.node_size)
    # else:
    #     pos = plot_networkx(G,networkgraph.pos,networkgraph.output_file,labels,colorslist,nodeTypeDic,args.id_name,cmd2templateCount,figsize=networkgraph.figsize,font_size=args.args.font_size,node_size=args.args.node_size)

    return G,networkgraph.weighted_edges,labels,clusters,pos

def main():
    args = Arguments()
    # args = parse_args()

    # ## get list of file args and remove None input files args that were not specified
    # file_args = list(filter(None,[args.input_file1, args.input_file2, args.input_file3, args.input_file4]))
    # output_names = args.output_names

    # check_fileArgs(file_args, output_names)

    if args.args.templatize:
        # temporal = args.temporal
        # stopwords = args.stopwords
        if args.args.stopwords == None:
            raise Exception('Stopwords file is missing. The stopwords file is required for templatization.')

        # global _NIX_COMMANDS
        # if _NIX_COMMANDS is None:
        #     _NIX_COMMANDS = set(open(stopwords).read().split('\n'))

        cmd2template = Templates(args.args.stopwords, args.file_args, args.args.temporal)
        # cmd2template = get_cmd2template(file_args,temporal)
    else:
        cmd2template = None

    data,weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates,cmd2templateCount,old_templates,templates_class = get_info2(args, cmd2template)
    # weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates,cmd2templateCount,old_templates = get_info(file_args, output_names, cmd2template, args)
    # G,weighted_edges,labels,clusters,pos = draw_networkx(args,args.output_names,weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,cmd2templateCount,old_templates)
    G,weighted_edges,labels,clusters,pos = draw_networkx2(args,data,cmdIPsDic,sourceDic,cmdToArray,templates_class,cmd2template,cmd2templateCount,old_templates)

    ## save NetworkX graph position file to pickle file
    if (args.args.position_file):
        pickle.dump(pos, open(args.args.position_file, "wb" ))

    ## save labels dict to pickle file
    if (args.args.labels_file):
        if (args.args.template_nodes):
            output_labels = {cmd:{'label':label, 'template':cmd2template[cmd]} for cmd,label in labels.items()}
            pickle.dump(output_labels, open(args.args.labels_file, "wb"))
        else:
            output_labels = {cmd:{'label':label, 'template':"N/A"} for cmd,label in labels.items()}
            # pickle.dump(labels, open(args.labels_file, "wb" ))

    ## create edge list to FSDB file
    if (args.args.edge_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            print("Graphing template nodes...")
            outh = pyfsdb.Fsdb(out_file=args.args.edge_list)
            outh.out_column_names=['cluster_id', 'weight', 'node1_id', 'node2_id', 'node1', 'node2', 'template1', 'template2']
            for cmd1,cmd2,weight in weighted_edges:
                cluster_id = clusters[cmd1]
                num1 = labels[cmd1]
                num2 = labels[cmd2]
                template1 = ' '.join(cmd2template[cmd1])
                template2 = ' '.join(cmd2template[cmd2])
                outh.append([cluster_id,round(weight,3),num1,num2,cmd1,cmd2,template1,template2])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.args.edge_list)
            outh.out_column_names=['cluster_id', 'node1_id', 'node2_id', 'node1', 'node2', 'weight']
            for cmd1,cmd2,weight in weighted_edges:
                cluster_id = clusters[cmd1]
                num1 = labels[cmd1]
                num2 = labels[cmd2]
                outh.append([cluster_id,num1,num2,cmd1,cmd2,round(weight,3)])
            outh.close()

    ## create cluster list to FSDB file
    if (args.args.cluster_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            outh = pyfsdb.Fsdb(out_file=args.args.cluster_list)
            outh.out_column_names=['cluster_id','command','template']
            for cmd,cluster_id in clusters.items():
                template = ' '.join(cmd2template[cmd])
                outh.append([cluster_id,cmd,template])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.cluster_list)
            outh.out_column_names=['cluster_id','command']
            for cmd,cluster_id in clusters.items():
                outh.append([cluster_id,cmd])
            outh.close()

    ## create template list to FSDB file
    if (args.args.template_list):
        if (args.args.template_nodes): ## if template nodes are being graphed, produce template list that contains template, example command, and label
            outh = pyfsdb.Fsdb(out_file=args.args.template_list)
            outh.out_column_names=['template','command','node','label']
            template_list = []
            for cmd in clusters.keys():
                template = ' '.join(cmd2template[cmd])
                node = labels[cmd]
                if sourceDic != {}:
                    label = sourceDic[cmd]
                else:
                    label = ''
                template_list.append([template,cmd,node,label])

            template_list = sorted(template_list)
            for line in template_list:
                outh.append(line)
            outh.close()
        else:
            print("Template nodes were not graphed. No template list to output.")
    
    if (args.args.templatecmd_list):
        if (args.args.template_nodes):
            outh = pyfsdb.Fsdb(out_file=args.args.templatecmd_list)
            outh.out_column_names=['template','command']
            tc_list = []
            for temp,cmds in templates.items():
                template = ' '.join(temp)
                cmds = sorted(set(cmds))
                cmds = [cmd[2:-2] for cmd in cmds]
                for cmd in cmds:
                    tc_list.append([template,cmd])
            tc_list = sorted(tc_list)
            for line in tc_list:
                outh.append(line)
            outh.close()
        else:
            print("Template nodes were not graphed. No template command list to output.")

if __name__ == "__main__":
    main()