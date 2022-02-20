from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
import pyfsdb
import pandas as pd
import os
import json
import collections
import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import Levenshtein
import itertools
import re
import pickle
import math

TOKENIZE_PATTERN = re.compile('[\s"=;|&><]')
_NIX_COMMANDS = None

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__,
                            epilog='Example Usage: python distance-graph.py -1 ../example/data1.fsdb command data1 ip -2 ../example/data2.fsdb commands data2 ip -3 ../example/data3.fsdb cmd data3 id -on command source ip --templatize -stop ../example/nix_commands.txt -e 0.05 -w 10 -H 8 -fs 10 -ns 250 -E edgelist.fsdb -c clusterlist.fsdb distance-graph.png')

    parser.add_argument("-1", "--input_file1", type=str, nargs=4,
                        required=True, help="Pass [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data1), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

    parser.add_argument("-2", "--input_file2", type=str, default=None, nargs=4,
                        help="Pass second [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data2), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

    parser.add_argument("-3", "--input_file3", type=str, default=None, nargs=4,
                        help="Pass third [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data3), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

    parser.add_argument("-4", "--input_file4", type=str, default=None, nargs=4,
                        help="Pass fourth [required] input filename, [required] node column name (e.g. command), label name of the data (e.g. data3), node's identifier column name (e.g. ip). Pass empty string '' if do not have label name or identifier column.")

    parser.add_argument("-on", "--output_names", type=str, nargs=3,
                        required=True, help="The names to use for output distance graph labels, edge list, and cluster list. Pass the [required] node name (e.g. command), column name for label (e.g. source), identifier name (e.g. ip). Pass empty string '' if input files do not have label name or identifier column.")

    parser.add_argument("--templatize", default=False, action="store_true",
                        help="Set this argument to templatize the nodes")

    parser.add_argument("--template-nodes", default=False, action="store_true",
                    help="Set this argument to graph template nodes only")

    parser.add_argument("-stop", "--stopwords", default=None, type=str,
                        help="Path to text file that contains stopwords separated by a new line")

    parser.add_argument("--temporal", type=int, default=None, nargs='+',
                        help="Pass the input file number(s) with 'new' data separated by a space to perform temporal analysis")

    parser.add_argument("-e", "--edge-weight", default=.95, type=float,
                        help="The edge weight threshold to use")

    parser.add_argument("-k", "--top-k", default=None, type=int,
                        help="Top k nodes to graph. If this is set, the edge weight threshold will not be used.")

    parser.add_argument("--top-k-edges", default=False, action="store_true",
                    help="Set this argument to graph top k edges for each node")

    parser.add_argument("-p", "--position", default=None, type=str,
                        help="Path to pickle file that contains NetworkX graph positional dictionary")

    parser.add_argument("-l", "--labels", default=None, type=str,
                        help="Path to pickle file that contains dictionary of node labels")

    parser.add_argument("-E", "--edge-list", default=None, type=str,
                        help="Output enumerated edge list to here")

    parser.add_argument("-c", "--cluster-list", default=None, type=str,
                        help="Output enumerated cluster list to here")

    parser.add_argument("-pf", "--position-file", default=None, type=str,
                        help="Output pickle file that contains generated NetworkX graph positional dictionary to here")

    parser.add_argument("-lf", "--labels-file", default=None, type=str,
                        help="Output pickle file that contains dictionary of node labels to here")

    parser.add_argument("-w", "--width", default=12, type=float,
                        help="The width of the plot in inches")

    parser.add_argument("-H", "--height", default=8, type=float,
                        help="The height of the plot in inches")

    parser.add_argument("-fs", "--font-size", default=10, type=int,
                        help="The font size of the node labels") 

    parser.add_argument("-ns", "--node-size", default=300, type=int,
                        help="The node size for each node")

    parser.add_argument("output_file", type=str, nargs=1, 
                        help="Pass path and filename for output network graph")

    args = parser.parse_args()

    return args


def check_fileArgs(file_args, output_names):
    """ Checks input file arguments with output name arguments to make sure correct arguments are provided. If not, raise exception
    Input:
        file_args (list) - list of lists (each list is a list of arguments for input file)
        output_names (list) - list of names to use for output distance graph and edge and cluster lists
    """
    output_node = output_names[0]
    output_source = output_names[1]
    output_id = output_names[2]

    if output_node == '':
        raise Exception("Missing name for output node")

    for i in range(len(file_args)):
        file_num = i+1
        inputfile_args = file_args[i]

        inputfile = inputfile_args[0]
        input_node = inputfile_args[1]
        input_source = inputfile_args[2]
        input_id = inputfile_args[3]

        if inputfile == '':
            raise Exception("Missing input file for input file {}".format(file_num))
        
        if input_node == '':
            raise Exception("Missing node column name for input file {}".format(file_num))
        
        if (output_source != '') and (input_source == ''):
            raise Exception("Source column name for input file {} not provided, but provided in output names.".format(file_num))
        
        if (output_source == '') and (input_source != ''):
            raise Exception("Source column name for input file {} is provided, but not provided in output names.".format(file_num))

        if (output_id != '') and (input_id == ''):
            raise Exception("Identifier column name for input file {} not provided, but provided in output names.".format(file_num))
        
        if (output_id == '') and (input_id != ''):
            raise Exception("Identifier column name for input file {} is provided, but not provided in output names.".format(file_num))


def get_cmd2template(file_args,temporal):
    """ Given data file with commands, return list of command templates dict. If performing temporal analysis, then get templates for period 1 and period 2
    Input:
        file_args (list) - list of input FSDB files
        temporal (None/list) - list of file numbers (period 2) for temporal analysis
    Output:
        cmd2template (list) - list of two cmd2template dicts. cmd2template - maps templatizable commands to highest degree template
    """
    cmds,cmds2 = get_commandCounts(file_args,temporal)
    cmd2template = templatize_cmds(cmds)

    if temporal:
        print("Finding period 2 templates")
        cmd2template2 = templatize_cmds(cmds2)
    else:
        cmd2template2 = {}

    return [cmd2template,cmd2template2]

def templatize_cmds(cmds):
    cmd_graph = CommandGraph()
    for cmd in tqdm.tqdm(cmds.keys()):
    #  print("==== CMD IS====\n%s" % cmd) 
        cmd_graph.add(CommandNodeEval(cmd))

    cmd_graph.finalize_degrees()
    sorted_by_degree = sorted(cmd_graph.template2degree.items(), key=lambda k: -k[1])
    cmd2template = cmd_graph.cmd_to_template()

    # print("Got templates. Done.")
    return cmd2template

def get_commandCounts(file_args, temporal):
    """ Counts number of commands run in the dataset and returns dict with command and respective counts
    Input:
        file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
    Output:
        cmdCount (dict) - maps command to number of times the cmd appears in the data
    """
    cmdCount = {}
    cmdCount2 = {}

    file_num = 1
    for input_file in file_args:
        filename = input_file[0]
        node_name = input_file[1]

        db = pyfsdb.Fsdb(filename)
        node_index = db.get_column_number(node_name)

        for row in db:
            node = row[node_index]
            
            if node[0] != "[":
                node = str([node])

            if temporal:
                if file_num in temporal: ## if file is in period 2, add nodes to cmdCount2
                    if node not in cmdCount2:
                        cmdCount2[node] = 1
                    else:
                        cmdCount2[node] += 1
                else:
                    if node not in cmdCount:
                        cmdCount[node] = 1
                    else:
                        cmdCount[node] += 1
            else:
                if node not in cmdCount:
                    cmdCount[node] = 1
                else:
                    cmdCount[node] += 1
        
        db.close()
        file_num += 1

    return cmdCount,cmdCount2

def get_inputFile_args(inputfile_args):
    """ Parse each input file arg and return filename, node column name, label column name, and identifier column name
    Input:
        inputfile_args (list) - list of input arguments user provided for input file
    Output:
        filename, node_name, label_name, identifier_name (str) - return input arguments
    """
    filename = inputfile_args[0]
    node_name = inputfile_args[1]
    label_name = inputfile_args[2]
    identifier_name = inputfile_args[3]

    return filename, node_name, label_name, identifier_name

def get_outputNames(output_names):
    """ Parse each output name argument and return node name, label name, and identifier name to be used for final output and graph
    Input:
        output_names (list) - list of output names to use for node, label, and identifier
    Output:
        node_name, label_name, identifier_name (str) - return node, label, and identifier for output
    """
    node_name = output_names[0]
    label_name = output_names[1]
    identifier_name = output_names[2]

    return node_name, label_name, identifier_name

def map_output_names(file_args, output_names):
    """ Maps input file column name to output name. Returns dictionary with 
    Input:
        file_args (list) - list of arguments user provided for input file
        output_names (list) - list of output names to use for node, label, and identifier
    Output:
        mapNameDic (dict) - key: input column name (str) / value: output column name (str)
    """
    mapNameDic = {}
    input_names = file_args[1:]
    if input_names[1] != '':
        input_names[1] = 'label'

    for i in range(len(input_names)):
        if output_names[i] != '':
            mapNameDic[input_names[i]] = output_names[i]

    return mapNameDic

def get_info(file_args, output_names, cmd2template, args):
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
    template_nodes = args.template_nodes
    temporal = args.temporal
    templates = {}

    df = pd.DataFrame()

    node_name, label_name, id_name = get_outputNames(output_names)

    for inputfile_args in file_args:
        filename = inputfile_args[0]
        map_input2output_names = map_output_names(inputfile_args, output_names)

        db = pyfsdb.Fsdb(filename)

        login_index = False

        data = db.get_pandas(data_has_comment_chars=True)

        if inputfile_args[2] != '':
            data['label'] = inputfile_args[2]

        data = data.rename(columns=map_input2output_names)
        
        df = pd.concat([df,data]).reset_index(drop=True)

        db.close()

    df[node_name] = df[node_name].apply(lambda x: str([x]) if x[0]!="[" else x)

    if id_name != '':
        loggedInOnly = get_loggedInOnly(df,node_name,label_name,id_name)

        df2 = df.copy()[~df[id_name].isin(loggedInOnly)]
        df2 = df2[df2[node_name]!='[]']
        cmds = list(df2[node_name].unique())

        cmdIPsDic,sourceDic = get_cmdIPsDic(file_args,loggedInOnly,id_name,login_index, temporal)
    else:
        df2 = df.copy()
        df2 = df2[df2[node_name]!='[]']
        cmds = list(df2[node_name].unique())
        labelDic = get_labelDic(file_args,login_index,temporal)
        cmdIPsDic = None

    if cmd2template:
        templates,cmd2template = get_templates(cmd2template)
        if cmdIPsDic:
            unique_cmds,cmdIPsDic,templates = get_uniqueCmds(cmds,cmdIPsDic,{},templates,temporal)
        else:
            unique_cmds,labelDic,templates = get_uniqueCmds(cmds,cmdIPsDic,labelDic,templates,temporal)

        if template_nodes:
            unique_cmds2 = []
            for cmd in unique_cmds:
                if (cmd in [cmd for lst in templates[0].values() for cmd in lst]) or (cmd in [cmd for lst in templates[1].values() for cmd in lst]):
                    unique_cmds2.append(cmd)
            
            # templateCounts = calc_templateCount(templates,df,node_name)
            unique_cmds = unique_cmds2
    else:
        unique_cmds = cmds

    cmdToArray = {cmd[2:-2]:cmd for cmd in unique_cmds}
    unique_cmds = [cmd[2:-2] for cmd in unique_cmds]

    distDic = get_distances(unique_cmds)
    weightDic = get_weights(distDic)

    if cmdIPsDic:
        sourceDic.update({cmd:"+".join(list(cmdIPsDic[cmdToArray[cmd]].keys()))+"_"+node_name for cmd in unique_cmds})
    else:
        sourceDic = {cmd:"+".join(labelDic[cmdToArray[cmd]])+"_"+node_name for cmd in unique_cmds}

    return weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates

def get_loggedInOnly(df,node_name,label,id_name):
    """ Returns list of IP addresses that only logged in and did not run any commands 
    Input:
        df (Pandas DataFrame) - dataframe with info on IPs, commands
        node_name (str) - column name of node (eg. "command")
        label (str) - column name of label (eg. "source")
        id_name (str) - column name of identifier column (eg. "ip")
    Output:
        loggedInOnly (list) - IPs that only logged in
    """
    loggedIn = df[df[node_name]=='[]'][id_name].unique()
    loggedInOnly = []
    labels = df[label].unique()

    for ip in loggedIn:
        cmdsRun = []
        for label_name in labels:
            cmdsRun = cmdsRun + list(df[(df[id_name]==ip) & (df[label]==label_name)][node_name].unique())
        if cmdsRun == ['[]']:
            loggedInOnly.append(ip)

    return loggedInOnly

def get_cmdIPsDic(file_args,loggedInOnly,id_name,login_index,temporal):
    """ Returns dict that contains IP addresses that ran the command and from what source
    Input:
        file_args (list) - list of file argument lists with format [filename, node name, label name, identifier name]
        loggedInOnly (list) - list of IPs that only logged in
        id_name (str) - column name of identifier column (eg. "ip")
        login_index (False/int) - index of login successful
        temporal (None/list) - list of file numbers (period 2) for temporal analysis
    Output:
        cmdIPsDic (dict) - key: command (str) / value: dictionary with key: source (str) & value: IPs that ran command (list)
    """
    cmdIPsDic = {}
    labelIPDic = {}

    seenNodes = []

    file_num = 1
    for inputfile_args in file_args:
        filename, input_node_name, input_label_name, input_id_name = get_inputFile_args(inputfile_args)
        db = pyfsdb.Fsdb(filename)
        seenNodes = list(set(seenNodes))

        id_index = db.get_column_number(input_id_name)
        node_index = db.get_column_number(input_node_name)

        for row in db:
            ident = row[id_index] ## identifier (IP address)
            
            if ident in loggedInOnly: ## if IP only logged in, do not record
                continue

            ## check if login_index is provided. Skip over data where login_successful is false
            if (login_index != False) and (row[login_index] == 'False'):
                continue
            
            node = row[node_index]
            label = input_label_name
            ident_label = label

            if temporal: ## if doing temporal analysis
                if (file_num in temporal) and (node not in seenNodes): ## if looking at period 2 input file and node not seen
                    label = "new_"+label
                elif node in seenNodes: ## if node has been seen, continue
                    continue
                else:
                    seenNodes.append(node)
            
            if node[0]!="[":
                node = str([node])
            
            if node not in cmdIPsDic:
                cmdIPsDic[node] = {label: [ident]}
            else:
                if (label in cmdIPsDic[node]) and (ident not in cmdIPsDic[node][label]):
                        cmdIPsDic[node][label].append(ident)
                else:
                    cmdIPsDic[node][label] = [ident]

            if ident not in labelIPDic:
                labelIPDic[ident] = [ident_label]
            elif ident in labelIPDic and ident_label not in labelIPDic[ident]:
                labelIPDic[ident] = labelIPDic[ident] + [ident_label]

        db.close()
        file_num += 1

    sourceDic = {ip:"+".join(labelIPDic[ip])+"_"+id_name for ip in labelIPDic.keys()}

    return cmdIPsDic,sourceDic

def get_labelDic(file_args, login_index, temporal):
    """ Returns dict that maps node to list of labels node has
    Input:
        input_file (str) - FSDB input file
        node_name (str): column name of node (eg. "command")
        label_name (str): column name of label (eg. "source")
        temporal (None/list) - list of file numbers (period 2) for temporal analysis
    Output:
        labelDic (dict) - key: node (str) / value: list of labels
    """
    labelDic = {}

    seenNodes = []
    file_num = 1

    for inputfile_args in file_args:
        filename, input_node_name, input_label_name, input_id_name = get_inputFile_args(inputfile_args)
        db = pyfsdb.Fsdb(filename)
        seenNodes = list(set(seenNodes))

        node_index = db.get_column_number(input_node_name)

        for row in db:
            node = row[node_index]
            label = input_label_name

            if (login_index != False) and (row[login_index] == 'False'):
                continue

            if temporal:
                if (file_num in temporal) and (node not in seenNodes):
                    label = "new_"+label
                elif node in seenNodes:
                    continue
                else:
                    seenNodes.append(node)
            
            if node[0]!="[":
                node = str([node])
            
            if node not in labelDic:
                labelDic[node] = [label]
            else:
                if label not in labelDic[node]:
                    labelDic[node].append(label)
        
        db.close()
        file_num += 1

    return labelDic

def get_templates(cmd2templates):
    """ Gets list of commands that belong to each template and returns a dictionary
    Input:
        cmd2template (dict) - key: command (str) / value: template (tuple)
    Output:
        template2cmd (dict) - key: template (tuple) / value: commands that belong to the template (list)
    """
    templates = []
    cmd2template2 = {}

    for cmd2template in cmd2templates:
        template2cmd = {}
        if cmd2template: ## cmd2template is not None
            for cmd,basename in cmd2template.items():
                template = basename[2]
                cmd2template2[cmd[2:-2]] = template
                if template not in template2cmd:
                    template2cmd[template] = [cmd]
                else:
                    template2cmd[template] = template2cmd[template] + [cmd]
            templates.append(template2cmd)
        else:
            templates.append({})

    return templates,cmd2template2

def calc_templateCount(template2cmd,df,node_name):
    """ Counts how many of the templatized commands were run in the data and returns dict
    Input:
        template2cmd (dict) - key: template (tuple) / value: list of commands that match the template (list)
        df (pandas DataFrame) - DataFrame consisting of commands run in the full data
        node_name - name of node column
    Output:
        templateCounts (dict) - key: template (tuple) / value: number of commands that match template (int)
    """
    templateCounts = {}
    for template,cmds in template2cmd.items():
        count = df[node_name].isin(cmds).sum()
        templateCounts[template] = count

    return templateCounts

def map_cmd2templateCount(cmd2template,templateCounts,unique_cmds):
    unique_cmds = [cmd[2:-2] for cmd in unique_cmds]
    #cmd2templateCount = {cmd:templateCounts[cmd2template[cmd]] for cmd in unique_cmds}
    cmd2templateCount = {cmd:int(math.sqrt(5*templateCounts[cmd2template[cmd]])) for cmd in unique_cmds}
    return cmd2templateCount

def get_uniqueCmds(cmds,cmdIPsDic,labelDic,templates,temporal):
    """ Returns list of unique commands, dict that maps command to a dict that has source and IPs that ran command
    Input:
        cmds (list) - list of commands,
        cmdIPsDic (dict) - dict that maps command to a dictionary that has source and IPs that ran the command
        templates (list) - list of template2cmd dicts
        template2cmd (dict) - key: template (tuple) / value: commands that belong to the template (list)
        temporal (None/list) - list of file numbers (period 2) for temporal analysis
    Output:
        unique_cmds (list) - list of unique commands
        cmdIPsDic (dict) - maps command to a dict that has source and IPs that ran the command
    """
    unique_cmds = cmds
    cmdTemplateDic = {}
    first_cmds = []
    templatized_cmds = []

    if temporal:
        new_templates = find_new_templates(templates)
        to_remove = []
        to_add = []

    template2cmd = combine_templates(templates)
    # for template2cmd in templates:
    for template,cmds in template2cmd.items():
        all_cmds = cmds
        first_cmd = cmds[0]
        cmds = cmds[1:]
        
        if cmdIPsDic:
            if first_cmd not in cmdIPsDic:
                for i in range(len(cmds)):
                    if cmds[i] in cmdIPsDic:
                        first_cmd = cmds[i]
                        cmds = cmds[i+1:]
                        break

        cmdTemplateDic[first_cmd] = cmds

        first_cmds = first_cmds + [first_cmd]
        templatized_cmds = templatized_cmds + [cmds]

        ## if doing temporal analysis and template is a new template
        if temporal:
            if cmdIPsDic:
                labels2cmds = find_labelCmds(cmdIPsDic, all_cmds, 'ip')
            else:
                labels2cmds = find_labelCmds(labelDic, all_cmds, 'label')
            for label,cmds in labels2cmds.items():
                if 'new_' not in label and template in new_templates:
                    new_label = 'new_'+label ## add new to label to indicate new template
                    to_remove.append((cmds,label))
                    if cmdIPsDic:
                        ips = [ips for cmd in cmds for ips in cmdIPsDic[cmd][label]]
                    else:
                        ips = None
                    to_add.append((cmds, new_label, ips))
                elif 'new_' in label and template not in new_templates: ## if cmd new, but template is old >> remote 'new_' from label
                    new_label = label.replace('new_','')
                    to_remove.append((cmds,label))
                    if cmdIPsDic:
                        ips = [ips for cmd in cmds for ips in cmdIPsDic[cmd][label]]
                    else:
                        ips = None
                    to_add.append((cmds, new_label, ips))

    if temporal:
        if cmdIPsDic:
            for cmds,label in to_remove:
                for cmd in cmds:
                    cmdIPsDic[cmd].pop(label)
            for cmds,label,value in to_add:
                for cmd in cmds:
                    cmdIPsDic[cmd][label] = value
        else:
            for cmds,label in to_remove:
                for cmd in cmds:
                    labelDic[cmd].remove(label)
            for cmds,label,value in to_add:
                for cmd in cmds:
                    labelDic[cmd] = labelDic[cmd]+[label]

    # only keep 1st command of templatized commands as an example
    templatized_cmds = [cmd for cmd in templatized_cmds if cmd not in first_cmds]
    unique_cmds = [x for x in unique_cmds if x not in templatized_cmds]

    for cmd_key,cmds in cmdTemplateDic.items():
        for cmd in cmds:
            if (cmdIPsDic and cmd not in cmdIPsDic) or (labelDic and cmd not in labelDic):
                cmdTemplateDic[cmd_key].remove(cmd)

    if cmdIPsDic:       
        cmdIPsDic = update_cmdIPsDic(cmdIPsDic,cmdTemplateDic)
        unique_cmds = list(cmdIPsDic.keys())
        # print("Finished with cmdIPsDic")
        return unique_cmds,cmdIPsDic,template2cmd
    else:
        labelDic = update_labelDic(labelDic, cmdTemplateDic)
        unique_cmds = list(labelDic.keys())
        # print("Finished with labelDic")
        return unique_cmds,labelDic,template2cmd

def find_new_templates(templates):
    """
    
    """
    templates1 = templates[0].keys()
    templates2 = templates[1].keys()
    new_templates = [template for template in templates2 if template not in templates1]

    return new_templates

def combine_templates(templates):
    templateDic = {}
    for template2cmd in templates:
        for template,cmds in template2cmd.items():
            if template not in templateDic:
                templateDic[template] = cmds
            else:
                templateDic[template] = templateDic[template] + cmds

    templateDic = {template:sorted(set(cmds)) for template,cmds in templateDic.items()}
    
    return templateDic

def find_labelIPs(cmdIPsDic, cmds):
    labels2ips = {}

    for cmd in cmds:
        ipsDic = cmdIPsDic[cmd]
        labels = list(cmdIPsDic[cmd].keys())
        for label in labels:
            if label not in labels2ips:
                labels2ips[label] = ipsDic[label]
            else:
                labels2ips[label] = labels2ips[label] + ipsDic[label]
    
    return labels2ips

def find_labelCmds(cmdIPsDic, cmds, type):
    labels2cmds = {}

    for cmd in cmds:
        if type == 'ip':
            labels = list(cmdIPsDic[cmd].keys())
        else:
            labels = cmdIPsDic[cmd]
        for label in labels:
            if label not in labels2cmds:
                labels2cmds[label] = [cmd]
            else:
                labels2cmds[label] = labels2cmds[label] + [cmd]
    
    return labels2cmds

def get_allLabels(cmdIPsDic,cmds):
    all_labels = []
    for cmd in cmds:
        labels = list(cmdIPsDic[cmd].keys())
        all_labels = all_labels + labels

    all_labels = set(all_labels)

    return all_labels

def update_cmdIPsDic(cmdIPsDic,cmdTemplateDic):
    """ Returns updated cmdIPsDic dict so templatized command IPs include all IPs that ran cmd with templatized cmd
    Input:
        cmdIPsDic (dict) - maps command to a dictionary that has source and IPs that ran the command,
        cmdTemplateDic (dict) - maps first command of template to list of all other commands of same template
    Output: 
        cmdIPs (dict) - maps command to dict that contains source and IP addresses
    """
    template_cmdIPsDic = {}
    for cmd in cmdTemplateDic: ## for every template
        labels = cmdIPsDic[cmd].keys()
        IPsDic = {}

        for label in labels:
            IPs = [cmdIPsDic[cmd][label]] + [cmdIPsDic[cmds][label] for cmds in cmdTemplateDic[cmd] if label in cmdIPsDic[cmds]]
            IPs = [ip for lst in IPs for ip in lst]
            IPsDic[label] = IPs

        template_cmdIPsDic[cmd] = IPsDic
    
    cmdIPs = cmdIPsDic.copy()

    for cmd in template_cmdIPsDic:
        cmdIPs[cmd] = template_cmdIPsDic[cmd]
        for cmds in cmdTemplateDic[cmd]:
            if cmds in cmdIPs:
                del cmdIPs[cmds]
    
    return cmdIPs

def update_labelDic(labelDic, cmdTemplateDic):
    updated_labelDic = labelDic.copy()
    template_labelDic = {}

    for cmd in cmdTemplateDic:
        labels = [labelDic[cmd]] + [labelDic[cmds] for cmds in cmdTemplateDic[cmd]]
        labels = [label for lst in labels for label in lst]
        labels = list(set(labels))
        template_labelDic[cmd] = labels

    for cmd in template_labelDic:
        updated_labelDic[cmd] = template_labelDic[cmd]
        for cmds in cmdTemplateDic[cmd]:
            if cmds in updated_labelDic:
                del updated_labelDic[cmds]
        #updated_labelDic[cmd] = template_labelDic[cmd]
    
    return updated_labelDic

def get_distances(cmds):
    """ Returns dict that maps every pair of commands with their calculated distance
    Input:
        cmds (list) - list of commands
    Output:
        distDic (dict) - key: pair of commands (tuple) / value: distance between commands (float)
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
    Input:
        distDic (dict) - dict with distances of every pair of commands
    Output:
        weightDic (dict) - key: pair of commands (tuple) / value: weight between commands (float)
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

def draw_networkx(args,output_names,weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2templates,templates):
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
    threshold = args.edge_weight
    pos = args.position
    output_file = args.output_file[0]
    figsize = tuple([args.width,args.height])
    node_name, label_name, id_name = get_outputNames(output_names)

    edgeweight = [tuple(list(k)+[v]) for k,v in weightDic.items()]

    if args.top_k:
        k = args.top_k
        weighted_edges = get_topK_edges(k, edgeweight, k_edges=args.top_k_edges)
    else:
        weighted_edges = [x for x in edgeweight if x[2] > threshold]

    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)

    if (args.labels):
        labels = pickle.load(open(args.labels,"rb"))
        labels = add_newLabels(G,labels,cmd2templates,templates)
    else:
        labels = get_numberNodes(G)

    clusters = get_clusters(G)

    if cmdIPsDic:
        add_IPnodes(G,cmdToArray,cmdIPsDic)

    nodeTypeDic,colorslist = set_nodeColors(G,sourceDic,id_name)

    if (args.temporal):
        pos = plot_temporal_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,figsize=figsize,font_size=args.font_size,node_size=args.node_size)
    else:
        pos = plot_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,figsize=figsize,font_size=args.font_size,node_size=args.node_size)

    return G,weighted_edges,labels,clusters,pos

def get_topK_edges(k,edgeweight,k_edges=False):
    """ Finds top k highest weight edges and returns list of top k edges
    Input:
        k (int): number of edges to return
        edgeweight (list): list of edgeweights where each element is (node1, node2, weight)
    Output:
        topK_edges (list) - list of top K edges
    """
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

def add_newLabels(G,labels,cmd2templates,templates):
    nodes = G.nodes()
    num_labels = list(labels.values())
    new_labels = {}

    if templates != {}:
        templatized_cmds = [cmd[2:-2] for lst in templates.values() for cmd in lst]
    else:
        templatized_cmds = []
    
    i=max(num_labels)+1
    for node in nodes:
        if node in labels:
            new_labels[node] = labels[node]
        elif node in templatized_cmds:
            template = cmd2templates[node]
            temp_cmds = templates[template]

            cmd_in_prev_labels = False
            for arrayCmd in temp_cmds:
                cmd = arrayCmd[2:-2]
                if cmd in labels:
                    new_labels[node] = labels[cmd]
                    cmd_in_prev_labels = True
                    break
            if cmd_in_prev_labels == False:
                new_labels[node] = i
                i+=1
        else:
            new_labels[node] = i
            i+=1
        # else:
        #     while i in num_labels:
        #         i += 1
        #     labels[node] = i
        #     num_labels.append(i)
    
    return new_labels

def get_numberNodes(G):
    """ Returns dict that has nodes mapped to a unique integer label
    Input:
        G (NetworkX graph) - graph with IP and command nodes
        sourceDic (dict) - maps command nodes to source label
    Output:
        labels (dict) - maps node to labeled number
    """
    nodes = G.nodes()
    labels = {}

    i=0
    for node in nodes:
        labels[node] = i
        i += 1
    
    return labels

def add_IPnodes(G,cmdToArray,cmdIPsDic):
    """  Adds IP edges to command nodes
    Input:
        G (NetworkX graph) - graph to add IP edges to
        cmdToArray (dict) - maps command to array style command
        cmdIPsDic (dict) - maps command to dict that contains source and IP addresses
    """
    nodes = list(G.nodes())

    for node in nodes:
        cmd = cmdToArray[node]
        ips = get_IPs(cmd,cmdIPsDic)
        edges = [(node,ip) for ip in ips]
        G.add_edges_from(edges)

def get_IPs(cmd,dic):
    """ Finds and returns all IP addressses that ran a command
    Input:
        cmd (str) - command in array style
        dic (dict) - dict with commands mapped to source IPs
    Output:
        (list) unique IPs that ran command
    """
    ips = []

    for label,label_ips in dic[cmd].items():
        ips = ips + label_ips
    
    return list(set(ips))

def set_nodeColors(G,sourceDic,id_name):
    """ Sets source as an attribute for each command node, returns nodeTypeDic and color list
    Input:
        G (NetworkX graph) - graph with command nodes
        sourceDic (dict) - maps command nodes to source label
        id_name (str) - column name of identifier column
    Output:
        nodeTypeDic (dict) - maps node type with nodes
        colorslist (list) - list of color for node types
    """
    nx.set_node_attributes(G,name="source",values=sourceDic)
    sources = set(sorted(nx.get_node_attributes(G,"source").values()))

    if id_name != '':
        id_labels = [label for label in sources if id_name in label]
    else:
        id_labels = []

    source_labels = [label for label in sources if label not in id_labels]
    types = sorted(source_labels) + sorted(id_labels)

    colorslist = get_colors(types)

    nodeTypeDic = get_nodeTypeDic(types,G.nodes(),sourceDic,id_name)
    return nodeTypeDic,colorslist

def get_colors(types):
    """ Given all types of nodes to graph, return list of colors to use for each node type. Nodes that come from the same sources will have the same colors
    Input:
        types (list): list of node types to graph (will appear in legend)
    Output:
        colors_to_use (list): list of colors to use for each node
    """
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

def get_nodeTypeDic(types,nodes,sourceDic,id_name):
    """ Maps node label to list of nodes and returns dict
    Input:
        types (list) - list of node types/labels
        nodes (NetworkX nodes) - nodes of NetworkX graph
        sourceDic (dict) - maps command nodes to source label
        id_name (str) - column name of identifier column
    Output:
        nodeTypeDic (dict) - maps node type to list of nodes with that node type
    """
    nodeTypeDic = {nodetype:[] for nodetype in types}
    
    for node in nodes:
        source = sourceDic[node]
        nodeTypeDic[source] = nodeTypeDic[source]+[node]
            
    return nodeTypeDic

def plot_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,figsize=(12,8),font_size=10,node_size=350,ip_alpha=0.2,cmd_alpha=0.2,edge_alpha=0.2):
    """ Plots NetworkX graph and saves image to output file
    Input:
        G (NetworkX graph) - graph with IP and command nodes to graph
        output_file (str) - filename for network graph image
        labels (dict) - maps node to integer label
        colorslist (list) - list of node colors
        nodeTypeDic (dict) - maps node type to list of nodes
        id_name (str) - column name of identifier column (eg. "ip")
    """  
    fig,ax = plt.subplots(1,figsize=figsize)

    if not pos:
        pos=nx.spring_layout(G)
    else:
        pos=pickle.load(open(pos,"rb"))
        fixed_nodes = pos.keys()
        pos=nx.spring_layout(G,pos=pos,fixed=fixed_nodes)
        
    i=0
    for nodetype in nodeTypeDic:
        nodelist = nodeTypeDic[nodetype]
        color = colorslist[i]
        i+=1

        if id_name and id_name in nodetype:
            alpha=ip_alpha
            nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size,node_shape="^",node_color=color)
        else:
            alpha=cmd_alpha
            nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size,node_color=color)

    nx.draw_networkx_edges(G,pos=pos,alpha=edge_alpha)
    nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=font_size)
    ax.legend(scatterpoints=1, markerscale=0.75)

    ## remove black border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_file, dpi=300)

    return pos

def plot_temporal_networkx(G,pos,output_file,labels,colorslist,nodeTypeDic,id_name,figsize=(12,8),font_size=10,node_size=350,ip_alpha=0.1,cmd_alpha=0.2,edge_alpha=0.1):
    """ Plots NetworkX graph and saves image to output file
    Input:
        G (NetworkX graph) - graph with IP and command nodes to graph
        output_file (str) - filename for network graph image
        labels (dict) - maps node to integer label
        colorslist (list) - list of node colors
        nodeTypeDic (dict) - maps node type to list of nodes
        id_name (str) - column name of identifier column (eg. "ip")
    """  
    fig,ax = plt.subplots(1,figsize=figsize)

    if not pos:
        pos=nx.spring_layout(G)
    else:
        pos=pickle.load(open(pos,"rb"))
        fixed_nodes = pos.keys()
        pos=nx.spring_layout(G,pos=pos,fixed=fixed_nodes,k=0.3)
        
    i=0
    for nodetype in nodeTypeDic:
        nodelist = nodeTypeDic[nodetype]
        color = colorslist[i]
        i+=1

        if id_name and id_name in nodetype:
            alpha=ip_alpha
            nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size,node_shape="^",node_color=color)
        else:
            if 'new' in nodetype:
                alpha=0.4
            else:
                alpha=cmd_alpha
            nx.draw_networkx_nodes(G,pos=pos,nodelist=nodelist,ax=ax,\
                        label=nodetype,alpha=alpha,node_size=node_size,node_color=color)

    nx.draw_networkx_edges(G,pos=pos,alpha=edge_alpha)
    nx.draw_networkx_labels(G,pos=pos,labels=labels,font_size=font_size)
    ax.legend(scatterpoints=1, markerscale=0.75)

    ## remove black border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(output_file, dpi=300)

    return pos

def get_clusters(G):
    """ Finds clusters of commands in NetworkX graph. A cluster is considered to be nodes that are connected by an edge
    Input:
        G (NetworkX graph) - graph to find clusters
    Output:
        cmdToCluster (dict) - key: command node (str) / value: cluster ID (int)
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
  """Returns True if s is bash syntax (e.g., fi), standard command (e.g., wget), or looks like an argument (e.g., -c).
  """
  if s.startswith('-'): return True
  global _NIX_COMMANDS
#   if _NIX_COMMANDS is None:
#     _NIX_COMMANDS = set(open('../example/nix_commands.txt').read().split('\n'))
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
    return command2template

def main():
    args = parse_args()

    ## get list of file args and remove None input files args that were not specified
    file_args = list(filter(None,[args.input_file1, args.input_file2, args.input_file3, args.input_file4]))
    output_names = args.output_names

    check_fileArgs(file_args, output_names)

    if args.templatize:
        temporal = args.temporal
        stopwords = args.stopwords
        if stopwords == None:
            raise Exception('Stopwords file is missing. The stopwords file is required for templatization.')

        global _NIX_COMMANDS
        if _NIX_COMMANDS is None:
            _NIX_COMMANDS = set(open(stopwords).read().split('\n'))

        cmd2template = get_cmd2template(file_args,temporal)
    else:
        cmd2template = None

    weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates = get_info(file_args, output_names, cmd2template, args)
    G,weighted_edges,labels,clusters,pos = draw_networkx(args,output_names,weightDic,cmdIPsDic,sourceDic,cmdToArray,cmd2template,templates)

    ## save NetworkX graph position file to pickle file
    if (args.position_file):
        pickle.dump(pos, open(args.position_file, "wb" ))

    ## save labels dict to pickle file
    if (args.labels_file):
        pickle.dump(labels, open(args.labels_file, "wb" ))

    ## create edge list to FSDB file
    if (args.edge_list):
        if (args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            print("Graphing template nodes...")
            outh = pyfsdb.Fsdb(out_file=args.edge_list)
            outh.out_column_names=['cluster_id', 'weight', 'node1_id', 'node2_id', 'node1', 'node2', 'template1', 'template2']
            for cmd1,cmd2,weight in weighted_edges:
                cluster_id = clusters[cmd1]
                num1 = labels[cmd1]
                num2 = labels[cmd2]
                template1 = cmd2template[cmd1]
                template2 = cmd2template[cmd2]
                outh.append([cluster_id,round(weight,3),num1,num2,cmd1,cmd2,template1,template2])
            outh.close()
        else:
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
        if (args.template_nodes): ## if template nodes are being graphed, produce edge list that includes templates
            outh = pyfsdb.Fsdb(out_file=args.cluster_list)
            outh.out_column_names=['cluster_id','command','template']
            for cmd,cluster_id in clusters.items():
                template = cmd2template[cmd]
                outh.append([cluster_id,cmd,template])
            outh.close()
        else:
            outh = pyfsdb.Fsdb(out_file=args.cluster_list)
            outh.out_column_names=['cluster_id','command']
            for cmd,cluster_id in clusters.items():
                outh.append([cluster_id,cmd])
            outh.close()

if __name__ == "__main__":
    main()